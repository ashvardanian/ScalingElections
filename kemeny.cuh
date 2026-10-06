/**
 *  @brief Exact Kemeny-Young consensus ranking over a pairwise preference matrix.
 *  @file kemeny.cuh
 *  @author Ash Vardanian
 *  @date August 24, 2026
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "types.cuh"

#pragma region Kemeny

/**
 *  @brief Votes each candidate loses to every subset of the others.
 *
 *  One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
 *  two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
 */
template <tally_arithmetic arithmetic_type_>
struct kemeny_sums {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const low_bits;
    std::size_t const low_states;
    std::size_t const high_states;
    std::vector<arithmetic_t> low;
    std::vector<arithmetic_t> high;

    template <typename stored_count_type_>
    kemeny_sums(strided_matrix<stored_count_type_ const, candidate_index_t> preferences)
        : low_bits(preferences.rows / 2), low_states(std::size_t {1} << low_bits),
          high_states(std::size_t {1} << (preferences.rows - low_bits)), low(preferences.rows * low_states, 0),
          high(preferences.rows * high_states, 0) {
        candidate_index_t const num_candidates = preferences.rows;
        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            accumulate(std::span(low).subspan(candidate * low_states, low_states), preferences, 0, candidate);
            accumulate(std::span(high).subspan(candidate * high_states, high_states), preferences, low_bits, candidate);
        }
    }

    /** Votes that preferred @p candidate to every member of @p subset. */
    arithmetic_t against(candidate_index_t candidate, subset_mask_t subset) const noexcept {
        return low[candidate * low_states + (subset & (low_states - 1))] +
               high[candidate * high_states + (subset >> low_bits)];
    }

  private:
    // Ignore diagonal entries so they cannot overflow a narrowed subset sum.
    template <typename stored_count_type_>
    static void accumulate(std::span<arithmetic_t> row,
                           strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                           candidate_index_t first_opponent, candidate_index_t candidate) noexcept {
        subset_mask_t const states = row.size();
        candidate_index_t opponent = first_opponent;
        for (subset_mask_t bit = 1; bit < states; bit <<= 1, opponent++)
            for (subset_mask_t subset = bit; subset < states; subset++)
                if (subset & bit)
                    row[subset] = row[subset ^ bit] +
                                  arithmetic_t(opponent == candidate ? 0 : preferences(candidate, opponent));
    }
};

/** Whether one or several complete orderings achieve the minimum disagreement. */
enum class ranking_multiplicity_t : std::uint8_t {
    /** Exactly one complete ordering is optimal. */
    unique_k,
    /** Several complete orderings are optimal, possibly with the same winner. */
    multiple_k
};

/** An optimal Kemeny-Young ordering, its disagreement score, and its tie metadata. */
struct kemeny_solution_t {
    /** One optimal ordering, best candidate first. */
    std::vector<candidate_index_t> ranking;
    /** Minimum total weight of pairwise preferences contradicted by the ordering. */
    voter_weight_t score = 0;
    /** All candidates that can rank first in an optimal ordering. */
    std::vector<candidate_index_t> winners;
    /** Whether the complete optimal ordering is unique. */
    ranking_multiplicity_t multiplicity = ranking_multiplicity_t::unique_k;
};

/**
 *  @brief The widest field whose middle binomial, the size of the widest layer, fits `binomial_count_t`.
 *
 *  Steps along the middle of Pascal's triangle, where `C(n + 1, n / 2)` follows `C(n, n / 2)` for
 *  even @p n and `C(n + 1, (n + 1) / 2)` doubles `C(n, n / 2)` for odd. Memory decides the rest at runtime.
 */
constexpr candidate_index_t kemeny_max_candidates_() noexcept {
    // Twice as wide as the binomial type, so the step past its range is still exact.
    std::uint64_t middle = 1;
    for (candidate_index_t n = 0;; n++) {
        std::uint64_t const next = n % 2 == 0 ? middle * (n + 1) / (n / 2 + 1) : middle * 2;
        if (next > std::numeric_limits<binomial_count_t>::max()) return n;
        middle = next;
    }
}

/** Every ordering contradicts at most the heavier side of each pair, so their sum bounds any score. */
template <typename stored_count_type_>
inline voter_weight_t kemeny_score_bound_(strided_matrix<stored_count_type_ const, candidate_index_t> preferences) {
    candidate_index_t const n = preferences.rows;
    if (n < 1 || n > kemeny_max_candidates_())
        throw std::invalid_argument(std::format("Kemeny is exact from 1 to {} candidates", kemeny_max_candidates_()));
    saturated<voter_weight_t> bound = 0;
    for (candidate_index_t row = 0; row < n; row++)
        for (candidate_index_t column = row + 1; column < n; column++)
            bound += std::max(preferences(row, column), preferences(column, row));
    return static_cast<voter_weight_t>(bound);
}

/**
 *  @brief Resolves automatic Kemeny arithmetic from the score bound, and validates an explicit request.
 *
 *  Each exact width reserves its maximum for unreachable subsets, so the bound must stay strictly
 *  below it. Refusing here happens before any exponential table is allocated.
 */
template <typename stored_count_type_>
inline score_type_t kemeny_resolve_score_type(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                              score_type_t requested) {
    voter_weight_t const bound = kemeny_score_bound_(preferences);
    voter_weight_t limit = 0;
    switch (requested) {
    case score_type_t::auto_k:
        if (bound < std::numeric_limits<std::uint32_t>::max()) return score_type_t::uint32_k;
        if (bound < std::numeric_limits<std::uint64_t>::max()) return score_type_t::uint64_k;
        return score_type_t::saturated64_k;
    case score_type_t::saturated64_k: return requested;
    case score_type_t::uint64_k: limit = std::numeric_limits<std::uint64_t>::max(); break;
    case score_type_t::uint32_k: limit = std::numeric_limits<std::uint32_t>::max(); break;
    case score_type_t::uint16_k: limit = std::numeric_limits<std::uint16_t>::max(); break;
    }
    if (bound >= limit) throw std::overflow_error("Kemeny score bound exceeds the selected arithmetic type");
    return requested;
}

/**
 *  @brief Pascal's triangle up to @p num_candidates , which is all the colex unranking needs.
 *
 *  Entry `[upper * (num_candidates + 1) + lower]` is `C(upper, lower)`.
 */
inline std::vector<binomial_count_t> kemeny_binomials_(candidate_index_t num_candidates) {
    candidate_index_t const stride = num_candidates + 1;
    std::vector<binomial_count_t> binomials(static_cast<std::size_t>(stride) * stride, 0u);
    for (candidate_index_t upper = 0; upper <= num_candidates; upper++) {
        binomials[upper * stride] = 1;
        for (candidate_index_t lower = 1; lower <= upper; lower++)
            binomials[upper * stride + lower] = binomials[(upper - 1) * stride + lower] +
                                                binomials[(upper - 1) * stride + lower - 1];
    }
    return binomials;
}

/**
 *  @brief The mask that colex @p rank names among the subsets seating @p seated candidates.
 *
 *  One binomial per candidate, so a layer costs exactly as many ranks as it has subsets and never
 *  has to be listed out: at 33 candidates the widest layer would be 8.7 GiB of masks.
 */
SCALING_ELECTIONS_HOST_DEVICE inline subset_mask_t kemeny_unrank_colex_( //
    strided_matrix<binomial_count_t const, candidate_index_t> binomials, candidate_index_t seated,
    binomial_count_t rank) noexcept {

    candidate_index_t const num_candidates = binomials.rows - 1;
    subset_mask_t subset = 0;
    candidate_index_t remaining = seated;
    for (candidate_index_t candidate = num_candidates; remaining != 0 && candidate != 0;) {
        candidate--;
        binomial_count_t const below = binomials(candidate, remaining);
        if (rank < below) continue;
        rank -= below;
        subset |= subset_mask_t {1} << candidate;
        remaining--;
    }
    return subset;
}

/**
 *  @brief The next mask of the same population count, which is the next colex rank in its layer.
 *
 *  Colex order within a layer is the numeric order of the masks, so one Gosper step walks it.
 *  @p subset must have at least one bit set, which every layer from the first one does.
 */
inline subset_mask_t kemeny_next_colex_(subset_mask_t subset) noexcept {
    subset_mask_t const lowest = subset & (~subset + 1);
    subset_mask_t const rippled = subset + lowest;
    return rippled | ((rippled ^ subset) >> (2 + std::countr_zero(subset)));
}

/**
 *  @brief Walks a completed cost table back out into a ranking, best candidate first.
 *
 *  Every backend fills the same table, so recovering the ranking in one place is also what
 *  keeps their tie-breaking identical. Every partial score stays within the validated bound, so
 *  the exact widths need no saturation here.
 */
template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
inline kemeny_solution_t kemeny_trace_(kemeny_sums<arithmetic_type_> const& sums,
                                       std::span<std::type_identity_t<arithmetic_type_> const> costs,
                                       strided_matrix<stored_count_type_ const, candidate_index_t> preferences) {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const num_candidates = preferences.rows;
    subset_mask_t const everyone = costs.size() - 1;

    arithmetic_t const optimum = costs[everyone];
    if (optimum == std::numeric_limits<arithmetic_t>::max())
        throw std::overflow_error("Kemeny optimum reaches the overflow sentinel");
    kemeny_solution_t solution;
    solution.score = static_cast<voter_weight_t>(optimum);
    solution.ranking.reserve(num_candidates);

    for (subset_mask_t subset = everyone; subset;) {
        subset_mask_t const before = subset;
        unsigned choices = 0;
        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            subset_mask_t const bit = subset_mask_t {1} << candidate;
            if (!(before & bit)) continue;
            subset_mask_t const rest = before ^ bit;
            if (costs[before] != costs[rest] + sums.against(candidate, rest)) continue;
            if (++choices == 1) {
                solution.ranking.push_back(candidate);
                subset = rest;
            }
        }
        // Every subset was filled from one of its members, so one of them has to match back.
        if (subset == before) throw std::runtime_error("The Kemeny cost table disagrees with its own sums");
        if (choices > 1) solution.multiplicity = ranking_multiplicity_t::multiple_k;
    }
    std::reverse(solution.ranking.begin(), solution.ranking.end());
    for (candidate_index_t candidate = 0; candidate < num_candidates; ++candidate) {
        arithmetic_t score = costs[everyone ^ (subset_mask_t {1} << candidate)];
        for (candidate_index_t other = 0; other < num_candidates; ++other)
            if (other != candidate) score += arithmetic_t(preferences(other, candidate));
        if (score == optimum) solution.winners.push_back(candidate);
    }
    return solution;
}

/**
 *  @brief Fills the exact Kemeny-Young subset cost table across an @b OpenMP team.
 *
 *  One barrier per population count, because subsets of one width depend only on the width below
 *  and never on each other. Numeric order is also a valid topological order, but it admits no
 *  block wider than one: clearing a bit always lands on the layer directly below.
 *
 *  A thread unranks the first subset of its slice and steps through the rest, since colex order
 *  within a layer is the numeric order of the masks.
 *
 *  @param[in] preferences The pairwise preference matrix.
 *  @param[in] sums The split subset sums built from the same matrix.
 *  @param[in] cancelled Polled between layers, aborting the fill once it reads non-zero.
 */
template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
inline std::vector<arithmetic_type_> compute_kemeny_costs_cpu(
    strided_matrix<stored_count_type_ const, candidate_index_t> preferences, kemeny_sums<arithmetic_type_> const& sums,
    cancellation_t cancelled = nullptr) {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const num_candidates = preferences.rows;
    std::vector<binomial_count_t> const binomials = kemeny_binomials_(num_candidates);
    candidate_index_t const binomials_stride = num_candidates + 1;
    auto const binomial_view = square_view<binomial_count_t const, candidate_index_t>(
        binomials.data(), binomials_stride, binomials_stride);
    std::size_t const states = std::size_t {1} << num_candidates;

    // Entry `subset` is the least disagreement achievable seating those candidates in the
    // leading places, counting only the pairs inside it.
    std::vector<arithmetic_t> costs;
    try {
        costs.resize(states);
    }
    catch (std::bad_alloc const&) {
        throw std::runtime_error(std::format("Kemeny over {} candidates wants {} MiB of host memory", num_candidates,
                                             (states * sizeof(arithmetic_t)) >> 20));
    }

    for (candidate_index_t seated = 1; seated <= num_candidates; seated++) {
        throw_if_cancelled_(cancelled);
        binomial_count_t const layer_states = binomial_view(num_candidates, seated);
        // Every subset costs the same, so each thread takes one equal contiguous slice of the layer.
#pragma omp parallel
        {
            binomial_count_t const threads = openmp_threads_count_();
            binomial_count_t const thread = openmp_thread_index_();
            binomial_count_t const share = layer_states / threads;
            binomial_count_t const spare = layer_states % threads;
            binomial_count_t const first_rank = share * thread + std::min(thread, spare);
            binomial_count_t const last_rank = first_rank + share + (thread < spare);
            subset_mask_t subset = kemeny_unrank_colex_(binomial_view, seated, first_rank);
            for (binomial_count_t rank = first_rank; rank < last_rank; rank++, subset = kemeny_next_colex_(subset)) {
                arithmetic_t best = std::numeric_limits<arithmetic_t>::max();
                for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
                    subset_mask_t const bit = subset_mask_t {1} << candidate;
                    if (!(subset & bit)) continue;
                    // Seating this candidate last within the subset costs the votes that preferred
                    // it to each of the others.
                    subset_mask_t const rest = subset ^ bit;
                    best = std::min<arithmetic_t>(best, costs[rest] + sums.against(candidate, rest));
                }
                costs[subset] = best;
            }
        }
    }

    return costs;
}

#pragma endregion Kemeny

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_GPU)

/** The split-mask tables of `kemeny_sums<arithmetic_type_>` as a kernel addresses them. */
template <tally_arithmetic arithmetic_type_>
struct kemeny_device_sums {
    using arithmetic_t = arithmetic_type_;
    /** Votes against each candidate, indexed by the mask's low half. */
    strided_matrix<arithmetic_t const, candidate_index_t> low;
    /** Votes against each candidate, indexed by the mask's high half. */
    strided_matrix<arithmetic_t const, candidate_index_t> high;
    /** How many of the mask's low bits the low table covers. */
    candidate_index_t low_bits;

    /** Votes that preferred @p candidate to every member of @p subset. */
    __forceinline__ __device__ arithmetic_t against(candidate_index_t candidate, subset_mask_t subset) const noexcept {
        return low(candidate, subset & (low.columns - 1)) + high(candidate, subset >> low_bits);
    }
};

/** Scores seating one candidate last, keeping whichever of the two orderings costs less. */
template <tally_arithmetic arithmetic_type_>
__forceinline__ __device__ arithmetic_type_ kemeny_relax_(arithmetic_type_ rest, arithmetic_type_ against,
                                                          arithmetic_type_ best) noexcept {
    arithmetic_type_ const score = rest + against;
    return score < best ? score : best;
}

/**
 *  @brief Fills every subset that seats @p seated candidates, one thread to a subset.
 *
 *  Threads take colex ranks that @p binomials unranks into a mask, so a layer costs exactly as
 *  many threads as it has subsets. Clearing a bit drops the population count by one, and that
 *  layer is complete before this one launches.
 */
template <tally_arithmetic arithmetic_type_>
__global__ void kemeny_layer_cuda_(                                      //
    kemeny_device_sums<arithmetic_type_> sums, arithmetic_type_* costs,  //
    strided_matrix<binomial_count_t const, candidate_index_t> binomials, //
    candidate_index_t seated, binomial_count_t layer_states) {
    using arithmetic_t = arithmetic_type_;

    // The widest layer fits the rank type with room to spare, so rounding up to whole blocks cannot wrap.
    binomial_count_t const rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= layer_states) return;
    candidate_index_t const num_candidates = binomials.rows - 1;
    subset_mask_t const subset = kemeny_unrank_colex_(binomials, seated, rank);

    arithmetic_t best = std::numeric_limits<arithmetic_t>::max();
    for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
        subset_mask_t const bit = subset_mask_t {1} << candidate;
        if (!(subset & bit)) continue;
        subset_mask_t const rest = subset ^ bit;
        best = kemeny_relax_(costs[rest], sums.against(candidate, rest), best);
    }
    costs[subset] = best;
}

/**
 *  @brief Fills the exact Kemeny-Young subset cost table on a CUDA or @b HIP device.
 *
 *  One launch per population count, because subsets of one width depend only on the width below
 *  and never on each other. The answer is the host's, entry for entry: the same integer sums
 *  reach the same minimum whatever order the threads take them in.
 *
 *  @param[in] preferences The pairwise preference matrix.
 *  @param[in] sums The split subset sums built from the same matrix.
 */
template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
inline managed_vector<arithmetic_type_> compute_kemeny_costs_gpu(
    strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
    kemeny_sums<arithmetic_type_> const& sums) {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const num_candidates = preferences.rows;
    std::vector<binomial_count_t> const host_binomials = kemeny_binomials_(num_candidates);
    std::size_t const states = std::size_t {1} << num_candidates;
    std::size_t const sums_states = sums.low.size() + sums.high.size();
    candidate_index_t const binomials_stride = num_candidates + 1;
    std::size_t const wanted_bytes = (states + sums_states) * sizeof(arithmetic_t) +
                                     host_binomials.size() * sizeof(binomial_count_t);

    // Managed memory oversubscribes rather than failing, so an unaffordable table is refused here.
    std::size_t free_bytes = 0;
    [[maybe_unused]] std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        throw std::runtime_error("Failed to query device memory");
    if (wanted_bytes > free_bytes)
        throw std::runtime_error(
            std::format("Kemeny over {} candidates wants {} MiB of device memory, of which {} MiB is free",
                        num_candidates, wanted_bytes >> 20, free_bytes >> 20));

    managed_vector<arithmetic_t> costs(states);
    managed_vector<arithmetic_t> device_sums(sums_states);
    managed_vector<binomial_count_t> binomials(host_binomials.size());

    std::copy(sums.low.begin(), sums.low.end(), device_sums.data());
    std::copy(sums.high.begin(), sums.high.end(), device_sums.data() + sums.low.size());
    std::copy(host_binomials.begin(), host_binomials.end(), binomials.data());

    kemeny_device_sums<arithmetic_t> const device_view {
        strided_view<arithmetic_t const, candidate_index_t>(device_sums.data(), num_candidates, sums.low_states,
                                                            sums.low_states),
        strided_view<arithmetic_t const, candidate_index_t>(device_sums.data() + sums.low.size(), num_candidates,
                                                            sums.high_states, sums.high_states),
        sums.low_bits,
    };
    auto const binomial_view = square_view<binomial_count_t const, candidate_index_t>(
        binomials.data(), binomials_stride, binomials_stride);

    // Zeroes the empty subset the first layer reads, and faults the table's pages onto the device.
    if (cudaMemset(costs.data(), 0, states * sizeof(arithmetic_t)) != cudaSuccess)
        throw std::runtime_error("Failed to clear device memory");

    int minimum_grid = 0;
    int block_size = 0;
    if (cudaOccupancyMaxPotentialBlockSize(&minimum_grid, &block_size, kemeny_layer_cuda_<arithmetic_t>) != cudaSuccess)
        throw std::runtime_error("Failed to size the Kemeny layer kernel");
    for (candidate_index_t seated = 1; seated <= num_candidates; seated++) {
        binomial_count_t const layer_states = host_binomials[num_candidates * binomials_stride + seated];
        binomial_count_t const blocks = divide_round_up(layer_states, static_cast<binomial_count_t>(block_size));
        kemeny_layer_cuda_<<<blocks, block_size>>>(device_view, costs.data(), binomial_view, seated, layer_states);

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA operations did not complete successfully");

    return costs;
}

#else

template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
inline std::vector<arithmetic_type_> compute_kemeny_costs_gpu(
    strided_matrix<stored_count_type_ const, candidate_index_t>, kemeny_sums<arithmetic_type_> const&) {
    throw std::runtime_error("This build has no GPU support compiled in");
}

#endif // defined(SCALING_ELECTIONS_WITH_GPU)

#pragma endregion CUDA

/** Finds one optimum and its tie metadata in @p arithmetic_type_ , without retaining the exponential table. */
template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
inline kemeny_solution_t compute_kemeny_ranking(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                                backend_t backend, cancellation_t cancelled = nullptr) {
    using arithmetic_t = arithmetic_type_;
    kemeny_sums<arithmetic_t> const sums(preferences);
    switch (backend) {
    case backend_t::cpu_k: {
        std::vector<arithmetic_t> const costs = compute_kemeny_costs_cpu<arithmetic_t>(preferences, sums, cancelled);
        return kemeny_trace_(sums, {costs.data(), costs.size()}, preferences);
    }
    case backend_t::gpu_k: {
        auto const costs = compute_kemeny_costs_gpu<arithmetic_t>(preferences, sums);
        return kemeny_trace_(sums, {costs.data(), costs.size()}, preferences);
    }
    }
    throw std::invalid_argument("backend must be cpu or gpu");
}
