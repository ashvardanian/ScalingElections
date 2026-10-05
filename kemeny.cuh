/**
 *  @brief Exact Kemeny-Young consensus ranking over a pairwise preference matrix.
 *  @file kemeny.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
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
template <typename arithmetic_type_ = default_kemeny_arithmetic_t>
struct kemeny_sums {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const low_bits;
    std::size_t const low_states;
    std::size_t const high_states;
    std::vector<arithmetic_t> low;
    std::vector<arithmetic_t> high;

    template <typename stored_count_type_>
    kemeny_sums(stored_count_type_ const* preferences, candidate_index_t num_candidates)
        : low_bits(num_candidates / 2), low_states(std::size_t {1} << low_bits),
          high_states(std::size_t {1} << (num_candidates - low_bits)), low(num_candidates * low_states, 0),
          high(num_candidates * high_states, 0) {
        using stored_count_t = stored_count_type_;
        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            stored_count_t const* row = preferences + candidate * num_candidates;
            accumulate(low.data() + candidate * low_states, low_states, row, 0, candidate);
            accumulate(high.data() + candidate * high_states, high_states, row + low_bits, low_bits, candidate);
        }
    }

    /** Votes that preferred @p candidate to every member of @p subset. */
    arithmetic_t against(candidate_index_t candidate, std::size_t subset) const noexcept {
        return low[candidate * low_states + (subset & (low_states - 1))] +
               high[candidate * high_states + (subset >> low_bits)];
    }

  private:
    // Ignore diagonal entries so they cannot overflow a narrowed subset sum.
    template <typename stored_count_type_>
    static void accumulate(arithmetic_t* row, std::size_t states, stored_count_type_ const* votes,
                           candidate_index_t first_opponent, candidate_index_t candidate) noexcept {
        candidate_index_t opponent = first_opponent;
        for (std::size_t bit = 1; bit < states; bit <<= 1, votes++, opponent++)
            for (std::size_t subset = bit; subset < states; subset++)
                if (subset & bit) row[subset] = row[subset ^ bit] + arithmetic_t(opponent == candidate ? 0 : *votes);
    }
};

/** An exact Kemeny-Young consensus ranking and the disagreement it achieves. */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t>
struct kemeny_solution {
    using arithmetic_t = arithmetic_type_;
    std::vector<candidate_index_t> ranking;
    arithmetic_t score = 0;
    std::vector<candidate_index_t> winners;
    bool unique = true;
};

/** The widest field an exact table can address, bounded by device memory rather than by time. */
constexpr candidate_index_t kemeny_max_candidates_k = 33;

/** Refuses a field the exact table cannot address, before anything has been allocated for it. */
inline void kemeny_require_supported_width_(candidate_index_t num_candidates) {
    if (num_candidates < 1 || num_candidates > kemeny_max_candidates_k)
        throw std::invalid_argument("Kemeny is exact from 1 to " + std::to_string(kemeny_max_candidates_k) +
                                    " candidates");
}

template <typename stored_count_type_>
inline std::uint64_t kemeny_score_bound_(stored_count_type_ const* preferences, candidate_index_t n) {
    kemeny_require_supported_width_(n);
    saturated<std::uint64_t> bound = 0;
    for (candidate_index_t row = 0; row < n; row++)
        for (candidate_index_t column = row + 1; column < n; column++)
            bound += std::max(preferences[row * n + column], preferences[column * n + row]);
    return static_cast<std::uint64_t>(bound);
}

template <typename arithmetic_type_, typename stored_count_type_>
inline void kemeny_require_score_range_(stored_count_type_ const* preferences, candidate_index_t n) {
    using arithmetic_t = arithmetic_type_;
    static_assert(std::numeric_limits<arithmetic_t>::is_integer && !std::numeric_limits<arithmetic_t>::is_signed);
    static_assert(sizeof(arithmetic_t) <= sizeof(std::uint64_t));
    if constexpr (!std::is_same_v<arithmetic_t, saturated<std::uint64_t>>)
        if (kemeny_score_bound_(preferences, n) >= std::numeric_limits<arithmetic_t>::max())
            throw std::overflow_error("Kemeny score bound exceeds the selected arithmetic type");
}

/**
 *  @brief Pascal's triangle up to @p num_candidates , which is all the colex unranking needs.
 *
 *  Entry `[upper * (num_candidates + 1) + lower]` is `C(upper, lower)`, and `C(33, 16)` is the
 *  widest value a 32-bit slot has to hold.
 */
inline std::vector<std::uint32_t> kemeny_binomials_(candidate_index_t num_candidates) {
    candidate_index_t const stride = num_candidates + 1;
    std::vector<std::uint32_t> binomials(static_cast<std::size_t>(stride) * stride, 0u);
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
SCALING_ELECTIONS_HOST_DEVICE inline std::size_t kemeny_unrank_colex_( //
    std::uint32_t const* binomials, candidate_index_t num_candidates, candidate_index_t seated,
    std::uint64_t rank) noexcept {

    std::size_t subset = 0;
    candidate_index_t remaining = seated;
    for (candidate_index_t candidate = num_candidates; remaining != 0 && candidate != 0;) {
        candidate--;
        std::uint32_t const below = binomials[candidate * (num_candidates + 1) + remaining];
        if (rank < below) continue;
        rank -= below;
        subset |= std::size_t {1} << candidate;
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
inline std::size_t kemeny_next_colex_(std::size_t subset) noexcept {
    std::size_t const lowest = subset & (~subset + 1);
    std::size_t const rippled = subset + lowest;
    return rippled | ((rippled ^ subset) >> (2 + std::countr_zero(subset)));
}

/** Ranks one host thread unranks for once, then walks by Gosper step rather than by binomials. */
constexpr std::int64_t kemeny_chunk_size_k = 1024;

/**
 *  @brief Walks a completed cost table back out into a ranking, best candidate first.
 *
 *  Every backend fills the same table, so recovering the ranking in one place is also what
 *  keeps their tie-breaking identical.
 */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t, typename stored_count_type_>
inline kemeny_solution<arithmetic_type_> kemeny_trace_(kemeny_sums<arithmetic_type_> const& sums,
                                                       arithmetic_type_ const* costs, candidate_index_t num_candidates,
                                                       stored_count_type_ const* preferences) {
    using arithmetic_t = arithmetic_type_;
    std::size_t const states = std::size_t {1} << num_candidates;

    kemeny_solution<arithmetic_t> solution;
    solution.score = costs[states - 1];
    if (solution.score == std::numeric_limits<arithmetic_t>::max())
        throw std::overflow_error("Kemeny optimum reaches the overflow sentinel");
    solution.ranking.reserve(num_candidates);

    for (std::size_t subset = states - 1; subset;) {
        std::size_t const before = subset;
        unsigned choices = 0;
        for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
            std::size_t const bit = std::size_t {1} << candidate;
            if (!(before & bit)) continue;
            std::size_t const rest = before ^ bit;
            if (costs[before] != costs[rest] + sums.against(candidate, rest)) continue;
            if (++choices == 1) {
                solution.ranking.push_back(candidate);
                subset = rest;
            }
        }
        // Every subset was filled from one of its members, so one of them has to match back.
        if (subset == before) throw std::runtime_error("The Kemeny cost table disagrees with its own sums");
        if (choices > 1) solution.unique = false;
    }
    std::reverse(solution.ranking.begin(), solution.ranking.end());
    for (candidate_index_t candidate = 0; candidate < num_candidates; ++candidate) {
        arithmetic_t score = costs[(states - 1) ^ (std::size_t {1} << candidate)];
        for (candidate_index_t other = 0; other < num_candidates; ++other)
            if (other != candidate) score += arithmetic_t(preferences[other * num_candidates + candidate]);
        if (score == solution.score) solution.winners.push_back(candidate);
    }
    return solution;
}

/**
 *  @brief Determines the exact Kemeny-Young consensus ranking across an @b OpenMP team.
 *
 *  One barrier per population count, because subsets of one width depend only on the width below
 *  and never on each other. Numeric order is also a valid topological order, but it admits no
 *  block wider than one: clearing a bit always lands on the layer directly below.
 *
 *  A thread unranks the first subset of its chunk and steps through the rest, since colex order
 *  within a layer is the numeric order of the masks.
 *
 *  @param preferences The pairwise preference matrix.
 *  @param num_candidates The number of candidates, which the score table bounds to 33.
 */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t, typename stored_count_type_>
inline kemeny_solution<arithmetic_type_> compute_kemeny_ranking_cpu(stored_count_type_ const* preferences,
                                                                    candidate_index_t num_candidates) {
    using arithmetic_t = arithmetic_type_;
    kemeny_require_supported_width_(num_candidates);
    kemeny_require_score_range_<arithmetic_t>(preferences, num_candidates);

    kemeny_sums<arithmetic_t> const sums(preferences, num_candidates);
    std::vector<std::uint32_t> const binomials = kemeny_binomials_(num_candidates);
    candidate_index_t const binomials_stride = num_candidates + 1;
    std::size_t const states = std::size_t {1} << num_candidates;

    // Entry `subset` is the least disagreement achievable seating those candidates in the
    // leading places, counting only the pairs inside it.
    std::vector<arithmetic_t> costs;
    try {
        costs.resize(states);
    }
    catch (std::bad_alloc const&) {
        throw std::runtime_error("Kemeny over " + std::to_string(num_candidates) + " candidates wants " +
                                 std::to_string((states * sizeof(arithmetic_t)) >> 20) + " MiB of host memory");
    }

    for (candidate_index_t seated = 1; seated <= num_candidates; seated++) {
        std::int64_t const layer_states = binomials[num_candidates * binomials_stride + seated];
        std::int64_t const chunks = (layer_states + kemeny_chunk_size_k - 1) / kemeny_chunk_size_k;
#pragma omp parallel for schedule(static)
        for (std::int64_t chunk = 0; chunk < chunks; chunk++) {
            std::int64_t const first_rank = chunk * kemeny_chunk_size_k;
            std::int64_t const last_rank = std::min(first_rank + kemeny_chunk_size_k, layer_states);
            std::size_t subset = kemeny_unrank_colex_(binomials.data(), num_candidates, seated,
                                                      static_cast<std::uint64_t>(first_rank));
            for (std::int64_t rank = first_rank; rank < last_rank; rank++, subset = kemeny_next_colex_(subset)) {
                arithmetic_t best = std::numeric_limits<arithmetic_t>::max();
                for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
                    std::size_t const bit = std::size_t {1} << candidate;
                    if (!(subset & bit)) continue;
                    // Seating this candidate last within the subset costs the votes that preferred
                    // it to each of the others.
                    std::size_t const rest = subset ^ bit;
                    best = std::min<arithmetic_t>(best, costs[rest] + sums.against(candidate, rest));
                }
                costs[subset] = best;
            }
        }
    }

    return kemeny_trace_(sums, costs.data(), num_candidates, preferences);
}

#pragma endregion Kemeny

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_CUDA)

/** Threads per block for one layer of the subset dynamic program. */
constexpr std::uint32_t kemeny_block_size_k = 256;

/** Device memory the tables must leave behind for the driver and everyone else. */
constexpr std::size_t kemeny_device_headroom_k = std::size_t {1} << 30;

/** The split-mask tables of `kemeny_sums<arithmetic_type_>` as a kernel addresses them. */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t>
struct kemeny_device_sums {
    using arithmetic_t = arithmetic_type_;
    /** Votes against each candidate, indexed by the mask's low half. */
    arithmetic_t const* low;
    /** Votes against each candidate, indexed by the mask's high half. */
    arithmetic_t const* high;
    /** How many of the mask's low bits the low table covers. */
    candidate_index_t low_bits;
    /** The low table's per-candidate stride. */
    std::uint32_t low_states;
    /** The high table's per-candidate stride. */
    std::uint32_t high_states;

    /** Votes that preferred @p candidate to every member of @p subset. */
    __forceinline__ __device__ arithmetic_t against(candidate_index_t candidate, std::size_t subset) const noexcept {
        return low[candidate * low_states + (subset & (low_states - 1))] +
               high[candidate * high_states + (subset >> low_bits)];
    }
};

/** Scores seating one candidate last, keeping whichever of the two orderings costs less. */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t>
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
template <typename arithmetic_type_ = default_kemeny_arithmetic_t>
__global__ void kemeny_layer_gpu(                                       //
    kemeny_device_sums<arithmetic_type_> sums, arithmetic_type_* costs, //
    std::uint32_t const* binomials, candidate_index_t num_candidates,   //
    candidate_index_t seated, std::uint64_t layer_states) {
    using arithmetic_t = arithmetic_type_;

    std::uint64_t const rank = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (rank >= layer_states) return;
    std::size_t const subset = kemeny_unrank_colex_(binomials, num_candidates, seated, rank);

    arithmetic_t best = std::numeric_limits<arithmetic_t>::max();
    for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
        std::size_t const bit = std::size_t {1} << candidate;
        if (!(subset & bit)) continue;
        std::size_t const rest = subset ^ bit;
        best = kemeny_relax_(costs[rest], sums.against(candidate, rest), best);
    }
    costs[subset] = best;
}

/**
 *  @brief Determines the exact Kemeny-Young consensus ranking on a CUDA or @b HIP device.
 *
 *  One launch per population count, because subsets of one width depend only on the width below
 *  and never on each other. The answer is the host's, entry for entry: the same integer sums
 *  reach the same minimum whatever order the threads take them in.
 *
 *  @param preferences The pairwise preference matrix.
 *  @param num_candidates The number of candidates, which free device memory bounds further.
 */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t, typename stored_count_type_>
inline kemeny_solution<arithmetic_type_> compute_kemeny_ranking_gpu(stored_count_type_ const* preferences,
                                                                    candidate_index_t num_candidates) {
    using arithmetic_t = arithmetic_type_;
    kemeny_require_supported_width_(num_candidates);
    kemeny_require_score_range_<arithmetic_t>(preferences, num_candidates);

    kemeny_sums<arithmetic_t> const sums(preferences, num_candidates);
    std::vector<std::uint32_t> const host_binomials = kemeny_binomials_(num_candidates);
    std::size_t const states = std::size_t {1} << num_candidates;
    std::size_t const sums_states = sums.low.size() + sums.high.size();
    candidate_index_t const binomials_stride = num_candidates + 1;
    std::size_t const wanted_bytes = (states + sums_states) * sizeof(arithmetic_t) +
                                     host_binomials.size() * sizeof(std::uint32_t);

    // Managed memory oversubscribes rather than failing, so an unaffordable table is refused here.
    std::size_t free_bytes = 0;
    [[maybe_unused]] std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
        throw std::runtime_error("Failed to query device memory");
    if (wanted_bytes + kemeny_device_headroom_k > free_bytes)
        throw std::runtime_error("Kemeny over " + std::to_string(num_candidates) + " candidates wants " +
                                 std::to_string(wanted_bytes >> 20) + " MiB of device memory, of which " +
                                 std::to_string(free_bytes >> 20) + " MiB is free");

    managed_vector<arithmetic_t> costs(states);
    managed_vector<arithmetic_t> device_sums(sums_states);
    managed_vector<std::uint32_t> binomials(host_binomials.size());

    std::copy(sums.low.begin(), sums.low.end(), device_sums.data());
    std::copy(sums.high.begin(), sums.high.end(), device_sums.data() + sums.low.size());
    std::copy(host_binomials.begin(), host_binomials.end(), binomials.data());

    kemeny_device_sums<arithmetic_t> const device_view {
        device_sums.data(),
        device_sums.data() + sums.low.size(),
        sums.low_bits,
        static_cast<std::uint32_t>(sums.low_states),
        static_cast<std::uint32_t>(sums.high_states),
    };

    // Zeroes the empty subset the first layer reads, and faults the table's pages onto the device.
    if (cudaMemset(costs.data(), 0, states * sizeof(arithmetic_t)) != cudaSuccess)
        throw std::runtime_error("Failed to clear device memory");

    for (candidate_index_t seated = 1; seated <= num_candidates; seated++) {
        std::uint64_t const layer_states = host_binomials[num_candidates * binomials_stride + seated];
        std::uint64_t const blocks = (layer_states + kemeny_block_size_k - 1) / kemeny_block_size_k;
        kemeny_layer_gpu<<<static_cast<unsigned int>(blocks), kemeny_block_size_k>>>( //
            device_view, costs.data(), binomials.data(), num_candidates, seated, layer_states);

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA operations did not complete successfully");

    return kemeny_trace_(sums, costs.data(), num_candidates, preferences);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion CUDA

#pragma region Dispatch

/** Runs exact ranking with the chosen arithmetic type on the requested device. */
template <typename arithmetic_type_ = default_kemeny_arithmetic_t, typename stored_count_type_>
inline kemeny_solution<arithmetic_type_> compute_kemeny_ranking_typed(stored_count_type_ const* preferences,
                                                                      candidate_index_t num_candidates,
                                                                      backend_t backend = backend_t::cpu_k) {
    using arithmetic_t = arithmetic_type_;
    kemeny_require_supported_width_(num_candidates);
    if (backend == backend_t::cpu_k) return compute_kemeny_ranking_cpu<arithmetic_t>(preferences, num_candidates);
#if defined(SCALING_ELECTIONS_WITH_CUDA)
    return compute_kemeny_ranking_gpu<arithmetic_t>(preferences, num_candidates);
#else
    throw std::runtime_error("This build has no GPU support compiled in");
#endif
}

template <typename stored_count_type_>
inline kemeny_solution<> compute_kemeny_ranking(stored_count_type_ const* preferences, candidate_index_t num_candidates,
                                                backend_t backend = backend_t::cpu_k,
                                                score_type_t score_type = score_type_t::auto_k) {
    if (score_type == score_type_t::auto_k) {
        auto const bound = kemeny_score_bound_(preferences, num_candidates);
        score_type = bound < std::numeric_limits<std::uint32_t>::max()   ? score_type_t::uint32_k
                     : bound < std::numeric_limits<std::uint64_t>::max() ? score_type_t::uint64_k
                                                                         : score_type_t::saturated64_k;
    }
    auto run = [&](auto score) {
        auto solution = compute_kemeny_ranking_typed<decltype(score)>(preferences, num_candidates, backend);
        return kemeny_solution<> {std::move(solution.ranking), static_cast<std::uint64_t>(solution.score),
                                  std::move(solution.winners), solution.unique};
    };
    switch (score_type) {
    case score_type_t::uint16_k: return run(std::uint16_t {});
    case score_type_t::uint32_k: return run(std::uint32_t {});
    case score_type_t::uint64_k: return run(std::uint64_t {});
    case score_type_t::saturated64_k: return run(saturated<std::uint64_t> {0});
    default: throw std::invalid_argument("Invalid score type");
    }
}

#pragma endregion Dispatch
