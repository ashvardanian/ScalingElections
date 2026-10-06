/**
 *  @brief Turns a pairwise preference matrix into the winning-votes graph the Schulze method walks.
 *  @file ballots.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "types.cuh"

/**
 *  @brief Seeds the strongest-paths matrix with the direct pairwise wins.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[out] graph The matrix of strongest paths, whose leading block this fills.
 *
 *  A cell keeps its vote count only where it strictly beats the opposite direction, so ties, losses,
 *  and the diagonal all read as zero, the identity of the max-min semiring.
 */
template <typename stored_count_type_, tally_arithmetic arithmetic_type_>
inline void winning_votes_graph(strided_matrix<stored_count_type_ const> preferences,
                                strided_matrix<arithmetic_type_> graph) {

    candidate_index_t const num_candidates = preferences.extent(0);
#pragma omp parallel for collapse(2)
    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++)
            graph(row, column) = row != column && preferences(row, column) > preferences(column, row)
                                     ? static_cast<arithmetic_type_>(preferences(row, column))
                                     : arithmetic_type_ {0};
}

/** Which graph the strongest-paths sweep closes over. */
enum class seed_graph_t : std::uint8_t {
    /** Winning votes, the variant Schulze runs on here. */
    winning_votes_k,
    /** Positive margins, which is what Split Cycle is defined on. */
    positive_margins_k,
};

/**
 *  @brief Seeds the strongest-paths matrix with each pair's positive margin.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[out] graph The matrix of strongest paths, whose leading block this fills.
 *
 *  A cell keeps its margin only where the pair is won, so at most one direction of any pair is
 *  ever non-zero. That is what lets the unchanged max-min kernel close over margins.
 */
template <typename stored_count_type_, tally_arithmetic arithmetic_type_>
inline void positive_margins_graph(strided_matrix<stored_count_type_ const> preferences,
                                   strided_matrix<arithmetic_type_> graph) {

    candidate_index_t const num_candidates = preferences.extent(0);
#pragma omp parallel for collapse(2)
    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++) {
            stored_count_type_ const forward = preferences(row, column);
            stored_count_type_ const backward = preferences(column, row);
            graph(row, column) = row != column && forward > backward ? static_cast<arithmetic_type_>(forward - backward)
                                                                     : 0;
        }
}

/** Seeds the matrix with whichever graph the method is defined on. */
template <typename stored_count_type_, tally_arithmetic arithmetic_type_>
inline void seed_graph(strided_matrix<stored_count_type_ const> preferences, strided_matrix<arithmetic_type_> graph,
                       seed_graph_t which) {
    if (which == seed_graph_t::positive_margins_k) positive_margins_graph(preferences, graph);
    else winning_votes_graph(preferences, graph);
}

/**
 *  @brief Names the candidates nobody defeats, which is the Split Cycle winning set.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] margin_paths Widest paths already closed over the positive-margin graph.
 *  @return The undefeated candidates, in increasing order.
 *
 *  Holliday and Pacuit's Lemma 3.17: one candidate defeats another when its margin is positive
 *  and exceeds the widest path running back the other way. The set is irresolute by Theorem 4.7,
 *  so it can name several winners where Schulze names one.
 */
template <typename stored_count_type_, tally_arithmetic arithmetic_type_>
inline std::vector<candidate_index_t> select_split_cycle_winners(strided_matrix<stored_count_type_ const> preferences,
                                                                 strided_matrix<arithmetic_type_> margin_paths) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::vector<candidate_index_t> undefeated;
    for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
        bool defeated = false;
        for (candidate_index_t rival = 0; rival < num_candidates && !defeated; rival++) {
            if (rival == candidate) continue;
            stored_count_type_ const forward = preferences(rival, candidate);
            stored_count_type_ const backward = preferences(candidate, rival);
            if (forward <= backward) continue;
            defeated = (forward - backward) > margin_paths(candidate, rival);
        }
        if (!defeated) undefeated.push_back(candidate);
    }
    return undefeated;
}

#pragma region Tally

/**
 *  @brief Adds one chunk of complete rankings into an existing pairwise matrix.
 *
 *  @param[in] rankings One chunk of complete rankings, best candidate first.
 *  @param[inout] preferences The matrix to accumulate into, which the caller zeroes first.
 *
 *  Accumulating rather than assigning is what lets an electorate arrive in chunks instead of
 *  having to sit in memory all at once.
 */
inline void tally_ballots_cpu(ballots_t rankings, uint32_matrix_t preferences) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::size_t const num_ballots = rankings.extent(0);
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
#pragma omp parallel
    {
        std::vector<std::uint32_t> private_counts(cells, 0);
#pragma omp for schedule(static)
        for (std::ptrdiff_t ballot = 0; ballot < static_cast<std::ptrdiff_t>(num_ballots); ballot++) {
            candidate_index_t const* const ranking = &rankings(ballot, 0);
            for (candidate_index_t position = 0; position + 1 < num_candidates; position++) {
                candidate_index_t const preferred = ranking[position];
                for (candidate_index_t later = position + 1; later < num_candidates; later++)
                    private_counts[std::size_t(preferred) * num_candidates + ranking[later]]++;
            }
        }
#pragma omp critical
        for (candidate_index_t row = 0; row < num_candidates; row++)
            for (candidate_index_t column = 0; column < num_candidates; column++)
                preferences(row, column) += private_counts[std::size_t(row) * num_candidates + column];
    }
}

#if defined(SCALING_ELECTIONS_WITH_CUDA)

/** Threads per block for the tally, chosen so one block still fits a private matrix in shared memory. */
constexpr std::uint32_t tally_block_size_k = 256;

/** Shared bytes one tally block needs: a private matrix, plus the ballot each warp is staging. */
inline std::size_t tally_shared_bytes(candidate_index_t num_candidates) noexcept {
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
    std::size_t const staged = static_cast<std::size_t>(tally_block_size_k / warp_size_k) * num_candidates;
    return cells * sizeof(std::uint32_t) + staged * sizeof(candidate_index_t);
}

/**
 *  @brief Accumulates each block's ballots into a shared-memory matrix, then merges once.
 *
 *  @param[in] rankings One chunk of complete rankings, best candidate first.
 *  @param[inout] preferences The matrix to accumulate into.
 *
 *  One warp takes one ballot, reads it once into shared memory, and splits its pairs across the
 *  lanes. Privatizing keeps the scattered increments in shared memory, where an atomic costs a
 *  fraction of the global one it replaces, and leaves one global atomic per cell per block.
 */
__global__ void tally_ballots_cuda_(ballots_t rankings, uint32_matrix_t preferences) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;

    extern __shared__ std::uint32_t counters[];
    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x) counters[cell] = 0;
    __syncthreads();

    std::uint32_t const warps_per_block = blockDim.x / warp_size_k;
    std::uint32_t const warp = threadIdx.x / warp_size_k;
    std::uint32_t const lane = threadIdx.x % warp_size_k;
    candidate_index_t* const staged = reinterpret_cast<candidate_index_t*>(counters + cells) + warp * num_candidates;

    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * warps_per_block;
    std::size_t const first = static_cast<std::size_t>(blockIdx.x) * warps_per_block + warp;
    for (std::size_t ballot = first; ballot < rankings.extent(0); ballot += stride) {
        for (candidate_index_t position = lane; position < num_candidates; position += warp_size_k)
            staged[position] = rankings(ballot, position);
        __syncwarp();

        for (candidate_index_t position = 0; position + 1 < num_candidates; position++) {
            candidate_index_t const preferred = staged[position];
            for (candidate_index_t later = position + 1 + lane; later < num_candidates; later += warp_size_k)
                atomic_add_relaxed<atomic_scope_t::block_k>(&counters[preferred * num_candidates + staged[later]], 1u);
        }
        __syncwarp();
    }
    __syncthreads();

    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
        if (counters[cell])
            atomic_add_relaxed<atomic_scope_t::device_k>(&preferences(cell / num_candidates, cell % num_candidates),
                                                         counters[cell]);
}

/** Whether a private matrix of this edge fits one block's shared memory on this device. */
inline bool tally_fits_shared_memory(candidate_index_t num_candidates, cudaDeviceProp const& device_properties) {
    return tally_shared_bytes(num_candidates) <= static_cast<std::size_t>(device_properties.sharedMemPerBlock);
}

/** Whether a kernel can already read @p pointer, which spares the chunk its staging copy. */
inline bool device_can_read(void const* pointer) noexcept {
    cudaPointerAttributes attributes {};
    if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess) {
        [[maybe_unused]] cudaError_t const cleared = cudaGetLastError();
        return false;
    }
    return attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged;
}

/**
 *  @brief Adds one chunk of complete rankings into an existing matrix, tallied on the device.
 *
 *  @param[in] rankings One chunk of complete rankings, best candidate first.
 *  @param[inout] preferences The matrix to accumulate into.
 *
 *  Rankings the device can already reach are read where they lie, so a caller streaming chunks
 *  through one managed buffer pays for the staging once rather than once per chunk.
 */
inline void tally_ballots_gpu(ballots_t rankings, uint32_matrix_t preferences) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::size_t const num_ballots = rankings.extent(0);
    cudaDeviceProp device_properties;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&device_properties, device) != cudaSuccess)
        throw std::runtime_error("No CUDA devices available");
    if (!tally_fits_shared_memory(num_candidates, device_properties))
        throw std::invalid_argument("A tally over " + std::to_string(num_candidates) +
                                    " candidates needs more shared memory than a block can hold");

    std::size_t const ballot_stride = rankings.stride(0);
    std::size_t const entries = num_ballots ? (num_ballots - 1) * ballot_stride + num_candidates : 0;
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
    bool const staging_needed = !device_can_read(rankings.data_handle());
    managed_vector<candidate_index_t> staging(staging_needed ? entries : 0);
    managed_vector<std::uint32_t> device_counts(cells);
    std::ranges::fill(device_counts, std::uint32_t {0});

    // A driver copy lands the chunk on the device outright, where a host loop would leave the
    // kernel to fault every page in one at a time.
    if (staging_needed && cudaMemcpy(staging.data(), rankings.data_handle(), staging.size() * sizeof(candidate_index_t),
                                     cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to copy ballots to the device");

    ballots_t const device_rankings = staging_needed ? strided_view<candidate_index_t const, std::size_t>(
                                                           staging.data(), num_ballots, num_candidates, ballot_stride)
                                                     : rankings;
    uint32_matrix_t const device_preferences = square_view(device_counts.data(), num_candidates, num_candidates);

    std::size_t const warps_per_block = tally_block_size_k / warp_size_k;
    std::size_t const wanted_blocks = divide_round_up(num_ballots, warps_per_block);
    unsigned int const blocks = static_cast<unsigned int>(std::min<std::size_t>(wanted_blocks, 65535));
    tally_ballots_cuda_<<<blocks, tally_block_size_k, tally_shared_bytes(num_candidates)>>>(device_rankings,
                                                                                            device_preferences);

    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA operations did not complete successfully");

    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++)
            preferences(row, column) += device_counts.data()[std::size_t(row) * num_candidates + column];
}

/** Adds one chunk of complete rankings into an existing matrix, on whichever processor was named. */
inline void tally_ballots(ballots_t rankings, uint32_matrix_t preferences, backend_t backend) {
    if (backend == backend_t::cpu_k) return tally_ballots_cpu(rankings, preferences);
    return tally_ballots_gpu(rankings, preferences);
}

#else

/** Adds one chunk of complete rankings into an existing matrix; a CPU-only build has one processor. */
inline void tally_ballots(ballots_t rankings, uint32_matrix_t preferences, backend_t backend) {
    if (backend == backend_t::gpu_k)
        throw std::runtime_error("This build has no CUDA support, so `gpu` is unavailable");
    tally_ballots_cpu(rankings, preferences);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion Tally

enum class unranked_t : std::uint8_t { unknown_k = 0, worse_k = 1 };
/** @brief The comparison counted for each ordered candidate pair. */
enum class pairwise_relation_t : std::uint8_t {
    /** @brief The row candidate is strictly preferred. */
    preference_k = 0,
    /** @brief The candidates share a rank. */
    indifference_k = 1,
    /** @brief The ballot leaves the comparison unspecified. */
    unknown_k = 2,
    /** @brief All three comparison matrices in preference, indifference, unknown order. */
    all_k = 3
};

SCALING_ELECTIONS_HOST_DEVICE inline pairwise_relation_t classify_relation(std::uint64_t left, std::uint64_t right,
                                                                           unranked_t unranked) {
    constexpr auto missing = std::numeric_limits<std::uint64_t>::max();
    if (unranked == unranked_t::unknown_k && (left == missing || right == missing))
        return pairwise_relation_t::unknown_k;
    if (left == right) return pairwise_relation_t::indifference_k;
    return pairwise_relation_t::preference_k;
}

/** @brief Borrowed CSR ballot inputs, valid for the duration of a tally call. */
struct ragged_ballots {
    /** @brief Flat candidate IDs, distinct within each ballot. */
    candidate_index_t const* candidates;
    /** @brief Ballot boundaries, including the terminal entry offset. */
    ballot_offset_t const* offsets;
    /** @brief Equal labels tie; null uses entry positions as strict ranks. */
    rank_label_t const* ranks;
    /** @brief Integer voter weights; null assigns unit weight. */
    voter_weight_t const* weights;
    /** @brief Number of rows in the ballot input. */
    std::size_t num_ballots;
    /** @brief Size of the candidate universe, including omitted candidates. */
    candidate_index_t num_candidates;
    /** @brief Omission policy used when per-voter policies are absent. */
    unranked_t unranked;
    /** @brief Optional per-voter omission policies. */
    unranked_t const* policies;

    SCALING_ELECTIONS_HOST_DEVICE unranked_t unranked_at(std::size_t ballot) const noexcept {
        return policies ? policies[ballot] : unranked;
    }
};

template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_cpu(ragged_ballots ballots, strided_matrix<arithmetic_type_> preferences,
                              pairwise_relation_t relation = pairwise_relation_t::preference_k) {
    using arithmetic_t = arithmetic_type_;
    std::size_t const n = ballots.num_candidates;
    std::size_t const cells = checked_product(n, n);
    std::size_t const output_cells = checked_product(cells, relation == pairwise_relation_t::all_k ? 3 : 1);
#pragma omp parallel
    {
        std::vector<arithmetic_t> counts(output_cells, arithmetic_t {0});
        std::vector<std::uint8_t> present(n);
        std::vector<std::uint64_t> labels(relation == pairwise_relation_t::preference_k ? 0 : n);
#pragma omp for schedule(static)
        for (std::size_t ballot = 0; ballot < ballots.num_ballots; ++ballot) {
            auto const first = ballots.offsets[ballot];
            auto const last = ballots.offsets[ballot + 1];
            arithmetic_t const weight(ballots.weights ? ballots.weights[ballot] : 1);
            if (weight == arithmetic_t {0}) continue;
            if (relation != pairwise_relation_t::preference_k) {
                std::ranges::fill(labels, std::numeric_limits<std::uint64_t>::max());
                for (auto entry = first; entry < last; ++entry)
                    labels[ballots.candidates[entry]] = ballots.ranks ? ballots.ranks[entry] : entry - first;
                for (std::size_t candidate = 0; candidate < n; ++candidate)
                    for (std::size_t opponent = 0; opponent < n; ++opponent) {
                        if (candidate == opponent) continue;
                        auto const actual = classify_relation(labels[candidate], labels[opponent],
                                                              ballots.unranked_at(ballot));
                        if (actual == pairwise_relation_t::preference_k && labels[candidate] >= labels[opponent])
                            continue;
                        if (relation == pairwise_relation_t::all_k || relation == actual) {
                            auto const plane = relation == pairwise_relation_t::all_k ? static_cast<std::size_t>(actual)
                                                                                      : 0;
                            counts[plane * cells + candidate * n + opponent] += weight;
                        }
                    }
                continue;
            }
            if (ballots.unranked_at(ballot) == unranked_t::worse_k) {
                std::ranges::fill(present, false);
                for (auto entry = first; entry < last; ++entry) present[ballots.candidates[entry]] = true;
            }
            for (auto entry = first; entry < last; ++entry) {
                std::size_t const preferred = ballots.candidates[entry];
                auto const rank = ballots.ranks ? ballots.ranks[entry] : entry - first;
                for (auto other = first; other < last; ++other)
                    if (rank < (ballots.ranks ? ballots.ranks[other] : other - first))
                        counts[preferred * n + ballots.candidates[other]] += weight;
                if (ballots.unranked_at(ballot) == unranked_t::worse_k)
                    for (candidate_index_t other = 0; other < n; ++other)
                        if (!present[other]) counts[preferred * n + other] += weight;
            }
        }
#pragma omp critical
        for (std::size_t cell = 0; cell < output_cells; ++cell) preferences(cell / n, cell % n) += counts[cell];
    }
}

#if defined(SCALING_ELECTIONS_WITH_CUDA)
template <tally_arithmetic arithmetic_type_>
__global__ void tally_ragged_ballots_cuda_(ragged_ballots ballots, arithmetic_type_* counts,
                                           pairwise_relation_t relation, std::uint64_t* rank_scratch) {
    using arithmetic_t = arithmetic_type_;
    extern __shared__ std::uint32_t present[];
    std::size_t const n = ballots.num_candidates;
    std::size_t const words = divide_round_up(n, 32);
    for (std::size_t ballot = blockIdx.x; ballot < ballots.num_ballots; ballot += gridDim.x) {
        auto const first = ballots.offsets[ballot];
        auto const last = ballots.offsets[ballot + 1];
        arithmetic_t const weight(ballots.weights ? ballots.weights[ballot] : 1);
        if (weight == arithmetic_t {0}) continue;
        if (relation != pairwise_relation_t::preference_k) {
            auto* labels = rank_scratch + std::size_t(blockIdx.x) * n;
            for (std::size_t candidate = threadIdx.x; candidate < n; candidate += blockDim.x)
                labels[candidate] = std::numeric_limits<std::uint64_t>::max();
            __syncthreads();
            for (auto entry = first + threadIdx.x; entry < last; entry += blockDim.x)
                labels[ballots.candidates[entry]] = ballots.ranks ? ballots.ranks[entry] : entry - first;
            __syncthreads();
            for (std::size_t candidate = threadIdx.x; candidate < n; candidate += blockDim.x)
                for (std::size_t opponent = 0; opponent < n; ++opponent) {
                    if (candidate == opponent) continue;
                    auto const actual = classify_relation(labels[candidate], labels[opponent],
                                                          ballots.unranked_at(ballot));
                    if (actual == pairwise_relation_t::preference_k && labels[candidate] >= labels[opponent]) continue;
                    if (relation == pairwise_relation_t::all_k || relation == actual) {
                        auto const plane = relation == pairwise_relation_t::all_k ? static_cast<std::size_t>(actual)
                                                                                  : 0;
                        atomic_add_relaxed<atomic_scope_t::device_k>(counts + (plane * n + candidate) * n + opponent,
                                                                     weight);
                    }
                }
            __syncthreads();
            continue;
        }
        if (ballots.unranked_at(ballot) == unranked_t::worse_k) {
            for (std::size_t word = threadIdx.x; word < words; word += blockDim.x) present[word] = 0;
            __syncthreads();
            for (auto entry = first + threadIdx.x; entry < last; entry += blockDim.x) {
                auto const candidate = ballots.candidates[entry];
                atomic_or_relaxed<atomic_scope_t::block_k>(present + candidate / 32,
                                                           std::uint32_t {1} << (candidate % 32));
            }
            __syncthreads();
        }
        for (auto entry = first + threadIdx.x; entry < last; entry += blockDim.x) {
            std::size_t const preferred = ballots.candidates[entry];
            auto const rank = ballots.ranks ? ballots.ranks[entry] : entry - first;
            for (auto other = first; other < last; ++other)
                if (rank < (ballots.ranks ? ballots.ranks[other] : other - first))
                    atomic_add_relaxed<atomic_scope_t::device_k>(counts + preferred * n + ballots.candidates[other],
                                                                 weight);
            if (ballots.unranked_at(ballot) == unranked_t::worse_k)
                for (candidate_index_t other = 0; other < n; ++other)
                    if (!(present[other / 32] & (std::uint32_t {1} << (other % 32))))
                        atomic_add_relaxed<atomic_scope_t::device_k>(counts + preferred * n + other, weight);
        }
        __syncthreads();
    }
}

template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_gpu(ragged_ballots ballots, strided_matrix<arithmetic_type_> preferences,
                              pairwise_relation_t relation = pairwise_relation_t::preference_k) {
    using arithmetic_t = arithmetic_type_;
    if (!ballots.num_ballots) return;
    std::size_t const entries = ballots.offsets[ballots.num_ballots];
    std::size_t const cells = checked_product(checked_product(ballots.num_candidates, ballots.num_candidates),
                                              relation == pairwise_relation_t::all_k ? 3 : 1);
    std::size_t const shared_bytes = relation == pairwise_relation_t::preference_k &&
                                             (ballots.policies || ballots.unranked == unranked_t::worse_k)
                                         ? divide_round_up(std::size_t(ballots.num_candidates), 32) * 4
                                         : 0;
    int device = 0;
    cudaDeviceProp properties;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("No CUDA devices available");
    if (shared_bytes > static_cast<std::size_t>(properties.sharedMemPerBlock))
        throw std::invalid_argument("Ballot presence bitmap exceeds device shared memory");
    managed_vector<candidate_index_t> candidates(entries);
    managed_vector<rank_label_t> ranks(ballots.ranks ? entries : 0);
    managed_vector<ballot_offset_t> offsets(ballots.num_ballots + 1);
    managed_vector<voter_weight_t> weights(ballots.weights ? ballots.num_ballots : 0);
    managed_vector<unranked_t> policies(ballots.policies ? ballots.num_ballots : 0);
    managed_vector<arithmetic_t> counts(cells);
    auto copy = [](auto& destination, auto const* source) {
        if (!destination.empty() &&
            cudaMemcpy(destination.data(), source, checked_product(destination.size(), sizeof(*source)),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Failed to copy ballots to device");
    };
    copy(candidates, ballots.candidates);
    copy(offsets, ballots.offsets);
    copy(ranks, ballots.ranks);
    copy(weights, ballots.weights);
    copy(policies, ballots.policies);
    if (cudaMemset(counts.data(), 0, checked_product(cells, sizeof(arithmetic_t))) != cudaSuccess)
        throw std::runtime_error("Failed to clear device memory");
    ragged_ballots const device_ballots {candidates.data(),
                                         offsets.data(),
                                         ballots.ranks ? ranks.data() : nullptr,
                                         ballots.weights ? weights.data() : nullptr,
                                         ballots.num_ballots,
                                         ballots.num_candidates,
                                         ballots.unranked,
                                         ballots.policies ? policies.data() : nullptr};
    std::size_t const rank_bytes = checked_product(ballots.num_candidates, sizeof(std::uint64_t));
    std::size_t const block_limit = relation == pairwise_relation_t::preference_k
                                        ? 65535
                                        : std::max<std::size_t>(1, (8 * 1024 * 1024) / rank_bytes);
    unsigned const blocks = static_cast<unsigned>(std::min(ballots.num_ballots, block_limit));
    managed_vector<std::uint64_t> rank_scratch(
        relation == pairwise_relation_t::preference_k ? 0 : checked_product(blocks, ballots.num_candidates));
    tally_ragged_ballots_cuda_<<<blocks, tally_block_size_k, shared_bytes>>>(device_ballots, counts.data(), relation,
                                                                             rank_scratch.data());
    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA tally did not complete");
    for (std::size_t cell = 0; cell < cells; ++cell)
        preferences(cell / ballots.num_candidates, cell % ballots.num_candidates) += counts[cell];
}
#endif

template <tally_arithmetic arithmetic_type_>
inline void tally_ballots(ragged_ballots ballots, strided_matrix<arithmetic_type_> preferences, backend_t backend,
                          pairwise_relation_t relation = pairwise_relation_t::preference_k) {
    if (backend == backend_t::cpu_k) return tally_ballots_cpu(ballots, preferences, relation);
#if defined(SCALING_ELECTIONS_WITH_CUDA)
    return tally_ballots_gpu(ballots, preferences, relation);
#else
    throw std::runtime_error("This build has no GPU support compiled in");
#endif
}
