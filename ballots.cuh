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
template <typename score_type_>
inline void winning_votes_graph(const_matrix_t preferences, strided_matrix<score_type_> graph) {

    candidate_index_t const num_candidates = preferences.extent(0);
#pragma omp parallel for collapse(2)
    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++)
            graph(row, column) = row != column && preferences(row, column) > preferences(column, row)
                                     ? preferences(row, column)
                                     : 0;
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
template <typename score_type_>
inline void positive_margins_graph(const_matrix_t preferences, strided_matrix<score_type_> graph) {

    candidate_index_t const num_candidates = preferences.extent(0);
#pragma omp parallel for collapse(2)
    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++) {
            votes_count_t const forward = preferences(row, column);
            votes_count_t const backward = preferences(column, row);
            graph(row, column) = row != column && forward > backward ? forward - backward : 0;
        }
}

/** Seeds the matrix with whichever graph the method is defined on. */
template <typename score_type_>
inline void seed_graph(const_matrix_t preferences, strided_matrix<score_type_> graph, seed_graph_t which) {
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
template <typename score_type_>
inline std::vector<candidate_index_t> select_split_cycle_winners(const_matrix_t preferences,
                                                                 strided_matrix<score_type_> margin_paths) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::vector<candidate_index_t> undefeated;
    for (candidate_index_t candidate = 0; candidate < num_candidates; candidate++) {
        bool defeated = false;
        for (candidate_index_t rival = 0; rival < num_candidates && !defeated; rival++) {
            if (rival == candidate) continue;
            votes_count_t const forward = preferences(rival, candidate);
            votes_count_t const backward = preferences(candidate, rival);
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
inline void tally_ballots_cpu(ballots_t rankings, matrix_t preferences) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::size_t const num_ballots = rankings.extent(0);
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
#pragma omp parallel
    {
        std::vector<votes_count_t> private_counts(cells, 0);
#pragma omp for schedule(static)
        for (std::ptrdiff_t ballot = 0; ballot < static_cast<std::ptrdiff_t>(num_ballots); ballot++) {
            candidate_index_t const* const ranking = &rankings(ballot, 0);
            for (candidate_index_t position = 0; position + 1 < num_candidates; position++) {
                candidate_index_t const preferred = ranking[position];
                for (candidate_index_t later = position + 1; later < num_candidates; later++)
                    private_counts[preferred * num_candidates + ranking[later]]++;
            }
        }
#pragma omp critical
        for (candidate_index_t row = 0; row < num_candidates; row++)
            for (candidate_index_t column = 0; column < num_candidates; column++)
                preferences(row, column) += private_counts[row * num_candidates + column];
    }
}

#if defined(SCALING_ELECTIONS_WITH_CUDA)

/** Threads per block for the tally, chosen so one block still fits a private matrix in shared memory. */
constexpr std::uint32_t tally_block_size_k = 256;

/** Shared bytes one tally block needs: a private matrix, plus the ballot each warp is staging. */
inline std::size_t tally_shared_bytes(candidate_index_t num_candidates) noexcept {
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
    std::size_t const staged = static_cast<std::size_t>(tally_block_size_k / warp_size_k) * num_candidates;
    return cells * sizeof(votes_count_t) + staged * sizeof(candidate_index_t);
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
__global__ void tally_ballots_cuda_(ballots_t rankings, matrix_t preferences) {

    candidate_index_t const num_candidates = preferences.extent(0);
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;

    extern __shared__ votes_count_t counters[];
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
                atomicAdd(&counters[preferred * num_candidates + staged[later]], 1u);
        }
        __syncwarp();
    }
    __syncthreads();

    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
        if (counters[cell]) atomicAdd(&preferences(cell / num_candidates, cell % num_candidates), counters[cell]);
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
inline void tally_ballots_gpu(ballots_t rankings, matrix_t preferences) {

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
    managed_vector<votes_count_t> device_counts(cells);
    std::fill(device_counts.begin(), device_counts.end(), votes_count_t {0});

    // A driver copy lands the chunk on the device outright, where a host loop would leave the
    // kernel to fault every page in one at a time.
    if (staging_needed && cudaMemcpy(staging.data(), rankings.data_handle(), staging.size() * sizeof(candidate_index_t),
                                     cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error("Failed to copy ballots to the device");

    ballots_t const device_rankings = staging_needed ? strided_view<candidate_index_t const, std::size_t>(
                                                           staging.data(), num_ballots, num_candidates, ballot_stride)
                                                     : rankings;
    matrix_t const device_preferences = square_view(device_counts.data(), num_candidates, num_candidates);

    std::size_t const warps_per_block = tally_block_size_k / warp_size_k;
    std::size_t const wanted_blocks = (num_ballots + warps_per_block - 1) / warps_per_block;
    unsigned int const blocks = static_cast<unsigned int>(std::min<std::size_t>(wanted_blocks, 65535));
    tally_ballots_cuda_<<<blocks, tally_block_size_k, tally_shared_bytes(num_candidates)>>>(device_rankings,
                                                                                            device_preferences);

    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA operations did not complete successfully");

    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++)
            preferences(row, column) += device_counts.data()[row * num_candidates + column];
}

/** Adds one chunk of complete rankings into an existing matrix, on whichever processor was named. */
inline void tally_ballots(ballots_t rankings, matrix_t preferences, backend_t backend) {
    if (backend == backend_t::cpu_k) return tally_ballots_cpu(rankings, preferences);
    return tally_ballots_gpu(rankings, preferences);
}

#else

/** Adds one chunk of complete rankings into an existing matrix; a CPU-only build has one processor. */
inline void tally_ballots(ballots_t rankings, matrix_t preferences, backend_t backend) {
    if (backend == backend_t::gpu_k)
        throw std::runtime_error("This build has no CUDA support, so `gpu` is unavailable");
    tally_ballots_cpu(rankings, preferences);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion Tally
