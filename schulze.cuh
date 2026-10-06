/**
 *  @brief Block-parallel Schulze strongest-paths kernels for CUDA, HIP, and OpenMP.
 *  @file schulze.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "ballots.cuh"
#include "types.cuh"

template <std::uint32_t tile_size_, typename element_type_ = default_schulze_arithmetic_t>
using votes_count_tile = element_type_[tile_size_][tile_size_];

/**
 *  @brief Which tile phase a processor runs, fixing both buffer aliasing and diagonal handling.
 *
 *  The three enumerators are the only combinations the recurrence produces, so the fourth
 *  pairing of the underlying flags cannot be spelled.
 */
enum class tile_phase_t : std::uint8_t {
    /** Output and inputs share one buffer, so every step needs a barrier. */
    aliased_k,
    /** Separate buffers, but the tile may straddle the matrix diagonal. */
    distinct_diagonal_k,
    /** Separate buffers, provably off the diagonal, so the update is a plain maximum. */
    distinct_independent_k,
};

/**
 *  @brief Where in the tile grid one step of the recurrence lands.
 *
 *  The output tile sits at (@p tile_row, @p tile_column) and both inputs share the pivot, so the
 *  three tiles are at (row, column), (row, pivot), and (pivot, column). Three indices therefore
 *  place all three tiles, which is what the diagonal tests need.
 */
struct tile_origin_t {
    /** The output tile's row in the tile grid. */
    candidate_index_t tile_row;
    /** The output tile's column in the tile grid. */
    candidate_index_t tile_column;
    /** The pivot tile both inputs are drawn from. */
    candidate_index_t pivot_tile;
};

/** Whether a tile copy bounds-checks its edges; `checked_k` also disables the NEON path. */
enum class tile_march_t : bool { fast_k = true, checked_k = false };

/** The view of @p graph running from tile (@p tile_row, @p tile_column) to its far corner. */
template <std::uint32_t tile_size_, typename element_type_>
inline strided_matrix<element_type_> tile_view(strided_matrix<element_type_> graph, candidate_index_t tile_row,
                                               candidate_index_t tile_column) noexcept {
    candidate_index_t const row = tile_row * tile_size_;
    candidate_index_t const column = tile_column * tile_size_;
    return strided_view<element_type_>(&graph(row, column), graph.extent(0) - row, graph.extent(1) - column,
                                       graph.stride(0));
}

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_CUDA)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#endif

#if defined(SCALING_ELECTIONS_KEPLER)

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *      in CUDA on Nvidia @b Kepler GPUs and newer (sm_30).
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tiles alias and whether the tile may straddle the diagonal.
 *  @tparam element_type_ The width one vote count occupies, deduced from the tiles.
 *
 *  Tile @p paths is the output, @p to_pivot and @p from_pivot the inputs; @p row and @p column
 *  address a cell within a tile, and @p origin places all three tiles in the global matrix.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_, typename element_type_>
__forceinline__ __device__ void process_tile_cuda_(                                      //
    votes_count_tile<tile_size_, element_type_>& paths,                                  //
    votes_count_tile<tile_size_, element_type_> const& to_pivot,                         //
    votes_count_tile<tile_size_, element_type_> const& from_pivot, tile_origin_t origin, //
    candidate_index_t row, candidate_index_t column) {
    using element_t = element_type_;

    element_t& paths_cell = paths[row][column];
    candidate_index_t const paths_row = origin.tile_row * tile_size_ + row;
    candidate_index_t const paths_column = origin.tile_column * tile_size_ + column;

#pragma unroll tile_size_
    for (candidate_index_t pivot = 0; pivot < tile_size_; pivot++) {
        element_t smallest = to_pivot[row][pivot] < from_pivot[pivot][column] ? to_pivot[row][pivot]
                                                                              : from_pivot[pivot][column];
        if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
            candidate_index_t const pivot_index = origin.pivot_tile * tile_size_ + pivot;
            std::uint32_t is_not_diagonal_paths = paths_row != paths_column;
            std::uint32_t is_not_diagonal_to_pivot = paths_row != pivot_index;
            std::uint32_t is_not_diagonal_from_pivot = pivot_index != paths_column;
            std::uint32_t is_bigger = smallest > paths_cell;
            std::uint32_t will_replace = is_not_diagonal_paths & is_not_diagonal_to_pivot & is_not_diagonal_from_pivot &
                                         is_bigger;
            // On Kepler an newer we can use `__funnelshift_lc` to avoid branches
            if constexpr (sizeof(element_t) <= sizeof(std::uint32_t))
                paths_cell = static_cast<element_t>(__funnelshift_lc(paths_cell, smallest, will_replace - 1));
            else paths_cell = will_replace ? smallest : paths_cell;
        }
        else paths_cell = paths_cell > smallest ? paths_cell : smallest;
        if constexpr (phase_ == tile_phase_t::aliased_k) __syncthreads();
    }
}

#else

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *      in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tiles alias and whether the tile may straddle the diagonal.
 *  @tparam element_type_ The width one vote count occupies, deduced from the tiles.
 *
 *  Tile @p paths is the output, @p to_pivot and @p from_pivot the inputs; @p row and @p column
 *  address a cell within a tile, and @p origin places all three tiles in the global matrix.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_, typename element_type_>
__forceinline__ __device__ void process_tile_cuda_(                                      //
    votes_count_tile<tile_size_, element_type_>& paths,                                  //
    votes_count_tile<tile_size_, element_type_> const& to_pivot,                         //
    votes_count_tile<tile_size_, element_type_> const& from_pivot, tile_origin_t origin, //
    candidate_index_t row, candidate_index_t column) {
    using element_t = element_type_;

    element_t& paths_cell = paths[row][column];
    candidate_index_t const paths_row = origin.tile_row * tile_size_ + row;
    candidate_index_t const paths_column = origin.tile_column * tile_size_ + column;

#pragma unroll tile_size_
    for (candidate_index_t pivot = 0; pivot < tile_size_; pivot++) {
        element_t smallest = to_pivot[row][pivot] < from_pivot[pivot][column] ? to_pivot[row][pivot]
                                                                              : from_pivot[pivot][column];
        if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
            candidate_index_t const pivot_index = origin.pivot_tile * tile_size_ + pivot;
            std::uint32_t is_not_diagonal_paths = paths_row != paths_column;
            std::uint32_t is_not_diagonal_to_pivot = paths_row != pivot_index;
            std::uint32_t is_not_diagonal_from_pivot = pivot_index != paths_column;
            std::uint32_t is_bigger = smallest > paths_cell;
            std::uint32_t will_replace = is_not_diagonal_paths & is_not_diagonal_to_pivot & is_not_diagonal_from_pivot &
                                         is_bigger;
            if (will_replace) paths_cell = smallest;
        }
        else paths_cell = paths_cell > smallest ? paths_cell : smallest;
        if constexpr (phase_ == tile_phase_t::aliased_k) __syncthreads();
    }
}

#endif

/**
 *  @brief Performs the diagonal step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam element_type_ The width one vote count occupies, deduced from the graph.
 *  @param[in] pivot_tile The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_, typename element_type_>
__global__ void schulze_diagonal_cuda_(candidate_index_t pivot_tile, strided_matrix<element_type_> graph) {
    candidate_index_t const row = threadIdx.y;
    candidate_index_t const column = threadIdx.x;

    alignas(16) __shared__ votes_count_tile<tile_size_, element_type_> paths;
    paths[row][column] = graph(pivot_tile * tile_size_ + row, pivot_tile * tile_size_ + column);

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        paths, paths, paths, tile_origin_t {pivot_tile, pivot_tile, pivot_tile}, row, column);

    graph(pivot_tile * tile_size_ + row, pivot_tile * tile_size_ + column) = paths[row][column];
}

/**
 *  @brief Performs the partially independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam element_type_ The width one vote count occupies, deduced from the graph.
 *  @param[in] pivot_tile The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_, typename element_type_>
__global__ void schulze_partial_cuda_(candidate_index_t pivot_tile, strided_matrix<element_type_> graph) {
    using element_t = element_type_;
    candidate_index_t const tile_index = blockIdx.x;
    candidate_index_t const row = threadIdx.y;
    candidate_index_t const column = threadIdx.x;

    if (tile_index == pivot_tile) return;

    alignas(16) __shared__ votes_count_tile<tile_size_, element_t> to_pivot;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_t> from_pivot;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_t> paths;

    // Partially dependent phase (first of two)
    // Walking down within a group of adjacent columns
    paths[row][column] = graph(tile_index * tile_size_ + row, pivot_tile * tile_size_ + column);
    from_pivot[row][column] = graph(pivot_tile * tile_size_ + row, pivot_tile * tile_size_ + column);

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        paths, paths, from_pivot, tile_origin_t {tile_index, pivot_tile, pivot_tile}, row, column);

    // Partially dependent phase (second of two)
    // Walking right within a group of adjacent rows
    __syncthreads();
    graph(tile_index * tile_size_ + row, pivot_tile * tile_size_ + column) = paths[row][column];
    paths[row][column] = graph(pivot_tile * tile_size_ + row, tile_index * tile_size_ + column);
    to_pivot[row][column] = graph(pivot_tile * tile_size_ + row, pivot_tile * tile_size_ + column);

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        paths, to_pivot, paths, tile_origin_t {pivot_tile, tile_index, pivot_tile}, row, column);

    graph(pivot_tile * tile_size_ + row, tile_index * tile_size_ + column) = paths[row][column];
}

/**
 *  @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam element_type_ The width one vote count occupies, deduced from the graph.
 *  @param[in] pivot_tile The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths.
 */
template <std::uint32_t tile_size_, typename element_type_>
__global__ void schulze_independent_cuda_(candidate_index_t pivot_tile, strided_matrix<element_type_> graph) {
    using element_t = element_type_;
    candidate_index_t const tile_column = blockIdx.x;
    candidate_index_t const tile_row = blockIdx.y;
    candidate_index_t const row = threadIdx.y;
    candidate_index_t const column = threadIdx.x;

    if (tile_row == pivot_tile && tile_column == pivot_tile) return;

    alignas(16) __shared__ votes_count_tile<tile_size_, element_t> to_pivot;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_t> from_pivot;
    alignas(16) __shared__ votes_count_tile<tile_size_, element_t> paths;

    paths[row][column] = graph(tile_row * tile_size_ + row, tile_column * tile_size_ + column);
    to_pivot[row][column] = graph(tile_row * tile_size_ + row, pivot_tile * tile_size_ + column);
    from_pivot[row][column] = graph(pivot_tile * tile_size_ + row, tile_column * tile_size_ + column);

    __syncthreads();
    tile_origin_t const origin {tile_row, tile_column, pivot_tile};
    if (tile_row == tile_column)
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
            paths, to_pivot, from_pivot, origin, row, column);
    else
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_independent_k>( //
            paths, to_pivot, from_pivot, origin, row, column);

    graph(tile_row * tile_size_ + row, tile_column * tile_size_ + column) = paths[row][column];
}

#pragma region Packed Sixteen Bit

/** Which element width the pivot sweep runs on. */
enum class graph_width_t : std::uint8_t {
    /** The 32-bit path, which every device and every backend supports. */
    wide_32_k,
    /** The 16-bit path, two candidates per word, needing every vote count below 65536. */
    narrow_16_k,
};

/** Copies the graph into a narrower shadow, raising @p overflowed for any cell that will not fit. */
template <typename wide_type_, typename narrow_type_>
__global__ void schulze_narrow_cuda_(std::size_t cells, wide_type_ const* wide, narrow_type_* narrow,
                                     wide_type_* overflowed) {
    using wide_t = wide_type_;
    using narrow_t = narrow_type_;
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t const first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    constexpr wide_t widest_k = static_cast<narrow_t>(~static_cast<narrow_t>(0));
    for (std::size_t cell = first; cell < cells; cell += stride) {
        wide_t const value = wide[cell];
        if (value > widest_k) *overflowed = 1;
        narrow[cell] = static_cast<narrow_t>(value);
    }
}

/** Copies the narrow shadow back over the graph the caller owns. */
template <typename wide_type_, typename narrow_type_>
__global__ void schulze_widen_cuda_(std::size_t cells, narrow_type_ const* narrow, wide_type_* wide) {
    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    std::size_t const first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::size_t cell = first; cell < cells; cell += stride) wide[cell] = narrow[cell];
}

#if !defined(SCALING_ELECTIONS_WITH_HIP)

/**
 *  @brief Performs the independent step on a 16-bit graph, two candidates per 32-bit word.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] pivot_tile The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths, viewed as pairs of adjacent vote counts.
 *
 *  Each thread owns two words, so one block covers a tile with a quarter of the threads the
 *  32-bit kernel needs. The max-min semiring has no packed primitive, so the pair is spelled as
 *  a three-way minimum with a repeated argument, then a three-way maximum.
 */
template <std::uint32_t tile_size_>
__global__ void schulze_independent_packed_cuda_(candidate_index_t pivot_tile, strided_matrix<std::uint32_t> graph) {
    static_assert(tile_size_ % 4 == 0, "A packed tile row must divide evenly across the lanes");
    constexpr std::uint32_t tile_words_k = tile_size_ / 2;
    constexpr std::uint32_t words_per_thread_k = 2;
    constexpr std::uint32_t lanes_k = tile_words_k / words_per_thread_k;

    candidate_index_t const tile_column = blockIdx.x;
    candidate_index_t const tile_row = blockIdx.y;
    candidate_index_t const row = threadIdx.y;
    candidate_index_t const lane = threadIdx.x;

    if (tile_row == pivot_tile && tile_column == pivot_tile) return;

    alignas(16) __shared__ std::uint32_t to_pivot[tile_size_][tile_words_k];
    alignas(16) __shared__ std::uint32_t from_pivot[tile_size_][tile_words_k];
    alignas(16) __shared__ std::uint32_t paths[tile_size_][tile_words_k];

    // Staging strides by the lane count so each warp reads one contiguous run.
#pragma unroll
    for (std::uint32_t slice = 0; slice < words_per_thread_k; slice++) {
        candidate_index_t const word = lane + slice * lanes_k;
        paths[row][word] = graph(tile_row * tile_size_ + row, tile_column * tile_words_k + word);
        to_pivot[row][word] = graph(tile_row * tile_size_ + row, pivot_tile * tile_words_k + word);
        from_pivot[row][word] = graph(pivot_tile * tile_size_ + row, tile_column * tile_words_k + word);
    }
    __syncthreads();

    candidate_index_t const first_word = lane * words_per_thread_k;
    candidate_index_t const diagonal_word = row / 2;
    uint2 paths_pair = *reinterpret_cast<uint2 const*>(&paths[row][first_word]);
    std::uint32_t const diagonal_before = diagonal_word == first_word ? paths_pair.x : paths_pair.y;

#pragma unroll tile_size_
    for (candidate_index_t step = 0; step < tile_size_; step++) {
        std::uint32_t const to_pivot_word = to_pivot[row][step / 2];
        std::uint32_t const to_pivot_pair = __byte_perm(to_pivot_word, 0u, (step % 2) ? 0x3232u : 0x1010u);
        uint2 const from_pivot_pair = *reinterpret_cast<uint2 const*>(&from_pivot[step][first_word]);
        std::uint32_t const smallest_low = __vimin3_u16x2(to_pivot_pair, from_pivot_pair.x, from_pivot_pair.x);
        std::uint32_t const smallest_high = __vimin3_u16x2(to_pivot_pair, from_pivot_pair.y, from_pivot_pair.y);
        paths_pair.x = __vimax3_u16x2(paths_pair.x, smallest_low, smallest_low);
        paths_pair.y = __vimax3_u16x2(paths_pair.y, smallest_high, smallest_high);
    }

    // A tile straddling the matrix diagonal leaves those cells at the semiring identity.
    if (tile_row == tile_column) {
        std::uint32_t const keep_diagonal = (row % 2) ? 0x7610u : 0x3254u;
        if (diagonal_word == first_word) paths_pair.x = __byte_perm(paths_pair.x, diagonal_before, keep_diagonal);
        else if (diagonal_word == first_word + 1)
            paths_pair.y = __byte_perm(paths_pair.y, diagonal_before, keep_diagonal);
    }

    *reinterpret_cast<uint2*>(&paths[row][first_word]) = paths_pair;
    __syncthreads();

#pragma unroll
    for (std::uint32_t slice = 0; slice < words_per_thread_k; slice++) {
        candidate_index_t const word = lane + slice * lanes_k;
        graph(tile_row * tile_size_ + row, tile_column * tile_words_k + word) = paths[row][word];
    }
}

#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#pragma endregion Packed Sixteen Bit

/**
 *  @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA (NVIDIA Hopper only).
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] pivot_tile The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths, as a @c CUtensorMap .
 *
 *  @note This kernel uses NVIDIA-specific Tensor Memory Access (TMA) and is not available on AMD GPUs.
 */
#if !defined(SCALING_ELECTIONS_WITH_HIP)
template <std::uint32_t tile_size_>
__global__ void schulze_independent_hopper_cuda_(candidate_index_t pivot_tile,
                                                 __grid_constant__ CUtensorMap const graph) {
    candidate_index_t const tile_column = blockIdx.x;
    candidate_index_t const tile_row = blockIdx.y;
    candidate_index_t const row = threadIdx.y;
    candidate_index_t const column = threadIdx.x;

#if defined(SCALING_ELECTIONS_HOPPER)

    if (tile_row == pivot_tile && tile_column == pivot_tile) return;

    alignas(128) __shared__ votes_count_tile<tile_size_, std::uint32_t> to_pivot;
    alignas(128) __shared__ votes_count_tile<tile_size_, std::uint32_t> from_pivot;
    alignas(128) __shared__ votes_count_tile<tile_size_, std::uint32_t> paths;

#pragma nv_diag_suppress static_var_with_dynamic_init
    // Initialize shared memory barrier with the number of threads participating in the barrier.
    __shared__ barrier_t tile_barrier;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // We have one thread per tile cell.
        init(&tile_barrier, tile_size_ * tile_size_);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    // Sync threads so initialized barrier is visible to all threads.
    __syncthreads();

    // Only the first thread in the tile invokes the bulk transfers.
    barrier_t::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Initiate three bulk tensor copies for different part of the graph.
        // The first coordinate is the column, as dimension 0 is the contiguous one.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&paths, &graph, tile_column * tile_size_, tile_row * tile_size_,
                                                      tile_barrier);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&to_pivot, &graph, pivot_tile * tile_size_, tile_row * tile_size_,
                                                      tile_barrier);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&from_pivot, &graph, tile_column * tile_size_,
                                                      pivot_tile * tile_size_, tile_barrier);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(tile_barrier, 1, sizeof(paths) + sizeof(to_pivot) + sizeof(from_pivot));
    }
    else {
        // Other threads just arrive.
        token = tile_barrier.arrive(1);
    }

    // Past this point the three tiles are resident in shared memory.
    tile_barrier.wait(std::move(token));

    tile_origin_t const origin {tile_row, tile_column, pivot_tile};
    if (tile_row == tile_column)
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
            paths, to_pivot, from_pivot, origin, row, column);
    else
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_independent_k>( //
            paths, to_pivot, from_pivot, origin, row, column);

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After `syncthreads`, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&graph, tile_column * tile_size_, tile_row * tile_size_, &paths);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();

        // The barrier needs no explicit destruction at the end of the kernel.
    }
#else
    // This is a trap :)
    if (tile_row == 0 && tile_column == 0 && row == 0 && column == 0)
        printf("This kernel is only supported on Hopper and newer GPUs\n");
#endif
}
#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get pointer to cuGetProcAddress
    cudaDriverEntryPointQueryResult driver_status;
    void* cuGetProcAddress_ptr = nullptr;
    cudaError_t error = cudaGetDriverEntryPoint("cuGetProcAddress", &cuGetProcAddress_ptr, cudaEnableDefault,
                                                &driver_status);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get cuGetProcAddress");
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress entry point");
    PFN_cuGetProcAddress_v12000 cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(cuGetProcAddress_ptr);

    // Use cuGetProcAddress to get a pointer to the CTK 12.0 version of cuTensorMapEncodeTiled
    CUdriverProcAddressQueryResult symbol_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUresult encode_status = cuGetProcAddress("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                              CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
    if (encode_status != CUDA_SUCCESS || symbol_status != CU_GET_PROC_ADDRESS_SUCCESS)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#if !defined(SCALING_ELECTIONS_WITH_HIP)

/** Tensor map for the Hopper bulk-tensor path, absent when the device or the layout rules it out. */
using tma_descriptor_t = std::optional<CUtensorMap>;

/**
 *  @brief Builds the tensor map describing the padded strongest-paths matrix.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] graph The padded matrix of strongest paths.
 *  @param[in] device_properties Properties of the device the kernels will run on.
 *  @return The descriptor, having thrown if the device or the layout cannot supply one.
 */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(uint32_matrix_t graph, cudaDeviceProp const& device_properties) {
    if (device_properties.major < 9)
        throw std::runtime_error("The Hopper kernel needs compute capability 9.0, found " +
                                 std::to_string(device_properties.major) + "." +
                                 std::to_string(device_properties.minor));

    CUtensorMap descriptor_map {};
    candidate_index_t const graph_stride = graph.stride(0);

    // rank is the number of dimensions of the array.
    constexpr std::uint32_t rank_k = 2;
    uint64_t size[rank_k] = {graph_stride, graph_stride};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank_k - 1] = {graph_stride * sizeof(std::uint32_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    std::uint32_t box_size[rank_k] = {tile_size_, tile_size_};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    std::uint32_t element_stride[rank_k] = {1, 1};

    // Create the tensor descriptor.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult encode_status = cuTensorMapEncodeTiled( //
        &descriptor_map,                             // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        rank_k,              // cuuint32_t tensorRank,
        graph.data_handle(), // void *globalAddress,
        size,                // const cuuint64_t *globalDim,
        stride,              // const cuuint64_t *globalStrides,
        box_size,            // const cuuint32_t *boxDim,
        element_stride,      // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines. Can be 64b, 128b, 256b, or none.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (encode_status != CUDA_SUCCESS)
        throw std::runtime_error("The Hopper kernel could not encode a tensor map for this layout");
    return descriptor_map;
}

/**
 *  @brief Launches the independent phase, preferring the bulk-tensor kernel where it is available.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] grid The grid shape covering every tile pair.
 *  @param[in] block The block shape, one thread per tile cell.
 *  @param[in] pivot_tile The index of the current pivot tile.
 *  @param[inout] graph The padded matrix of strongest paths.
 *  @param[in] tma The tensor-map descriptor produced by @c require_tma_ .
 */
template <std::uint32_t tile_size_>
void launch_independent_(dim3 grid, dim3 block, candidate_index_t pivot_tile, uint32_matrix_t graph,
                         tma_descriptor_t const& tma) {
    if (tma) schulze_independent_hopper_cuda_<tile_size_><<<grid, block>>>(pivot_tile, *tma);
    else schulze_independent_cuda_<tile_size_><<<grid, block>>>(pivot_tile, graph);
}

#else

/** HIP has no bulk-tensor engine, so the descriptor type can never hold one. */
using tma_descriptor_t = std::nullopt_t;

template <std::uint32_t tile_size_>
void launch_independent_(dim3 grid, dim3 block, candidate_index_t pivot_tile, uint32_matrix_t graph,
                         tma_descriptor_t const&) {
    schulze_independent_cuda_<tile_size_><<<grid, block>>>(pivot_tile, graph);
}

/** HIP has no bulk-tensor engine, so the descriptor can never be built. */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(uint32_matrix_t, cudaDeviceProp const&) {
    throw std::runtime_error("The Hopper kernel is unavailable in a HIP build");
}

#endif

#if !defined(SCALING_ELECTIONS_WITH_HIP)

#if defined(__CUDA_ARCH_LIST__)
/** Whether this device can load native code containing the sm_90 instructions. */
constexpr bool carries_native_sm90_code_(cudaDeviceProp const& device_properties) {
    for (int compiled : {__CUDA_ARCH_LIST__})
        if (compiled >= 900 && compiled / 100 == device_properties.major &&
            compiled % 100 <= device_properties.minor * 10)
            return true;
    return false;
}
#else
/** Without an architecture list the build cannot promise a native packed min-max. */
constexpr bool carries_native_sm90_code_(cudaDeviceProp const&) { return false; }
#endif

/**
 *  @brief Picks the width the sweep would prefer, before any vote count has been looked at.
 *
 *  Below sm_90 the packed intrinsics are emulated and lose to the 32-bit path, so both the
 *  compiled architectures and the running device have to offer the real instruction.
 */
template <std::uint32_t tile_size_>
graph_width_t choose_width_(cudaDeviceProp const& device_properties) {
    if constexpr (tile_size_ % 4 != 0) return graph_width_t::wide_32_k;
    else {
        if (!carries_native_sm90_code_(device_properties)) return graph_width_t::wide_32_k;
        if (device_properties.major < 9) return graph_width_t::wide_32_k;
        return graph_width_t::narrow_16_k;
    }
}

/** Runs every pivot step on the 16-bit shadow, with the independent phase packed two per word. */
template <std::uint32_t tile_size_>
void sweep_narrow_(strided_matrix<std::uint16_t> graph) {
    candidate_index_t const graph_stride = graph.extent(0);
    candidate_index_t const tiles_count = graph_stride / tile_size_;
    strided_matrix<std::uint32_t> const packed = strided_view<std::uint32_t>(
        reinterpret_cast<std::uint32_t*>(graph.data_handle()), graph_stride, graph_stride / 2, graph_stride / 2);
    dim3 const tile_shape(tile_size_, tile_size_, 1);
    dim3 const packed_shape(tile_size_ / 4, tile_size_, 1);
    dim3 const independent_grid(tiles_count, tiles_count, 1);
    for (candidate_index_t pivot_tile = 0; pivot_tile < tiles_count; pivot_tile++) {
        schulze_diagonal_cuda_<tile_size_><<<1, tile_shape>>>(pivot_tile, graph);
        schulze_partial_cuda_<tile_size_><<<tiles_count, tile_shape>>>(pivot_tile, graph);
        schulze_independent_packed_cuda_<tile_size_><<<independent_grid, packed_shape>>>(pivot_tile, packed);

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
}

#else

/** HIP has no packed 16-bit min-max, so the sweep never narrows. */
template <std::uint32_t tile_size_>
graph_width_t choose_width_(cudaDeviceProp const&) {
    return graph_width_t::wide_32_k;
}

/** HIP has no packed 16-bit min-max, so the narrow sweep can never run. */
template <std::uint32_t tile_size_>
void sweep_narrow_(strided_matrix<std::uint16_t>) {
    throw std::runtime_error("The packed 16-bit path is unavailable in a HIP build");
}

#endif

/** Runs every pivot step on the 32-bit graph, using TMA when a tensor map is available. */
template <std::uint32_t tile_size_, tally_arithmetic arithmetic_type_>
void sweep_wide_(strided_matrix<arithmetic_type_> graph, tma_descriptor_t const& tma) {
    candidate_index_t const tiles_count = graph.extent(0) / tile_size_;
    dim3 const tile_shape(tile_size_, tile_size_, 1);
    dim3 const independent_grid(tiles_count, tiles_count, 1);
    for (candidate_index_t pivot_tile = 0; pivot_tile < tiles_count; pivot_tile++) {
        schulze_diagonal_cuda_<tile_size_><<<1, tile_shape>>>(pivot_tile, graph);
        schulze_partial_cuda_<tile_size_><<<tiles_count, tile_shape>>>(pivot_tile, graph);
        if constexpr (std::is_same_v<arithmetic_type_, std::uint32_t>)
            launch_independent_<tile_size_>(independent_grid, tile_shape, pivot_tile, graph, tma);
        else schulze_independent_cuda_<tile_size_><<<independent_grid, tile_shape>>>(pivot_tile, graph);

        cudaError_t const error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
}

/** How many blocks a flat pass over @p cells needs, capped so the grid stays resident. */
inline std::uint32_t flat_blocks_(std::size_t cells, std::uint32_t threads_per_block) {
    std::size_t const needed = divide_round_up(cells, threads_per_block);
    return static_cast<std::uint32_t>(std::min<std::size_t>(needed, 4096));
}

/** Fills the 16-bit shadow, reporting the width the sweep can actually use. */
inline graph_width_t narrow_graph_(std::size_t cells, std::uint32_t const* graph, std::uint16_t* narrow,
                                   std::uint32_t* overflowed) {
    constexpr std::uint32_t threads_per_block_k = 256;
    *overflowed = 0;
    schulze_narrow_cuda_<<<flat_blocks_(cells, threads_per_block_k), threads_per_block_k>>>(cells, graph, narrow,
                                                                                            overflowed);
    cudaError_t const error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return *overflowed != 0 ? graph_width_t::wide_32_k : graph_width_t::narrow_16_k;
}

/** Copies the 16-bit shadow back over the graph the caller owns. */
inline void widen_graph_(std::size_t cells, std::uint16_t const* narrow, std::uint32_t* graph) {
    constexpr std::uint32_t threads_per_block_k = 256;
    schulze_widen_cuda_<<<flat_blocks_(cells, threads_per_block_k), threads_per_block_k>>>(cells, narrow, graph);
    cudaError_t const error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] preferences The preferences matrix.
 *  @param[out] graph The padded output matrix of strongest paths, a whole number of tiles wide.
 *  @param[in] seed Which graph the sweep closes over.
 *
 *  On sm_90, safe 16-bit graphs use packed min-max; wider graphs use TMA.
 *  Other devices use the ordinary tiled sweep.
 */
template <std::uint32_t tile_size_, tally_arithmetic arithmetic_type_ = default_schulze_arithmetic_t,
          typename stored_count_type_ = default_stored_count_t>
void compute_strongest_paths_gpu(strided_matrix<stored_count_type_ const> preferences,
                                 strided_matrix<arithmetic_type_> graph,
                                 seed_graph_t seed = seed_graph_t::winning_votes_k,
                                 score_type_t score_type = score_type_t::auto_k) {
    using arithmetic_t = arithmetic_type_;

    candidate_index_t const num_candidates = preferences.extent(0);
    candidate_index_t const graph_stride = graph.stride(0);
    seed_graph(preferences, square_view(graph.data_handle(), num_candidates, graph_stride), seed);

    // Check if we can use newer CUDA features.
    cudaError_t error;
    int current_device;
    cudaDeviceProp device_properties;
    error = cudaGetDevice(&current_device);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get current device");
    error = cudaGetDeviceProperties(&device_properties, current_device);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get device properties");

    tma_descriptor_t tma {std::nullopt};
    if constexpr (std::is_same_v<arithmetic_t, std::uint16_t>) {
        if (choose_width_<tile_size_>(device_properties) == graph_width_t::narrow_16_k) {
            sweep_narrow_<tile_size_>(graph);
            return;
        }
    }
    if constexpr (std::is_same_v<arithmetic_t, std::uint32_t>) {
        std::size_t const cells = static_cast<std::size_t>(graph_stride) * graph_stride;
        if (score_type == score_type_t::auto_k &&
            choose_width_<tile_size_>(device_properties) == graph_width_t::narrow_16_k) {
            managed_vector<std::uint16_t> narrow(cells);
            managed_vector<std::uint32_t> overflowed(1);
            if (narrow_graph_(cells, graph.data_handle(), narrow.data(), overflowed.data()) ==
                graph_width_t::narrow_16_k) {
                sweep_narrow_<tile_size_>(square_view(narrow.data(), graph_stride, graph_stride));
                widen_graph_(cells, narrow.data(), graph.data_handle());
                return;
            }
        }

#if !defined(SCALING_ELECTIONS_WITH_HIP)
        if (carries_native_sm90_code_(device_properties) && device_properties.major >= 9) {
            tma = require_tma_<tile_size_>(graph, device_properties);
        }
#endif
    }
    sweep_wide_<tile_size_>(graph, tma);
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion CUDA

#pragma region OpenMP

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze
 *      voting algorithm on CPU using @b OpenMP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam phase_ Whether the tile may straddle the matrix diagonal.
 *
 *  Tile @p paths is the output, @p to_pivot and @p from_pivot the inputs, and @p origin places all
 *  three in the global matrix. Every cell is walked serially, so an aliased phase needs no barrier.
 */
template <std::uint32_t tile_size_, tile_phase_t phase_, tally_arithmetic arithmetic_type_>
inline void process_tile_openmp_(                                     //
    votes_count_tile<tile_size_, arithmetic_type_>& paths,            //
    votes_count_tile<tile_size_, arithmetic_type_> const& to_pivot,   //
    votes_count_tile<tile_size_, arithmetic_type_> const& from_pivot, //
    tile_origin_t origin) {
    using arithmetic_t = arithmetic_type_;

    candidate_index_t const paths_row_origin = origin.tile_row * tile_size_;
    candidate_index_t const paths_column_origin = origin.tile_column * tile_size_;
    candidate_index_t const pivot_origin = origin.pivot_tile * tile_size_;

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<arithmetic_t, std::uint32_t>() && tile_size_ % 4 == 0) {
        uint32x4_t column_step = {0, 1, 2, 3};
        for (candidate_index_t pivot = 0; pivot < tile_size_; pivot++) {
            uint32x4_t pivot_index_vec = vdupq_n_u32(pivot_origin + pivot);
            for (candidate_index_t row = 0; row < tile_size_; row++) {
                uint32x4_t to_pivot_vec = vdupq_n_u32(to_pivot[row][pivot]);
                uint32x4_t paths_row_vec = vdupq_n_u32(paths_row_origin + row);
                uint32x4_t is_not_diagonal_to_pivot = vmvnq_u32(vceqq_u32(paths_row_vec, pivot_index_vec));
                SCALING_ELECTIONS_UNROLL
                for (candidate_index_t column = 0; column < tile_size_; column += 4) {
                    arithmetic_t* paths_cells = &paths[row][column];
                    uint32x4_t paths_vec = vld1q_u32(paths_cells);
                    uint32x4_t from_pivot_vec = vld1q_u32(&from_pivot[pivot][column]);
                    uint32x4_t smallest = vminq_u32(to_pivot_vec, from_pivot_vec);

                    if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
                        uint32x4_t paths_column_vec = vaddq_u32(vdupq_n_u32(paths_column_origin + column), column_step);
                        uint32x4_t is_diagonal_paths = vceqq_u32(paths_row_vec, paths_column_vec);
                        uint32x4_t is_diagonal_from_pivot = vceqq_u32(pivot_index_vec, paths_column_vec);
                        uint32x4_t is_bigger = vcgtq_u32(smallest, paths_vec);
                        uint32x4_t will_replace =                                                //
                            vandq_u32(                                                           //
                                vmvnq_u32(vorrq_u32(is_diagonal_paths, is_diagonal_from_pivot)), //
                                vandq_u32(is_not_diagonal_to_pivot, is_bigger));
                        paths_vec = vbslq_u32(will_replace, smallest, paths_vec);
                    }
                    else { paths_vec = vmaxq_u32(paths_vec, smallest); }
                    vst1q_u32(paths_cells, paths_vec);
                }
            }
        }
        return;
    }
#endif
    for (candidate_index_t pivot = 0; pivot < tile_size_; pivot++) {
        candidate_index_t const pivot_index = pivot_origin + pivot;
        for (candidate_index_t row = 0; row < tile_size_; row++) {
            arithmetic_t* const paths_cells = &paths[row][0];
#pragma omp simd
            for (candidate_index_t column = 0; column < tile_size_; column++) {
                arithmetic_t paths_cell = paths_cells[column];
                arithmetic_t smallest = std::min(to_pivot[row][pivot], from_pivot[pivot][column]);
                if constexpr (phase_ != tile_phase_t::distinct_independent_k) {
                    std::uint32_t is_not_diagonal_paths = (paths_row_origin + row) != (paths_column_origin + column);
                    std::uint32_t is_not_diagonal_to_pivot = (paths_row_origin + row) != pivot_index;
                    std::uint32_t is_not_diagonal_from_pivot = pivot_index != (paths_column_origin + column);
                    std::uint32_t is_bigger = smallest > paths_cell;
                    std::uint32_t will_replace = is_not_diagonal_paths & is_not_diagonal_to_pivot &
                                                 is_not_diagonal_from_pivot & is_bigger;
                    paths_cells[column] = will_replace ? smallest : paths_cell;
                }
                else { paths_cells[column] = std::max(paths_cell, smallest); }
            }
        }
    }
}

/** Stages the tile whose top-left corner @p source names into @p target , zero-filling any tail. */
template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k,
          tally_arithmetic arithmetic_type_ = default_schulze_arithmetic_t>
void memcpy2d(strided_matrix<arithmetic_type_> source, votes_count_tile<tile_size_, arithmetic_type_>& target) {
    using arithmetic_t = arithmetic_type_;

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<arithmetic_t, std::uint32_t>() && tile_size_ % 4 == 0 &&
                  march_ == tile_march_t::fast_k) {
        for (candidate_index_t row = 0; row < tile_size_; row++) {
            SCALING_ELECTIONS_UNROLL
            for (candidate_index_t column = 0; column < tile_size_; column += 4) {
                vst1q_u32(&target[row][column], vld1q_u32(&source(row, column)));
            }
        }
        return;
    }
#endif

    if constexpr (march_ == tile_march_t::checked_k) {
        candidate_index_t const remaining_rows = source.extent(0);
        candidate_index_t const remaining_columns = source.extent(1);
        for (candidate_index_t row = 0; row < tile_size_; row++)
            for (candidate_index_t column = 0; column < tile_size_; column++)
                target[row][column] = row < remaining_rows && column < remaining_columns ? source(row, column) : 0;
    }
    // One row lookup per row, since a strided view multiplies on every cell it is asked for.
    else
        for (candidate_index_t row = 0; row < tile_size_; row++) {
            arithmetic_t const* const source_row = &source(row, 0);
            for (candidate_index_t column = 0; column < tile_size_; column++) target[row][column] = source_row[column];
        }
}

/** Writes @p source back over the tile whose top-left corner @p target names, dropping any tail. */
template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k,
          tally_arithmetic arithmetic_type_ = default_schulze_arithmetic_t>
void memcpy2d(votes_count_tile<tile_size_, arithmetic_type_> const& source, strided_matrix<arithmetic_type_> target) {
    using arithmetic_t = arithmetic_type_;

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<arithmetic_t, std::uint32_t>() && tile_size_ % 4 == 0 &&
                  march_ == tile_march_t::fast_k) {
        for (candidate_index_t row = 0; row < tile_size_; row++) {
            SCALING_ELECTIONS_UNROLL
            for (candidate_index_t column = 0; column < tile_size_; column += 4) {
                vst1q_u32(&target(row, column), vld1q_u32(&source[row][column]));
            }
        }
        return;
    }
#endif

    if constexpr (march_ == tile_march_t::checked_k) {
        candidate_index_t const remaining_rows = target.extent(0);
        candidate_index_t const remaining_columns = target.extent(1);
        for (candidate_index_t row = 0; row < tile_size_; row++)
            for (candidate_index_t column = 0; column < tile_size_; column++)
                if (row < remaining_rows && column < remaining_columns) target(row, column) = source[row][column];
    }
    // One row lookup per row, since a strided view multiplies on every cell it is asked for.
    else
        for (candidate_index_t row = 0; row < tile_size_; row++) {
            arithmetic_t* const target_row = &target(row, 0);
            for (candidate_index_t column = 0; column < tile_size_; column++) target_row[column] = source[row][column];
        }
}

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm using @b OpenMP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam march_ Whether tile copies bounds-check their edges.
 *  @param[in] preferences The preferences matrix.
 *  @param[out] graph The output matrix of strongest paths, packed to the candidate count per row.
 *  @param[in] cancelled Polled between pivots, aborting the run once it reads non-zero.
 *  @param[in] seed Which graph the sweep closes over.
 */
template <std::uint32_t tile_size_, tile_march_t march_ = tile_march_t::fast_k,
          tally_arithmetic arithmetic_type_ = default_schulze_arithmetic_t,
          typename stored_count_type_ = default_stored_count_t>
void compute_strongest_paths_tiled_cpu( //
    strided_matrix<stored_count_type_ const> preferences, strided_matrix<arithmetic_type_> graph,
    volatile std::sig_atomic_t const* cancelled = nullptr, seed_graph_t seed = seed_graph_t::winning_votes_k) {
    using arithmetic_t = arithmetic_type_;

    seed_graph(preferences, graph, seed);

    // Time for the actual core implementation
    candidate_index_t const num_candidates = preferences.extent(0);
    candidate_index_t const tiles_count = divide_round_up(num_candidates, tile_size_);
    for (candidate_index_t pivot_tile = 0; pivot_tile < tiles_count; pivot_tile++) {

        if (cancelled && *cancelled) throw std::runtime_error("Stopped by signal");

        // Dependent phase
        {
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
            memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, pivot_tile, pivot_tile), paths);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                paths, paths, paths, tile_origin_t {pivot_tile, pivot_tile, pivot_tile});
            memcpy2d<tile_size_, march_>(paths, tile_view<tile_size_>(graph, pivot_tile, pivot_tile));
        }
        // Partially dependent phase (first of two)
#pragma omp parallel for schedule(dynamic)
        for (candidate_index_t tile_row = 0; tile_row < tiles_count; tile_row++) {
            if (tile_row == pivot_tile) continue;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> from_pivot;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
            memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, tile_row, pivot_tile), paths);
            memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, pivot_tile, pivot_tile), from_pivot);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                paths, paths, from_pivot, tile_origin_t {tile_row, pivot_tile, pivot_tile});
            memcpy2d<tile_size_, march_>(paths, tile_view<tile_size_>(graph, tile_row, pivot_tile));
        }
        // Partially dependent phase (second of two)
#pragma omp parallel for schedule(dynamic)
        for (candidate_index_t tile_column = 0; tile_column < tiles_count; tile_column++) {
            if (tile_column == pivot_tile) continue;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> to_pivot;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
            memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, pivot_tile, tile_column), paths);
            memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, pivot_tile, pivot_tile), to_pivot);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                paths, to_pivot, paths, tile_origin_t {pivot_tile, tile_column, pivot_tile});
            memcpy2d<tile_size_, march_>(paths, tile_view<tile_size_>(graph, pivot_tile, tile_column));
        }
        // Independent phase
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (candidate_index_t tile_row = 0; tile_row < tiles_count; tile_row++) {
            for (candidate_index_t tile_column = 0; tile_column < tiles_count; tile_column++) {
                if (tile_row == pivot_tile || tile_column == pivot_tile) continue;
                alignas(64) votes_count_tile<tile_size_, arithmetic_t> to_pivot;
                alignas(64) votes_count_tile<tile_size_, arithmetic_t> from_pivot;
                alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
                memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, tile_row, tile_column), paths);
                memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, tile_row, pivot_tile), to_pivot);
                memcpy2d<tile_size_, march_>(tile_view<tile_size_>(graph, pivot_tile, tile_column), from_pivot);
                tile_origin_t const origin {tile_row, tile_column, pivot_tile};
                if (tile_row != tile_column)
                    process_tile_openmp_<tile_size_, tile_phase_t::distinct_independent_k>( //
                        paths, to_pivot, from_pivot, origin);
                else
                    process_tile_openmp_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
                        paths, to_pivot, from_pivot, origin);
                memcpy2d<tile_size_, march_>(paths, tile_view<tile_size_>(graph, tile_row, tile_column));
            }
        }
    }
}

#pragma endregion OpenMP
