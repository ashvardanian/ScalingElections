/**
 * @brief  CUDA-accelerated Schulze voting algorithm implementation.
 * @file   scaling_elections.cu
 * @author Ash Vardanian
 * @date   July 12, 2024
 * @see    https://ashvardanian.com/ScalingElections
 */
#include <algorithm> // `std::min`, `std::max`
#include <csignal>   // `std::signal`
#include <cstdint>   // `std::uint32_t`
#include <cstdio>    // `std::printf`
#include <cstdlib>   // `std::rand`
#include <stdexcept> // `std::runtime_error`
#include <thread>    // `std::thread::hardware_concurrency()`
#include <vector>    // `std::vector`

// OpenMP support detection
#if defined(_OPENMP)
#include <omp.h> // `omp_set_num_threads`
#define SCALING_ELECTIONS_WITH_OPENMP (1)
#endif

#if (defined(__ARM_NEON) || defined(__aarch64__))
#define SCALING_ELECTIONS_WITH_NEON (1)
#endif
#if defined(__NVCC__)
#define SCALING_ELECTIONS_WITH_CUDA (1)
#endif
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define SCALING_ELECTIONS_WITH_HIP (1)
#define SCALING_ELECTIONS_WITH_CUDA (1) // HIP is CUDA-compatible
#endif

#if defined(SCALING_ELECTIONS_WITH_NEON)
#include <arm_neon.h>
#endif

#if defined(SCALING_ELECTIONS_WITH_CUDA) && !defined(SCALING_ELECTIONS_WITH_HIP)
// NVIDIA CUDA headers
#include <cuda.h> // `CUtensorMap`
#include <cuda/barrier>
#include <cudaTypedefs.h> // `PFN_cuTensorMapEncodeTiled`
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

// AMD HIP headers (CUDA-compatible)
#elif defined(SCALING_ELECTIONS_WITH_HIP)
#include <hip/hip_runtime.h>

// HIP compatibility layer: map CUDA types/functions to HIP equivalents
#if defined(__HIP_PLATFORM_AMD__)
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t
#define cudaMallocManaged hipMallocManaged
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetDeviceCount hipGetDeviceCount

#endif
#endif

/*
 * If we are only testing the raw kernels, we don't need to link to PyBind.
 */
#if !defined(SCALING_ELECTIONS_TEST)
#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300 && !defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_KEPLER (1)
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_HOPPER (1)
#endif

using votes_count_t = std::uint32_t;
using candidate_idx_t = std::uint32_t;

template <std::uint32_t tile_size> using votes_count_tile = votes_count_t[tile_size][tile_size];

/**
 * @brief   Stores the interrupt signal status.
 */
volatile std::sig_atomic_t global_signal_status = 0;

void signal_handler(int signal) { global_signal_status = signal; }

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_CUDA)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#endif

#if defined(SCALING_ELECTIONS_KEPLER)

/**
 * @brief   Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *          in CUDA on Nvidia @b Kepler GPUs and newer (sm_30).
 *
 * @tparam tile_size The size of the tile to be processed.
 * @tparam synchronize Whether to synchronize threads within the tile processing.
 * @tparam may_be_diagonal Whether the tile may contain diagonal elements.
 *
 * @param c The output tile.
 * @param a The first input tile.
 * @param b The second input tile.
 * @param bi Row index within the tile.
 * @param bj Column index within the tile.
 * @param c_row Row index of the output tile in the global matrix.
 * @param c_col Column index of the output tile in the global matrix.
 * @param a_row Row index of the first input tile in the global matrix.
 * @param a_col Column index of the first input tile in the global matrix.
 * @param b_row Row index of the second input tile in the global matrix.
 * @param b_col Column index of the second input tile in the global matrix.
 */
template <std::uint32_t tile_size, bool synchronize = true, bool may_be_diagonal = true>
__forceinline__ __device__ void process_tile_cuda_( //
    votes_count_tile<tile_size>& c,                 //
    votes_count_tile<tile_size> const& a,           //
    votes_count_tile<tile_size> const& b,           //
    candidate_idx_t bi, candidate_idx_t bj,         //
    candidate_idx_t c_row, candidate_idx_t c_col,   //
    candidate_idx_t a_row, candidate_idx_t a_col,   //
    candidate_idx_t b_row, candidate_idx_t b_col) {

    votes_count_t& c_cell = c[bi][bj];

#pragma unroll tile_size
    for (candidate_idx_t k = 0; k < tile_size; k++) {
        votes_count_t smallest = umin(a[bi][k], b[k][bj]);
        if constexpr (may_be_diagonal) {
            std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            std::uint32_t is_bigger = smallest > c_cell;
            std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            // On Kepler an newer we can use `__funnelshift_lc` to avoid branches
            c_cell = __funnelshift_lc(c_cell, smallest, will_replace - 1);
        } else
            c_cell = umax(c_cell, smallest);
        if constexpr (synchronize)
            __syncthreads();
    }
}

#else

/**
 * @brief   Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *          in CUDA or @b HIP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @tparam synchronize Whether to synchronize threads within the tile processing.
 * @tparam may_be_diagonal Whether the tile may contain diagonal elements.
 * @param c The output tile.
 * @param a The first input tile.
 * @param b The second input tile.
 * @param bi Row index within the tile.
 * @param bj Column index within the tile.
 * @param c_row Row index of the output tile in the global matrix.
 * @param c_col Column index of the output tile in the global matrix.
 * @param a_row Row index of the first input tile in the global matrix.
 * @param a_col Column index of the first input tile in the global matrix.
 * @param b_row Row index of the second input tile in the global matrix.
 * @param b_col Column index of the second input tile in the global matrix.
 */
template <std::uint32_t tile_size, bool synchronize = true, bool may_be_diagonal = true>
__forceinline__ __device__ void process_tile_cuda_( //
    votes_count_tile<tile_size>& c,                 //
    votes_count_tile<tile_size> const& a,           //
    votes_count_tile<tile_size> const& b,           //
    candidate_idx_t bi, candidate_idx_t bj,         //
    candidate_idx_t c_row, candidate_idx_t c_col,   //
    candidate_idx_t a_row, candidate_idx_t a_col,   //
    candidate_idx_t b_row, candidate_idx_t b_col) {

    votes_count_t& c_cell = c[bi][bj];

#pragma unroll tile_size
    for (candidate_idx_t k = 0; k < tile_size; k++) {
        votes_count_t smallest = min(a[bi][k], b[k][bj]);
        if constexpr (may_be_diagonal) {
            std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
            std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
            std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
            std::uint32_t is_bigger = smallest > c_cell;
            std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
            if (will_replace)
                c_cell = smallest;
        } else
            c_cell = max(c_cell, smallest);
        if constexpr (synchronize)
            __syncthreads();
    }
}

#endif

/**
 * @brief Performs the diagonal step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths.
 */
template <std::uint32_t tile_size>
__global__ void cuda_diagonal_(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    alignas(16) __shared__ votes_count_t c[tile_size][tile_size];
    c[bi][bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj];

    __syncthreads();
    process_tile_cuda_<tile_size>(    //
        c, c, c, bi, bj,              //
        tile_size * k, tile_size * k, //
        tile_size * k, tile_size * k, //
        tile_size * k, tile_size * k  //
    );

    graph[k * tile_size * n + k * tile_size + bi * n + bj] = c[bi][bj];
}

/**
 * @brief Performs the partially independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths.
 */
template <std::uint32_t tile_size>
__global__ void cuda_partially_independent_(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const i = blockIdx.x;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    if (i == k)
        return;

    alignas(16) __shared__ votes_count_tile<tile_size> a;
    alignas(16) __shared__ votes_count_tile<tile_size> b;
    alignas(16) __shared__ votes_count_tile<tile_size> c;

    // Partially dependent phase (first of two)
    // Walking down within a group of adjacent columns
    c[bi][bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj];
    b[bi][bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj];

    __syncthreads();
    process_tile_cuda_<tile_size>(    //
        c, c, b, bi, bj,              //
        i * tile_size, k * tile_size, //
        i * tile_size, k * tile_size, //
        k * tile_size, k * tile_size);

    // Partially dependent phase (second of two)
    // Walking right within a group of adjacent rows
    __syncthreads();
    graph[i * tile_size * n + k * tile_size + bi * n + bj] = c[bi][bj];
    c[bi][bj] = graph[k * tile_size * n + i * tile_size + bi * n + bj];
    a[bi][bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj];

    __syncthreads();
    process_tile_cuda_<tile_size>(    //
        c, a, c, bi, bj,              //
        k * tile_size, i * tile_size, //
        k * tile_size, k * tile_size, //
        k * tile_size, i * tile_size  //
    );

    graph[k * tile_size * n + i * tile_size + bi * n + bj] = c[bi][bj];
}

/**
 * @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths.
 */
template <std::uint32_t tile_size>
__global__ void cuda_independent_(candidate_idx_t n, candidate_idx_t k, votes_count_t* graph) {
    candidate_idx_t const j = blockIdx.x;
    candidate_idx_t const i = blockIdx.y;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

    if (i == k && j == k)
        return;

    alignas(16) __shared__ votes_count_tile<tile_size> a;
    alignas(16) __shared__ votes_count_tile<tile_size> b;
    alignas(16) __shared__ votes_count_tile<tile_size> c;

    c[bi][bj] = graph[i * tile_size * n + j * tile_size + bi * n + bj];
    a[bi][bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj];
    b[bi][bj] = graph[k * tile_size * n + j * tile_size + bi * n + bj];

    __syncthreads();
    if (i == j)
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        process_tile_cuda_<tile_size, false, true>( //
            c, a, b, bi, bj,                        //
            i * tile_size, j * tile_size,           //
            i * tile_size, k * tile_size,           //
            k * tile_size, j * tile_size            //
        );
    else
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        // We also mark as "non diagonal", because the `i != j`, and in that case
        // we can avoid some branches.
        process_tile_cuda_<tile_size, false, false>( //
            c, a, b, bi, bj,                         //
            i * tile_size, j * tile_size,            //
            i * tile_size, k * tile_size,            //
            k * tile_size, j * tile_size             //
        );

    graph[i * tile_size * n + j * tile_size + bi * n + bj] = c[bi][bj];
}

/**
 * @brief Performs then independent step of the block-parallel Schulze voting algorithm in CUDA (NVIDIA Hopper only).
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param n The number of candidates.
 * @param k The index of the current tile being processed.
 * @param graph The graph of strongest paths represented as a `CUtensorMap`.
 *
 * @note This kernel uses NVIDIA-specific Tensor Memory Access (TMA) and is not available on AMD GPUs.
 */
#if !defined(SCALING_ELECTIONS_WITH_HIP)
template <std::uint32_t tile_size>
__global__ void cuda_independent_hopper_(candidate_idx_t n, candidate_idx_t k,
                                         __grid_constant__ CUtensorMap const graph) {
    candidate_idx_t const j = blockIdx.x;
    candidate_idx_t const i = blockIdx.y;
    candidate_idx_t const bi = threadIdx.y;
    candidate_idx_t const bj = threadIdx.x;

#if defined(SCALING_ELECTIONS_HOPPER)

    if (i == k && j == k)
        return;

    alignas(128) __shared__ votes_count_tile<tile_size> a;
    alignas(128) __shared__ votes_count_tile<tile_size> b;
    alignas(128) __shared__ votes_count_tile<tile_size> c;

#pragma nv_diag_suppress static_var_with_dynamic_init
    // Initialize shared memory barrier with the number of threads participating in the barrier.
    __shared__ barrier_t bar;
    if (threadIdx.x == 0) {
        // We have one thread per tile cell.
        init(&bar, tile_size * tile_size);
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    // Sync threads so initialized barrier is visible to all threads.
    __syncthreads();

    // Only the first thread in the tile invokes the bulk transfers.
    barrier_t::arrival_token token;
    if (threadIdx.x == 0) {
        // Initiate three bulk tensor copies for different part of the graph.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&c, &graph, i * tile_size, j * tile_size, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&a, &graph, i * tile_size, k * tile_size, bar);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&b, &graph, k * tile_size, j * tile_size, bar);
        // Arrive on the barrier and tell how many bytes are expected to come in.
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(c) + sizeof(a) + sizeof(b));
    } else {
        // Other threads just arrive.
        token = bar.arrive(1);
    }

    // Wait for the data to have arrived.
    // After this point we expect shared memory to contain the following data:
    //
    //  c[bi * tile_size + bj] = graph[i * tile_size * n + j * tile_size + bi * n + bj];
    //  a[bi * tile_size + bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj];
    //  b[bi * tile_size + bj] = graph[k * tile_size * n + j * tile_size + bi * n + bj];
    bar.wait(std::move(token));

    if (i == j)
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        process_tile_cuda_<tile_size, false, true>( //
            c, a, b, bi, bj,                        //
            i * tile_size, j * tile_size,           //
            i * tile_size, k * tile_size,           //
            k * tile_size, j * tile_size            //
        );
    else
        // We don't need to "synchronize", because A, C, and B tile arguments
        // are different in the independent state and will address different shared buffers.
        // We also mark as "non diagonal", because the `i != j`, and in that case
        // we can avoid some branches.
        process_tile_cuda_<tile_size, false, false>( //
            c, a, b, bi, bj,                         //
            i * tile_size, j * tile_size,            //
            i * tile_size, k * tile_size,            //
            k * tile_size, j * tile_size             //
        );

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After `syncthreads`, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&graph, i * tile_size, j * tile_size, &c);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();

        // Destroy barrier. This invalidates the memory region of the barrier. If
        // further computations were to take place in the kernel, this allows the
        // memory location of the shared memory barrier to be reused.
        // But as we are at the end, we know it will be destroyed anyways :)
        //
        //      bar.~barrier();
    }
#else
    // This is a trap :)
    if (i == 0 && j == 0 && bi == 0 && bj == 0)
        printf("This kernel is only supported on Hopper and newer GPUs\n");
#endif
}
#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

#if !defined(SCALING_ELECTIONS_WITH_HIP)
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get pointer to cuGetProcAddress
    cudaDriverEntryPointQueryResult driver_status;
    void* cuGetProcAddress_ptr = nullptr;
    cudaError_t error =
        cudaGetDriverEntryPoint("cuGetProcAddress", &cuGetProcAddress_ptr, cudaEnableDefault, &driver_status);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress");
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress entry point");
    PFN_cuGetProcAddress_v12000 cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(cuGetProcAddress_ptr);

    // Use cuGetProcAddress to get a pointer to the CTK 12.0 version of cuTensorMapEncodeTiled
    CUdriverProcAddressQueryResult symbol_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUresult res = cuGetProcAddress("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000,
                                    CU_GET_PROC_ADDRESS_DEFAULT, &symbol_status);
    if (res != CUDA_SUCCESS || symbol_status != CU_GET_PROC_ADDRESS_SUCCESS)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
#endif // !defined(SCALING_ELECTIONS_WITH_HIP)

/**
 * @brief Computes the strongest paths for the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @param preferences The preferences matrix.
 * @param num_candidates The number of candidates.
 * @param row_stride The stride between rows in the preferences matrix.
 * @param graph The output matrix of strongest paths.
 */
template <std::uint32_t tile_size> //
void compute_strongest_paths_cuda( //
    votes_count_t* preferences, candidate_idx_t num_candidates, candidate_idx_t row_stride, votes_count_t* graph,
    bool allow_tma) {

#if defined(SCALING_ELECTIONS_WITH_OPENMP)
#pragma omp parallel for collapse(2)
#endif
    for (candidate_idx_t i = 0; i < num_candidates; i++)
        for (candidate_idx_t j = 0; j < num_candidates; j++)
            if (i != j)
                graph[i * num_candidates + j] = preferences[i * row_stride + j] > preferences[j * row_stride + i] //
                                                    ? preferences[i * row_stride + j]
                                                    : 0;

    // Check if we can use newer CUDA features.
    cudaError_t error;
    int current_device;
    cudaDeviceProp device_props;
    error = cudaGetDevice(&current_device);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to get current device");
    error = cudaGetDeviceProperties(&device_props, current_device);
    if (error != cudaSuccess)
        throw std::runtime_error("Failed to get device properties");

#if !defined(SCALING_ELECTIONS_WITH_HIP)
    bool supports_tma = device_props.major >= 9;

    CUtensorMap strongest_paths_tensor_map{};
    // rank is the number of dimensions of the array.
    constexpr std::uint32_t rank = 2;
    uint64_t size[rank] = {num_candidates, num_candidates};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride[rank - 1] = {num_candidates * sizeof(votes_count_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    std::uint32_t box_size[rank] = {tile_size, tile_size};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    std::uint32_t elem_stride[rank] = {1, 1};

    // Create the tensor descriptor.
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUresult res = cuTensorMapEncodeTiled( //
        &strongest_paths_tensor_map,       // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        rank,        // cuuint32_t tensorRank,
        graph,       // void *globalAddress,
        size,        // const cuuint64_t *globalDim,
        stride,      // const cuuint64_t *globalStrides,
        box_size,    // const cuuint32_t *boxDim,
        elem_stride, // const cuuint32_t *elementStrides,
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
#else
    // HIP/AMD GPUs don't support Tensor Memory Access (TMA)
    bool supports_tma = false;
#endif

    candidate_idx_t tiles_count = (num_candidates + tile_size - 1) / tile_size;
    dim3 tile_shape(tile_size, tile_size, 1);
    dim3 independent_grid(tiles_count, tiles_count, 1);
    for (candidate_idx_t k = 0; k < tiles_count; k++) {
        cuda_diagonal_<tile_size><<<1, tile_shape>>>(num_candidates, k, graph);
        cuda_partially_independent_<tile_size><<<tiles_count, tile_shape>>>(num_candidates, k, graph);
#if !defined(SCALING_ELECTIONS_WITH_HIP)
        if (supports_tma && allow_tma)
            cuda_independent_hopper_<tile_size>
                <<<independent_grid, tile_shape>>>(num_candidates, k, strongest_paths_tensor_map);
        else
#endif
            cuda_independent_<tile_size><<<independent_grid, tile_shape>>>(num_candidates, k, graph);

        error = cudaGetLastError();
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
    }
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion CUDA

#pragma region OpenMP
/**
 * @brief   Processes a tile of the preferences matrix for the block-parallel Schulze
 *          voting algorithm on CPU using @b OpenMP.
 *
 * @tparam tile_size The size of the tile to be processed.
 * @tparam synchronize Whether to synchronize threads within the tile processing.
 * @tparam may_be_diagonal Whether the tile may contain diagonal elements.
 * @param c The output tile.
 * @param a The first input tile.
 * @param b The second input tile.
 * @param bi Row index within the tile.
 * @param bj Column index within the tile.
 * @param c_row Row index of the output tile in the global matrix.
 * @param c_col Column index of the output tile in the global matrix.
 * @param a_row Row index of the first input tile in the global matrix.
 * @param a_col Column index of the first input tile in the global matrix.
 * @param b_row Row index of the second input tile in the global matrix.
 * @param b_col Column index of the second input tile in the global matrix.
 */
template <std::uint32_t tile_size, bool may_be_diagonal = true>
inline void process_tile_openmp_(                 //
    votes_count_tile<tile_size>& c,               //
    votes_count_tile<tile_size> const& a,         //
    votes_count_tile<tile_size> const& b,         //
    candidate_idx_t c_row, candidate_idx_t c_col, //
    candidate_idx_t a_row, candidate_idx_t a_col, //
    candidate_idx_t b_row, candidate_idx_t b_col) {

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<votes_count_t, std::uint32_t>() && tile_size % 4 == 0) {
        uint32x4_t bj_step = {0, 1, 2, 3};
        for (candidate_idx_t k = 0; k < tile_size; k++) {
            uint32x4_t b_row_plus_k_vec = vdupq_n_u32(b_row + k);
            for (candidate_idx_t bi = 0; bi < tile_size; bi++) {
                uint32x4_t a_vec = vdupq_n_u32(a[bi][k]);
                uint32x4_t is_not_diagonal_a = vdupq_n_u32((a_row + bi) != (a_col + k));
                uint32x4_t c_row_plus_bi_vec = vdupq_n_u32(c_row + bi);
#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#pragma clang loop unroll(full)
#else
#pragma unroll full
#endif
                for (candidate_idx_t bj = 0; bj < tile_size; bj += 4) {
                    votes_count_t* c_ptr = &c[bi][bj];
                    uint32x4_t c_vec = vld1q_u32(c_ptr);
                    uint32x4_t b_vec = vld1q_u32(&b[k][bj]);
                    uint32x4_t smallest = vminq_u32(a_vec, b_vec);

                    if constexpr (may_be_diagonal) {
                        uint32x4_t is_diagonal_c =       //
                            vceqq_u32(c_row_plus_bi_vec, //
                                      vaddq_u32(vdupq_n_u32(c_col + bj), bj_step));
                        uint32x4_t is_diagonal_b =      //
                            vceqq_u32(b_row_plus_k_vec, //
                                      vaddq_u32(vdupq_n_u32(b_col + bj), bj_step));
                        uint32x4_t is_bigger = vcgtq_u32(smallest, c_vec);
                        uint32x4_t will_replace =                                   //
                            vandq_u32(                                              //
                                vmvnq_u32(vorrq_u32(is_diagonal_c, is_diagonal_b)), //
                                vandq_u32(is_not_diagonal_a, is_bigger));
                        c_vec = vbslq_u32(smallest, c_vec, will_replace);
                    } else {
                        c_vec = vmaxq_u32(c_vec, smallest);
                    }
                    vst1q_u32(c_ptr, c_vec);
                }
            }
        }
        return;
    }
#else
    for (candidate_idx_t k = 0; k < tile_size; k++) {
        for (candidate_idx_t bi = 0; bi < tile_size; bi++) {
            votes_count_t* const c_cells = &c[bi][0];
#pragma omp simd
            for (candidate_idx_t bj = 0; bj < tile_size; bj++) {
                votes_count_t c_cell = c_cells[bj];
                votes_count_t smallest = std::min(a[bi][k], b[k][bj]);
                if constexpr (may_be_diagonal) {
                    std::uint32_t is_not_diagonal_c = (c_row + bi) != (c_col + bj);
                    std::uint32_t is_not_diagonal_a = (a_row + bi) != (a_col + k);
                    std::uint32_t is_not_diagonal_b = (b_row + k) != (b_col + bj);
                    std::uint32_t is_bigger = smallest > c_cell;
                    std::uint32_t will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger;
                    c_cells[bj] = will_replace ? smallest : c_cell;
                } else {
                    c_cells[bj] = std::max(c_cell, smallest);
                }
            }
        }
    }
#endif
}

template <std::uint32_t tile_size, bool check_tail = false>
void memcpy2d(votes_count_t const* source, candidate_idx_t stride, votes_count_tile<tile_size>& target,
              candidate_idx_t remaining_rows, candidate_idx_t remaining_cols) {

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<votes_count_t, std::uint32_t>() && tile_size % 4 == 0 && !check_tail) {
        for (candidate_idx_t i = 0; i < tile_size; i++) {
#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#pragma clang loop unroll(full)
#else
#pragma unroll full
#endif
            for (candidate_idx_t j = 0; j < tile_size; j += 4) {
                vst1q_u32(&target[i][j], vld1q_u32(&source[i * stride + j]));
            }
        }
        return;
    }
#else

    for (candidate_idx_t i = 0; i < tile_size; i++)
        for (candidate_idx_t j = 0; j < tile_size; j++)
            if constexpr (check_tail) {
                if (i < remaining_rows && j < remaining_cols)
                    target[i][j] = source[i * stride + j];
            } else
                target[i][j] = source[i * stride + j];
#endif
}

template <std::uint32_t tile_size, bool check_tail = false>
void memcpy2d(votes_count_tile<tile_size> const& source, candidate_idx_t stride, votes_count_t* target,
              candidate_idx_t remaining_rows, candidate_idx_t remaining_cols) {

#if defined(SCALING_ELECTIONS_WITH_NEON)
    if constexpr (std::is_same<votes_count_t, std::uint32_t>() && tile_size % 4 == 0 && !check_tail) {
        for (candidate_idx_t i = 0; i < tile_size; i++) {
#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#pragma clang loop unroll(full)
#else
#pragma unroll full
#endif
            for (candidate_idx_t j = 0; j < tile_size; j += 4) {
                vst1q_u32(&target[i * stride + j], vld1q_u32(&source[i][j]));
            }
        }
        return;
    }
#else
    for (candidate_idx_t i = 0; i < tile_size; i++)
        for (candidate_idx_t j = 0; j < tile_size; j++)
            if constexpr (check_tail) {
                if (i < remaining_rows && j < remaining_cols)
                    target[i * stride + j] = source[i][j];
            } else
                target[i * stride + j] = source[i][j];
#endif
}

template <std::uint32_t tile_size, bool check_tail = false> //
void compute_strongest_paths_openmp(                        //
    votes_count_t* preferences, candidate_idx_t num_candidates, candidate_idx_t row_stride, votes_count_t* graph) {

#pragma omp parallel for schedule(dynamic) collapse(2)
    // Populate the strongest paths matrix based on direct comparisons
    for (candidate_idx_t i = 0; i < num_candidates; i++)
        for (candidate_idx_t j = 0; j < num_candidates; j++)
            if (i != j)
                graph[i * num_candidates + j] =                                       //
                    preferences[i * row_stride + j] > preferences[j * row_stride + i] //
                        ? preferences[i * row_stride + j]
                        : 0;
            else
                graph[i * num_candidates + j] = 0;

    // Time for the actual core implementation
    candidate_idx_t const tiles_count = (num_candidates + tile_size - 1) / tile_size;
    for (candidate_idx_t k = 0; k < tiles_count; k++) {

        if (global_signal_status != 0)
            throw std::runtime_error("Stopped by signal");

        // Dependent phase
        {
            alignas(64) votes_count_t c[tile_size][tile_size];
            memcpy2d<tile_size, check_tail>(graph + k * tile_size * num_candidates + k * tile_size, num_candidates, c,
                                            num_candidates - k * tile_size, num_candidates - k * tile_size);
            process_tile_openmp_<tile_size>(  //
                c, c, c,                      //
                tile_size * k, tile_size * k, //
                tile_size * k, tile_size * k, //
                tile_size * k, tile_size * k  //
            );
            memcpy2d<tile_size, check_tail>(c, num_candidates, graph + k * tile_size * num_candidates + k * tile_size,
                                            num_candidates - k * tile_size, num_candidates - k * tile_size);
        }
        // Partially dependent phase (first of two)
#pragma omp parallel for schedule(dynamic)
        for (candidate_idx_t i = 0; i < tiles_count; i++) {
            if (i == k)
                continue;
            alignas(64) votes_count_tile<tile_size> b;
            alignas(64) votes_count_tile<tile_size> c;
            memcpy2d<tile_size, check_tail>(graph + i * tile_size * num_candidates + k * tile_size, num_candidates, c,
                                            num_candidates - i * tile_size, num_candidates - k * tile_size);
            memcpy2d<tile_size, check_tail>(graph + k * tile_size * num_candidates + k * tile_size, num_candidates, b,
                                            num_candidates - k * tile_size, num_candidates - k * tile_size);
            process_tile_openmp_<tile_size>(  //
                c, c, b,                      //
                i * tile_size, k * tile_size, //
                i * tile_size, k * tile_size, //
                k * tile_size, k * tile_size);
            memcpy2d<tile_size, check_tail>(c, num_candidates, graph + i * tile_size * num_candidates + k * tile_size,
                                            num_candidates - i * tile_size, num_candidates - k * tile_size);
        }
        // Partially dependent phase (second of two)
#pragma omp parallel for schedule(dynamic)
        for (candidate_idx_t j = 0; j < tiles_count; j++) {
            if (j == k)
                continue;
            alignas(64) votes_count_tile<tile_size> a;
            alignas(64) votes_count_tile<tile_size> c;
            memcpy2d<tile_size, check_tail>(graph + k * tile_size * num_candidates + j * tile_size, num_candidates, c,
                                            num_candidates - k * tile_size, num_candidates - j * tile_size);
            memcpy2d<tile_size, check_tail>(graph + k * tile_size * num_candidates + k * tile_size, num_candidates, a,
                                            num_candidates - k * tile_size, num_candidates - k * tile_size);
            process_tile_openmp_<tile_size>(  //
                c, a, c,                      //
                k * tile_size, j * tile_size, //
                k * tile_size, k * tile_size, //
                k * tile_size, j * tile_size  //
            );
            memcpy2d<tile_size, check_tail>(c, num_candidates, graph + k * tile_size * num_candidates + j * tile_size,
                                            num_candidates - k * tile_size, num_candidates - j * tile_size);
        }
        // Independent phase
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (candidate_idx_t i = 0; i < tiles_count; i++) {
            for (candidate_idx_t j = 0; j < tiles_count; j++) {
                if (i == k || j == k)
                    continue;
                alignas(64) votes_count_tile<tile_size> a;
                alignas(64) votes_count_tile<tile_size> b;
                alignas(64) votes_count_tile<tile_size> c;
                memcpy2d<tile_size, check_tail>(graph + i * tile_size * num_candidates + j * tile_size, num_candidates,
                                                c, num_candidates - i * tile_size, num_candidates - j * tile_size);
                memcpy2d<tile_size, check_tail>(graph + i * tile_size * num_candidates + k * tile_size, num_candidates,
                                                a, num_candidates - i * tile_size, num_candidates - k * tile_size);
                memcpy2d<tile_size, check_tail>(graph + k * tile_size * num_candidates + j * tile_size, num_candidates,
                                                b, num_candidates - k * tile_size, num_candidates - j * tile_size);
                if (i != j)
                    process_tile_openmp_<tile_size, false>( //
                        c, a, b,                            //
                        i * tile_size, j * tile_size,       //
                        i * tile_size, k * tile_size,       //
                        k * tile_size, j * tile_size        //
                    );
                else
                    process_tile_openmp_<tile_size, true>( //
                        c, a, b,                           //
                        i * tile_size, j * tile_size,      //
                        i * tile_size, k * tile_size,      //
                        k * tile_size, j * tile_size       //
                    );
                memcpy2d<tile_size, check_tail>(c, num_candidates,
                                                graph + i * tile_size * num_candidates + j * tile_size,
                                                num_candidates - i * tile_size, num_candidates - j * tile_size);
            }
        }
    }
}

#pragma endregion OpenMP

#pragma region Python bindings
#if !defined(SCALING_ELECTIONS_TEST)

/**
 * @brief Computes the strongest paths for the block-parallel Schulze voting algorithm.
 *
 * @param preferences The preferences matrix.
 * @param allow_tma Whether to use Tensor Memory Access (TMA) for the computation.
 * @return A NumPy array containing the strongest paths matrix.
 */
static py::array_t<votes_count_t> compute_strongest_paths(      //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    bool allow_tma, bool allow_gpu, std::size_t tile_size = 0) {

    auto buf = preferences.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");
    if (buf.shape[0] != buf.shape[1])
        throw std::runtime_error("Preferences matrix must be square");
    auto preferences_ptr = reinterpret_cast<votes_count_t*>(buf.ptr);
    auto num_candidates = static_cast<candidate_idx_t>(buf.shape[0]);
    auto row_stride = static_cast<candidate_idx_t>(buf.strides[0] / sizeof(votes_count_t));

    // Allocate NumPy array for the result
    auto result = py::array_t<votes_count_t>({num_candidates, num_candidates});
    auto result_buf = result.request();
    auto result_ptr = reinterpret_cast<votes_count_t*>(result_buf.ptr);
    auto result_row_stride = static_cast<candidate_idx_t>(result_buf.strides[0] / sizeof(votes_count_t));
    if (result_row_stride != num_candidates)
        throw std::runtime_error("Result matrix must be contiguous");

#if defined(SCALING_ELECTIONS_WITH_CUDA)

    if (allow_gpu) {
        votes_count_t* strongest_paths_ptr = nullptr;
        cudaError_t error;
        error = cudaMallocManaged(&strongest_paths_ptr, num_candidates * num_candidates * sizeof(votes_count_t));
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to allocate memory on device");

        using cuda_kernel_t = void (*)(votes_count_t*, candidate_idx_t, candidate_idx_t, votes_count_t*, bool);
        cuda_kernel_t cuda_kernel = nullptr;
        switch (tile_size) {
        case 4: cuda_kernel = &compute_strongest_paths_cuda<4>; break;
        case 8: cuda_kernel = &compute_strongest_paths_cuda<8>; break;
        case 16: cuda_kernel = &compute_strongest_paths_cuda<16>; break;
        case 32: cuda_kernel = &compute_strongest_paths_cuda<32>; break;
        default: throw std::runtime_error("Unsupported tile size");
        }

        cudaMemset(strongest_paths_ptr, 0, num_candidates * num_candidates * sizeof(votes_count_t));
        cuda_kernel(preferences_ptr, num_candidates, row_stride, strongest_paths_ptr, allow_tma);

        // Synchronize to ensure all CUDA operations are complete
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cudaFree(strongest_paths_ptr);
            throw std::runtime_error("CUDA operations did not complete successfully");
        }

        // Copy data from the GPU to the NumPy array
        error = cudaMemcpy(result_ptr, strongest_paths_ptr, num_candidates * num_candidates * sizeof(votes_count_t),
                           cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            cudaFree(strongest_paths_ptr);
            throw std::runtime_error("Failed to copy data from device to host");
        }

        // Synchronize to ensure all CUDA transfers are complete
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cudaFree(strongest_paths_ptr);
            throw std::runtime_error("CUDA transfers did not complete successfully");
        }

        // Free the GPU memory
        error = cudaFree(strongest_paths_ptr);
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to free memory on device");
        return result;
    }
#endif // defined(__NVCC__)

#if defined(SCALING_ELECTIONS_WITH_OPENMP)
    omp_set_dynamic(0); // ? Explicitly disable dynamic teams
    omp_set_num_threads(std::thread::hardware_concurrency());
#endif

    // Probe for the largest possible tile size, if not previously specified
    using kernel_t = void (*)(votes_count_t*, candidate_idx_t, candidate_idx_t, votes_count_t*);
    struct {
        std::size_t tile_size;
        kernel_t aligned_kernel;
        kernel_t unaligned_kernel;
    } tiled_kernels[] = {
        {4, compute_strongest_paths_openmp<4, false>, compute_strongest_paths_openmp<4, true>},
        {8, compute_strongest_paths_openmp<8, false>, compute_strongest_paths_openmp<8, true>},
        {16, compute_strongest_paths_openmp<16, false>, compute_strongest_paths_openmp<16, true>},
        {32, compute_strongest_paths_openmp<32, false>, compute_strongest_paths_openmp<32, true>},
        {64, compute_strongest_paths_openmp<64, false>, compute_strongest_paths_openmp<64, true>},
        {128, compute_strongest_paths_openmp<128, false>, compute_strongest_paths_openmp<128, true>},
    };
    kernel_t aligned_kernel = nullptr;
    kernel_t unaligned_kernel = nullptr;
    if (tile_size == 0) {
        for (auto const& kernel : tiled_kernels) {
            if (num_candidates >= kernel.tile_size) {
                tile_size = kernel.tile_size;
                aligned_kernel = kernel.aligned_kernel;
                unaligned_kernel = kernel.unaligned_kernel;
                break;
            }
        }
        if (tile_size == 0)
            throw std::runtime_error("Number of candidates should be at least 4, ideally divisible by 4");
    } else {
        if (tile_size > num_candidates)
            throw std::runtime_error("Tile size should be less than or equal to the number of candidates");
        for (auto const& kernel : tiled_kernels) {
            if (tile_size == kernel.tile_size) {
                aligned_kernel = kernel.aligned_kernel;
                unaligned_kernel = kernel.unaligned_kernel;
                break;
            }
        }
        if (aligned_kernel == nullptr)
            throw std::runtime_error("Unsupported tile size");
    }

    // Check if we can use the aligned kernel
    bool is_aligned = num_candidates % tile_size == 0;
    if (is_aligned)
        aligned_kernel(preferences_ptr, num_candidates, row_stride, result_ptr);
    else
        unaligned_kernel(preferences_ptr, num_candidates, row_stride, result_ptr);
    return result;
}

PYBIND11_MODULE(scaling_elections, m) {

    std::signal(SIGINT, signal_handler);

    // Let's show how to wrap `void` functions for basic logging
    m.def("log_gpus", []() {
#if defined(SCALING_ELECTIONS_WITH_CUDA)
        int device_count;
        cudaDeviceProp device_props;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess)
            throw std::runtime_error("Failed to get device count");
        for (int i = 0; i < device_count; i++) {
            error = cudaGetDeviceProperties(&device_props, i);
            if (error != cudaSuccess)
                throw std::runtime_error("Failed to get device properties");
            std::printf("Device %d: %s\n", i, device_props.name);
            std::printf("\tSMs: %d\n", device_props.multiProcessorCount);
            std::printf("\tGlobal mem: %.2fGB\n",
                        static_cast<float>(device_props.totalGlobalMem) / (1024 * 1024 * 1024));
            std::printf("\tCUDA Cap: %d.%d\n", device_props.major, device_props.minor);
        }
#else
        throw std::runtime_error("No CUDA devices available\n");
#endif
    });

    // This is how we could have used `thrust::` for higher-level operations
    m.def("reduce", [](py::array_t<float> const& data) -> float {
#if defined(SCALING_ELECTIONS_WITH_CUDA) && !defined(SCALING_ELECTIONS_WITH_HIP)
        // Thrust support - CUDA only (rocThrust not guaranteed to be available)
        py::buffer_info buf = data.request();
        if (buf.ndim != 1 || buf.strides[0] != sizeof(float))
            throw std::runtime_error("Input should be a contiguous 1D float array");
        float* ptr = static_cast<float*>(buf.ptr);
        thrust::device_vector<float> d_data(ptr, ptr + buf.size);
        return thrust::reduce(thrust::device, d_data.begin(), d_data.end(), 0.0f);
#else
        // CPU fallback for HIP and non-CUDA builds
        return std::accumulate(data.data(), data.data() + data.size(), 0.0f);
#endif
    });

    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("allow_tma") = false,                        //
          py::arg("allow_gpu") = false,                        //
          py::arg("tile_size") = 0);
}

#endif // !defined(SCALING_ELECTIONS_TEST)
#pragma endregion Python bindings

#if defined(SCALING_ELECTIONS_TEST)

int main() {

    std::size_t num_candidates = 256;
    std::vector<votes_count_t> preferences(num_candidates * num_candidates);
    std::generate(preferences.begin(), preferences.end(),
                  [=]() { return static_cast<votes_count_t>(std::rand() % num_candidates); });

    std::vector<votes_count_t> graph(num_candidates * num_candidates);
    compute_strongest_paths_openmp<64>(preferences.data(), num_candidates, num_candidates, graph.data());

    return 0;
}

#endif // defined(SCALING_ELECTIONS_TEST)