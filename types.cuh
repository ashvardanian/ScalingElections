/**
 *  @brief Shared scalar types, feature macros, and the HIP-to-CUDA shim.
 *  @file types.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include <csignal> // `std::sig_atomic_t`, `std::signal`, `std::raise`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::rand`

#include <algorithm>   // `std::min`, `std::max`, `std::reverse`
#include <span>        // `std::span`
#include <bit>         // `std::countr_zero`
#include <limits>      // `std::numeric_limits`
#include <numeric>     // `std::accumulate`
#include <optional>    // `std::optional`, `std::nullopt`
#include <stdexcept>   // `std::runtime_error`
#include <format>      // `std::format`
#include <string>      // `std::string`
#include <string_view> // `std::string_view`
#include <thread>      // `std::thread::hardware_concurrency()`
#include <type_traits> // `std::integral_constant`, `std::is_same`, `std::type_identity_t`
#include <vector>      // `std::vector`

#if defined(_OPENMP)
#include <omp.h> // `omp_set_num_threads`
#define SCALING_ELECTIONS_WITH_OPENMP (1)
#endif

#if (defined(__ARM_NEON) || defined(__aarch64__))
#define SCALING_ELECTIONS_WITH_NEON (1)
#endif

#if defined(__clang__) // Apple's Clang can't handle `#pragma unroll`
#define SCALING_ELECTIONS_UNROLL _Pragma("clang loop unroll(full)")
#else
#define SCALING_ELECTIONS_UNROLL _Pragma("unroll full")
#endif

#if defined(__NVCC__) || defined(__HIP__)
#define SCALING_ELECTIONS_HOST_DEVICE __host__ __device__
#else
#define SCALING_ELECTIONS_HOST_DEVICE
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#define SCALING_ELECTIONS_WITH_HIP (1)
#elif defined(__NVCC__)
#define SCALING_ELECTIONS_WITH_CUDA (1)
#endif
#if defined(SCALING_ELECTIONS_WITH_CUDA) || defined(SCALING_ELECTIONS_WITH_HIP)
#define SCALING_ELECTIONS_WITH_GPU (1)
#endif

#if defined(SCALING_ELECTIONS_WITH_NEON)
#include <arm_neon.h>
#endif

#if defined(SCALING_ELECTIONS_WITH_CUDA)
#include <cuda.h>         // `CUtensorMap`
#include <cuda/barrier>   // `cuda::barrier`, `cuda::device::barrier_arrive_tx`
#include <cuda/atomic>    // `cuda::atomic_ref`, `cuda::memory_order_relaxed`, `cuda::thread_scope`
#include <cudaTypedefs.h> // `PFN_cuTensorMapEncodeTiled_v12000`, `PFN_cuGetProcAddress_v12000`
#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree`, `cudaDeviceProp`, `cudaError_t`

#elif defined(SCALING_ELECTIONS_WITH_HIP)
#include <hip/hip_runtime.h> // `hipMallocManaged`, `hipFree`, `hipDeviceProp_t`, `hipError_t`

#if defined(__HIP_PLATFORM_AMD__)
#define cudaError_t                                    hipError_t
#define cudaSuccess                                    hipSuccess
#define cudaGetDevice                                  hipGetDevice
#define cudaGetDeviceProperties                        hipGetDeviceProperties
#define cudaDeviceProp                                 hipDeviceProp_t
#define cudaMallocManaged                              hipMallocManaged
#define cudaFree                                       hipFree
#define cudaMemGetInfo                                 hipMemGetInfo
#define cudaMemcpy                                     hipMemcpy
#define cudaMemcpy2D                                   hipMemcpy2D
#define cudaMemcpyDeviceToHost                         hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice                         hipMemcpyHostToDevice
#define cudaMemset                                     hipMemset
#define cudaDeviceSynchronize                          hipDeviceSynchronize
#define cudaGetLastError                               hipGetLastError
#define cudaGetErrorString                             hipGetErrorString
#define cudaGetDeviceCount                             hipGetDeviceCount
#define cudaPointerAttributes                          hipPointerAttribute_t
#define cudaPointerGetAttributes                       hipPointerGetAttributes
#define cudaMemoryTypeDevice                           hipMemoryTypeDevice
#define cudaMemoryTypeManaged                          hipMemoryTypeManaged
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor  hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxPotentialBlockSize             hipOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSizeVariableSMem hipOccupancyMaxPotentialBlockSizeVariableSMem

#endif
#endif

#if defined(SCALING_ELECTIONS_WITH_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
#define SCALING_ELECTIONS_KEPLER (1)
#endif
#if defined(SCALING_ELECTIONS_WITH_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define SCALING_ELECTIONS_HOPPER (1)
#endif

/**
 *  The tile edge every backend is compiled for, derived from the block limit: a GPU tile runs one
 *  thread per cell, and 32 squared is the 1024 threads CUDA, HIP and Metal all cap a block at.
 */
constexpr std::uint32_t tile_size_k = 32;

using rank_label_t = std::uint32_t;
using ballot_offset_t = std::uint64_t;
using voter_weight_t = std::uint64_t;
using candidate_index_t = std::uint32_t;
/** A rank label or an entry offset within a ballot, wide enough that its maximum marks an unranked candidate. */
using rank_position_t = std::uint64_t;
/** An entry of Pascal's triangle up to the widest Kemeny field, whose width sets that field. */
using binomial_count_t = std::uint32_t;
/** A set of candidates, one bit each, wide enough for every Kemeny field on every target. */
using subset_mask_t = std::uint64_t;

enum class backend_t : std::uint8_t { cpu_k, gpu_k };
enum class score_type_t : std::uint8_t { auto_k, saturated64_k, uint64_k, uint32_k, uint16_k };

/** Polled between independent steps of a CPU solver, which throws once it reads non-zero. */
using cancellation_t = volatile std::sig_atomic_t const*;

/** Thrown by a CPU solver that observed its cancellation flag. */
inline void throw_if_cancelled_(cancellation_t cancelled) {
    if (cancelled && *cancelled) throw std::runtime_error("Stopped by signal");
}

#if defined(SCALING_ELECTIONS_WITH_OPENMP)

/** The calling thread's place in the innermost OpenMP team. */
inline unsigned openmp_thread_index_() noexcept { return static_cast<unsigned>(omp_get_thread_num()); }

/** How many threads the innermost OpenMP team holds. */
inline unsigned openmp_threads_count_() noexcept { return static_cast<unsigned>(omp_get_num_threads()); }

#else

inline unsigned openmp_thread_index_() noexcept { return 0; }

inline unsigned openmp_threads_count_() noexcept { return 1; }

#endif

template <typename count_type_>
struct saturated {
    using count_t = count_type_;
    count_t value;

    saturated() = default;
    SCALING_ELECTIONS_HOST_DEVICE constexpr saturated(count_t count) noexcept : value(count) {}
    SCALING_ELECTIONS_HOST_DEVICE constexpr explicit operator count_t() const noexcept { return value; }
    SCALING_ELECTIONS_HOST_DEVICE constexpr saturated operator+(saturated other) const noexcept {
        count_t const limit = std::numeric_limits<count_t>::max();
        return value > limit - other.value ? limit : value + other.value;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr saturated& operator+=(saturated other) noexcept {
        return *this = *this + other;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr bool operator==(saturated other) const noexcept {
        return value == other.value;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr bool operator!=(saturated other) const noexcept {
        return value != other.value;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr bool operator<(saturated other) const noexcept {
        return value < other.value;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr bool operator>(saturated other) const noexcept {
        return value > other.value;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr bool operator<=(saturated other) const noexcept {
        return value <= other.value;
    }
    SCALING_ELECTIONS_HOST_DEVICE constexpr bool operator>=(saturated other) const noexcept {
        return value >= other.value;
    }
};

template <typename count_type_>
struct std::numeric_limits<saturated<count_type_>> : std::numeric_limits<count_type_> {
    SCALING_ELECTIONS_HOST_DEVICE static constexpr saturated<count_type_> max() noexcept {
        return std::numeric_limits<count_type_>::max();
    }
};

/** @brief Unsigned integer arithmetic with 2 to 64 value bits. */
template <typename scalar_type_>
concept tally_arithmetic = std::numeric_limits<scalar_type_>::is_integer &&
                           !std::numeric_limits<scalar_type_>::is_signed &&
                           (std::numeric_limits<scalar_type_>::digits > 1) &&
                           (std::numeric_limits<scalar_type_>::digits <= 64);

inline std::size_t checked_product(std::size_t count, std::size_t width) {
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
        throw std::overflow_error("Allocation size exceeds the addressable range");
    return count * width;
}

#if defined(SCALING_ELECTIONS_WITH_GPU)
enum class atomic_scope_t { block_k, device_k };

// Barriers and kernel completion publish these counters; their updates need no ordering.
template <atomic_scope_t scope_, typename count_type_>
__device__ inline void atomic_add_relaxed(count_type_* counter, count_type_ value) {
#if defined(SCALING_ELECTIONS_WITH_HIP)
    constexpr int scope = scope_ == atomic_scope_t::block_k ? __HIP_MEMORY_SCOPE_WORKGROUP : __HIP_MEMORY_SCOPE_AGENT;
    __hip_atomic_fetch_add(counter, value, __ATOMIC_RELAXED, scope);
#else
    constexpr auto scope = scope_ == atomic_scope_t::block_k ? cuda::thread_scope_block : cuda::thread_scope_device;
    cuda::atomic_ref<count_type_, scope>(*counter).fetch_add(value, cuda::memory_order_relaxed);
#endif
}

template <atomic_scope_t scope_>
__device__ inline void atomic_or_relaxed(std::uint32_t* counter, std::uint32_t value) {
#if defined(SCALING_ELECTIONS_WITH_HIP)
    constexpr int scope = scope_ == atomic_scope_t::block_k ? __HIP_MEMORY_SCOPE_WORKGROUP : __HIP_MEMORY_SCOPE_AGENT;
    __hip_atomic_fetch_or(counter, value, __ATOMIC_RELAXED, scope);
#else
    constexpr auto scope = scope_ == atomic_scope_t::block_k ? cuda::thread_scope_block : cuda::thread_scope_device;
    cuda::atomic_ref<std::uint32_t, scope>(*counter).fetch_or(value, cuda::memory_order_relaxed);
#endif
}

template <atomic_scope_t scope_>
__device__ inline void atomic_add_relaxed(saturated<std::uint64_t>* counter, saturated<std::uint64_t> value) {
#if defined(SCALING_ELECTIONS_WITH_HIP)
    constexpr int scope = scope_ == atomic_scope_t::block_k ? __HIP_MEMORY_SCOPE_WORKGROUP : __HIP_MEMORY_SCOPE_AGENT;
    auto* word = &counter->value;
    auto observed = __hip_atomic_load(word, __ATOMIC_RELAXED, scope);
    while (!__hip_atomic_compare_exchange_weak(word, &observed, (saturated<std::uint64_t>(observed) + value).value,
                                               __ATOMIC_RELAXED, __ATOMIC_RELAXED, scope)) {}
#else
    constexpr auto scope = scope_ == atomic_scope_t::block_k ? cuda::thread_scope_block : cuda::thread_scope_device;
    cuda::atomic_ref<std::uint64_t, scope> word(counter->value);
    auto observed = word.load(cuda::memory_order_relaxed);
    while (!word.compare_exchange_weak(observed, (saturated<std::uint64_t>(observed) + value).value,
                                       cuda::memory_order_relaxed, cuda::memory_order_relaxed)) {}
#endif
}
#endif

template <typename integer_type_>
SCALING_ELECTIONS_HOST_DEVICE constexpr integer_type_ divide_round_up(
    integer_type_ value, std::type_identity_t<integer_type_> divisor) noexcept {
    return value / divisor + (value % divisor != 0);
}

#pragma region Shaped Views

/** A two-dimensional view of @p rows by @p columns cells whose rows sit @p stride elements apart. */
template <typename element_type_, typename index_type_ = std::size_t>
struct strided_matrix {
    using element_t = element_type_;
    using index_t = index_type_;

    element_t* data;
    index_t rows;
    index_t columns;
    index_t stride;

    SCALING_ELECTIONS_HOST_DEVICE element_t& operator()(index_t row, index_t column) const noexcept {
        return data[static_cast<std::size_t>(row) * stride + column];
    }

    SCALING_ELECTIONS_HOST_DEVICE operator strided_matrix<element_t const, index_t>() const noexcept
        requires(!std::is_const_v<element_t>)
    {
        return {data, rows, columns, stride};
    }
};

/** A read-only view over a chunk of complete rankings, one ballot to a row, best candidate first. */
using ballots_t = strided_matrix<candidate_index_t const, std::size_t>;

/** Views @p data as @p rows by @p columns cells whose rows sit @p stride apart. */
template <typename element_type_, typename index_type_ = std::size_t>
inline strided_matrix<element_type_, index_type_> strided_view( //
    element_type_* data, std::type_identity_t<index_type_> rows, std::type_identity_t<index_type_> columns,
    std::type_identity_t<index_type_> stride) noexcept {
    return {data, rows, columns, stride};
}

/** Views @p data as @p edge by @p edge cells whose rows sit @p stride apart. */
template <typename element_type_, typename index_type_ = std::size_t>
inline strided_matrix<element_type_, index_type_> square_view( //
    element_type_* data, std::type_identity_t<index_type_> edge, std::type_identity_t<index_type_> stride) noexcept {
    return strided_view<element_type_, index_type_>(data, edge, edge, stride);
}

#pragma endregion Shaped Views

#if defined(SCALING_ELECTIONS_WITH_GPU)

/** Properties of the device the calling thread launches on. */
inline cudaDeviceProp current_device_properties_() {
    int device = 0;
    cudaDeviceProp device_properties;
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&device_properties, device) != cudaSuccess)
        throw std::runtime_error("No CUDA devices available");
    return device_properties;
}

/** Blocks of @p kernel that stay resident on the whole device at once, which is all a grid-stride launch uses. */
template <typename kernel_type_>
inline unsigned resident_blocks_(kernel_type_ kernel, int block_size, std::size_t shared_bytes) {
    cudaDeviceProp const device_properties = current_device_properties_();
    int blocks_per_multiprocessor = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_multiprocessor, kernel, block_size, shared_bytes) !=
            cudaSuccess ||
        blocks_per_multiprocessor == 0)
        throw std::runtime_error("A kernel block does not fit this device");
    return static_cast<unsigned>(
        std::min(blocks_per_multiprocessor * device_properties.multiProcessorCount, device_properties.maxGridSize[0]));
}

/** Draws from CUDA's unified memory, so one allocation is addressable from both the host and the device. */
template <typename value_type_>
struct managed_allocator {
    using value_t = value_type_;
    using value_type = value_t;

    managed_allocator() = default;
    template <typename other_type_>
    constexpr managed_allocator(managed_allocator<other_type_> const&) noexcept {}

    value_t* allocate(std::size_t count) {
        value_t* pointer = nullptr;
        if (cudaMallocManaged(&pointer, checked_product(count, sizeof(value_t))) != cudaSuccess) throw std::bad_alloc();
        return pointer;
    }

    void deallocate(value_t* pointer, std::size_t) noexcept { cudaFree(pointer); }

    /** Leaves elements uninitialized, since a host-side zero-fill would fault a device-bound table onto the host. */
    template <typename other_type_>
    void construct(other_type_*) const noexcept {
        static_assert(std::is_trivially_default_constructible_v<other_type_>, "Elements are left uninitialized");
    }

    template <typename other_type_>
    bool operator==(managed_allocator<other_type_> const&) const noexcept {
        return true;
    }
};

/** A resizable buffer both the host and the device address, whose elements start uninitialized. */
template <typename value_type_>
using managed_vector = std::vector<value_type_, managed_allocator<value_type_>>;

#endif
