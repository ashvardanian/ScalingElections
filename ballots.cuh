/**
 *  @brief Dense and ragged ballot tallies into pairwise preference, indifference and unknown matrices.
 *  @file ballots.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "types.cuh"

#pragma region Dense Tally

/**
 *  @brief Adds one chunk of complete rankings into an existing pairwise matrix.
 *
 *  @param[in] rankings One chunk of complete rankings, best candidate first.
 *  @param[inout] preferences The matrix to accumulate into, which the caller zeroes first.
 *  @param[in] cancelled Polled once per ballot, aborting the tally once it reads non-zero.
 *
 *  Accumulating rather than assigning is what lets an electorate arrive in chunks instead of
 *  having to sit in memory all at once.
 */
template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_cpu(ballots_t rankings, strided_matrix<arithmetic_type_> preferences,
                              cancellation_t cancelled = nullptr) {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const num_candidates = preferences.rows;
    std::size_t const num_ballots = rankings.rows;
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
#pragma omp parallel
    {
        std::vector<arithmetic_t> private_counts(cells, arithmetic_t {0});
#pragma omp for schedule(static)
        for (std::ptrdiff_t ballot = 0; ballot < static_cast<std::ptrdiff_t>(num_ballots); ballot++) {
            // An OpenMP loop cannot throw, so a cancelled tally skips its remaining ballots instead.
            if (cancelled && *cancelled) continue;
            candidate_index_t const* const ranking = &rankings(ballot, 0);
            for (candidate_index_t position = 0; position + 1 < num_candidates; position++) {
                candidate_index_t const preferred = ranking[position];
                for (candidate_index_t later = position + 1; later < num_candidates; later++)
                    private_counts[std::size_t(preferred) * num_candidates + ranking[later]] += arithmetic_t {1};
            }
        }
#pragma omp critical
        for (candidate_index_t row = 0; row < num_candidates; row++)
            for (candidate_index_t column = 0; column < num_candidates; column++)
                preferences(row, column) += private_counts[std::size_t(row) * num_candidates + column];
    }
    throw_if_cancelled_(cancelled);
}

#if defined(SCALING_ELECTIONS_WITH_GPU)

/** Shared bytes one tally block needs: a private matrix, plus the ballot each of its warps is staging. */
template <tally_arithmetic arithmetic_type_>
inline std::size_t tally_shared_bytes_(candidate_index_t num_candidates, int block_size, int warp_size) noexcept {
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
    std::size_t const staged = static_cast<std::size_t>(block_size / warp_size) * num_candidates;
    return cells * sizeof(arithmetic_type_) + staged * sizeof(candidate_index_t);
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
template <tally_arithmetic arithmetic_type_>
__global__ void tally_ballots_cuda_(ballots_t rankings, strided_matrix<arithmetic_type_> preferences) {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const num_candidates = preferences.rows;
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;

    extern __shared__ alignas(16) std::byte tally_shared_[];
    arithmetic_t* const counters = reinterpret_cast<arithmetic_t*>(tally_shared_);
    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x) counters[cell] = arithmetic_t {0};
    __syncthreads();

    unsigned const warps_per_block = blockDim.x / warpSize;
    unsigned const warp = threadIdx.x / warpSize;
    unsigned const lane = threadIdx.x % warpSize;
    candidate_index_t* const staged = reinterpret_cast<candidate_index_t*>(counters + cells) + warp * num_candidates;

    std::size_t const stride = static_cast<std::size_t>(gridDim.x) * warps_per_block;
    std::size_t const first = static_cast<std::size_t>(blockIdx.x) * warps_per_block + warp;
    for (std::size_t ballot = first; ballot < rankings.rows; ballot += stride) {
        for (candidate_index_t position = lane; position < num_candidates; position += warpSize)
            staged[position] = rankings(ballot, position);
        __syncwarp();

        for (candidate_index_t position = 0; position + 1 < num_candidates; position++) {
            candidate_index_t const preferred = staged[position];
            for (candidate_index_t later = position + 1 + lane; later < num_candidates; later += warpSize)
                atomic_add_relaxed<atomic_scope_t::block_k>(&counters[preferred * num_candidates + staged[later]],
                                                            arithmetic_t {1});
        }
        __syncwarp();
    }
    __syncthreads();

    for (std::size_t cell = threadIdx.x; cell < cells; cell += blockDim.x)
        if (counters[cell] != arithmetic_t {0})
            atomic_add_relaxed<atomic_scope_t::device_k>(&preferences(cell / num_candidates, cell % num_candidates),
                                                         counters[cell]);
}

/** Whether a one-warp block of the dense kernel fits its private matrix in shared memory on the current device. */
template <tally_arithmetic arithmetic_type_>
inline bool tally_dense_fits_(candidate_index_t num_candidates) {
    cudaDeviceProp const device_properties = current_device_properties_();
    return tally_shared_bytes_<arithmetic_type_>(num_candidates, device_properties.warpSize,
                                                 device_properties.warpSize) <=
           static_cast<std::size_t>(device_properties.sharedMemPerBlock);
}

/** Whether a kernel can already read @p pointer, which spares the chunk its staging copy. */
inline bool device_can_read_(void const* pointer) noexcept {
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
template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_gpu(ballots_t rankings, strided_matrix<arithmetic_type_> preferences) {
    using arithmetic_t = arithmetic_type_;
    candidate_index_t const num_candidates = preferences.rows;
    std::size_t const num_ballots = rankings.rows;
    if (!num_ballots) return;
    if (!tally_dense_fits_<arithmetic_t>(num_candidates))
        throw std::invalid_argument(
            std::format("A tally over {} candidates needs more shared memory than a block can hold", num_candidates));

    std::size_t const ballot_stride = rankings.stride;
    if (!device_can_read_(rankings.data)) {
        // A driver copy lands the chunk on the device outright, where a host loop would leave the
        // kernel to fault every page in one at a time.
        managed_vector<candidate_index_t> staging((num_ballots - 1) * ballot_stride + num_candidates);
        if (cudaMemcpy(staging.data(), rankings.data, staging.size() * sizeof(candidate_index_t),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Failed to copy ballots to the device");
        return tally_ballots_gpu(strided_view<candidate_index_t const, std::size_t>(staging.data(), num_ballots,
                                                                                    num_candidates, ballot_stride),
                                 preferences);
    }

    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
    managed_vector<arithmetic_t> device_counts(cells);
    std::fill(device_counts.begin(), device_counts.end(), arithmetic_t {0});

    // The shared footprint grows with the warps per block, so the block size is chosen against it.
    int const warp_size = current_device_properties_().warpSize;
    auto const shared_bytes_of = [&](int block_size) {
        return tally_shared_bytes_<arithmetic_t>(num_candidates, block_size, warp_size);
    };
    int minimum_grid = 0;
    int block_size = 0;
    if (cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minimum_grid, &block_size, tally_ballots_cuda_<arithmetic_t>,
                                                       shared_bytes_of) != cudaSuccess)
        throw std::runtime_error("Failed to size the tally kernel");
    std::size_t const shared_bytes = shared_bytes_of(block_size);
    std::size_t const wanted_blocks = divide_round_up(num_ballots, static_cast<std::size_t>(block_size / warp_size));
    unsigned const blocks = static_cast<unsigned>(std::min<std::size_t>(
        wanted_blocks, resident_blocks_(tally_ballots_cuda_<arithmetic_t>, block_size, shared_bytes)));
    tally_ballots_cuda_<<<blocks, block_size, shared_bytes>>>(
        rankings, square_view(device_counts.data(), num_candidates, num_candidates));

    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("CUDA operations did not complete successfully");

    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++)
            preferences(row, column) += device_counts.data()[std::size_t(row) * num_candidates + column];
}

#else

template <tally_arithmetic arithmetic_type_>
inline bool tally_dense_fits_(candidate_index_t) {
    throw std::runtime_error("This build has no GPU support compiled in");
}

template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_gpu(ballots_t, strided_matrix<arithmetic_type_>) {
    throw std::runtime_error("This build has no GPU support compiled in");
}

#endif // defined(SCALING_ELECTIONS_WITH_GPU)

/** Counters the device can add to atomically, which leaves out anything narrower than a 32-bit word. */
template <typename arithmetic_type_>
concept device_tally_arithmetic = tally_arithmetic<arithmetic_type_> && sizeof(arithmetic_type_) >= 4;

/** Adds one chunk of complete rankings into an existing matrix, on whichever processor was named. */
template <tally_arithmetic arithmetic_type_>
inline void tally_ballots(ballots_t rankings, strided_matrix<arithmetic_type_> preferences, backend_t backend,
                          cancellation_t cancelled = nullptr) {
    switch (backend) {
    case backend_t::cpu_k: return tally_ballots_cpu(rankings, preferences, cancelled);
    case backend_t::gpu_k:
        if constexpr (device_tally_arithmetic<arithmetic_type_>) return tally_ballots_gpu(rankings, preferences);
        else throw std::invalid_argument("A GPU tally counts in 32 bits or wider");
    }
}

#pragma endregion Dense Tally

#pragma region Ragged Tally

/** How a ballot treats the candidates it leaves out. */
enum class unranked_t : std::uint8_t {
    /** Comparisons against an omitted candidate are unspecified. */
    unknown_k = 0,
    /** Omitted candidates rank below every listed one and tie among themselves. */
    worse_k = 1,
};

/** The comparison counted for each ordered candidate pair. */
enum class pairwise_relation_t : std::uint8_t {
    /** The row candidate is strictly preferred. */
    preference_k = 0,
    /** The candidates share a rank. */
    indifference_k = 1,
    /** The ballot leaves the comparison unspecified. */
    unknown_k = 2,
    /** All three comparison matrices in preference, indifference, unknown order. */
    all_k = 3,
};

/** How many square matrices a tally of @p relation fills. */
SCALING_ELECTIONS_HOST_DEVICE constexpr std::size_t relation_planes_(pairwise_relation_t relation) noexcept {
    switch (relation) {
    case pairwise_relation_t::all_k: return 3;
    default: return 1;
    }
}

/** Marks a candidate the ballot leaves out. */
constexpr rank_position_t unranked_position_k = std::numeric_limits<rank_position_t>::max();

SCALING_ELECTIONS_HOST_DEVICE inline pairwise_relation_t classify_relation_(rank_position_t left, rank_position_t right,
                                                                            unranked_t unranked) {
    if (unranked == unranked_t::unknown_k && (left == unranked_position_k || right == unranked_position_k))
        return pairwise_relation_t::unknown_k;
    if (left == right) return pairwise_relation_t::indifference_k;
    return pairwise_relation_t::preference_k;
}

/** Borrowed CSR ballot inputs, valid for the duration of a tally call. */
struct ragged_ballots {
    /** Flat candidate IDs, distinct within each ballot. */
    candidate_index_t const* candidates;
    /** Ballot boundaries, including the terminal entry offset. */
    ballot_offset_t const* offsets;
    /** Equal labels tie; null uses entry positions as strict ranks. */
    rank_label_t const* ranks;
    /** Integer voter weights; null assigns unit weight. */
    voter_weight_t const* weights;
    /** Number of rows in the ballot input. */
    std::size_t num_ballots;
    /** Size of the candidate universe, including omitted candidates. */
    candidate_index_t num_candidates;
    /** Omission policy used when per-voter policies are absent. */
    unranked_t unranked;
    /** Optional per-voter omission policies. */
    unranked_t const* policies;

    SCALING_ELECTIONS_HOST_DEVICE unranked_t unranked_at(std::size_t ballot) const noexcept {
        return policies ? policies[ballot] : unranked;
    }
};

/** Rejects offsets that do not start at zero or that decrease, and IDs out of range or repeated within a ballot. */
inline void validate_ragged_ballots_(ragged_ballots const& ballots) {
    if (ballots.offsets[0] != 0) throw std::invalid_argument("Offsets must span the candidate entries");
    std::vector<std::size_t> seen(ballots.num_candidates, std::numeric_limits<std::size_t>::max());
    for (std::size_t ballot = 0; ballot < ballots.num_ballots; ++ballot) {
        ballot_offset_t const first = ballots.offsets[ballot];
        ballot_offset_t const last = ballots.offsets[ballot + 1];
        if (last < first) throw std::invalid_argument("Offsets must be nondecreasing");
        for (ballot_offset_t entry = first; entry < last; ++entry) {
            candidate_index_t const candidate = ballots.candidates[entry];
            if (candidate >= ballots.num_candidates || seen[candidate] == ballot)
                throw std::invalid_argument("Candidate IDs must be in range and distinct within each ballot");
            seen[candidate] = ballot;
        }
    }
}

/** Resolves automatic tally arithmetic from the total voter weight, and validates an explicit request against it. */
inline score_type_t tally_resolve_score_type(ragged_ballots const& ballots, score_type_t requested) {
    saturated<voter_weight_t> const bound = ballots.weights ? std::accumulate(ballots.weights,
                                                                              ballots.weights + ballots.num_ballots,
                                                                              saturated<voter_weight_t> {0})
                                                            : saturated<voter_weight_t>(ballots.num_ballots);
    voter_weight_t limit = 0;
    switch (requested) {
    case score_type_t::auto_k:
        if (bound.value <= std::numeric_limits<std::uint32_t>::max()) return score_type_t::uint32_k;
        if (bound.value < std::numeric_limits<std::uint64_t>::max()) return score_type_t::uint64_k;
        return score_type_t::saturated64_k;
    case score_type_t::saturated64_k: return requested;
    case score_type_t::uint64_k: limit = std::numeric_limits<std::uint64_t>::max() - 1; break;
    case score_type_t::uint32_k: limit = std::numeric_limits<std::uint32_t>::max(); break;
    case score_type_t::uint16_k: limit = std::numeric_limits<std::uint16_t>::max(); break;
    }
    if (bound.value > limit) throw std::overflow_error("Tally bound exceeds the selected arithmetic type");
    return requested;
}

/**
 *  @brief Adds weighted ragged ballots into an existing matrix across an @b OpenMP team.
 *
 *  Each weight fits @p arithmetic_type_ , because the resolved type already bounds their sum.
 */
template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_cpu(ragged_ballots ballots, strided_matrix<arithmetic_type_> preferences,
                              pairwise_relation_t relation, cancellation_t cancelled = nullptr) {
    using arithmetic_t = arithmetic_type_;
    std::size_t const n = ballots.num_candidates;
    std::size_t const cells = checked_product(n, n);
    std::size_t const output_cells = checked_product(cells, relation_planes_(relation));
#pragma omp parallel
    {
        std::vector<arithmetic_t> counts(output_cells, arithmetic_t {0});
        std::vector<rank_position_t> labels(n);
#pragma omp for schedule(static)
        for (std::size_t ballot = 0; ballot < ballots.num_ballots; ++ballot) {
            // An OpenMP loop cannot throw, so a cancelled tally skips its remaining ballots instead.
            if (cancelled && *cancelled) continue;
            ballot_offset_t const first = ballots.offsets[ballot];
            ballot_offset_t const last = ballots.offsets[ballot + 1];
            arithmetic_t const weight(ballots.weights ? ballots.weights[ballot] : 1);
            if (weight == arithmetic_t {0}) continue;
            unranked_t const unranked = ballots.unranked_at(ballot);
            if (relation != pairwise_relation_t::preference_k || unranked == unranked_t::worse_k) {
                std::fill(labels.begin(), labels.end(), unranked_position_k);
                for (ballot_offset_t entry = first; entry < last; ++entry)
                    labels[ballots.candidates[entry]] = ballots.ranks ? ballots.ranks[entry] : entry - first;
            }
            if (relation != pairwise_relation_t::preference_k) {
                for (std::size_t candidate = 0; candidate < n; ++candidate)
                    for (std::size_t opponent = 0; opponent < n; ++opponent) {
                        if (candidate == opponent) continue;
                        auto const actual = classify_relation_(labels[candidate], labels[opponent], unranked);
                        if (actual == pairwise_relation_t::preference_k && labels[candidate] >= labels[opponent])
                            continue;
                        if (relation != pairwise_relation_t::all_k && relation != actual) continue;
                        std::size_t const plane = relation == pairwise_relation_t::all_k
                                                      ? static_cast<std::size_t>(actual)
                                                      : 0;
                        counts[plane * cells + candidate * n + opponent] += weight;
                    }
                continue;
            }
            for (ballot_offset_t entry = first; entry < last; ++entry) {
                std::size_t const preferred = ballots.candidates[entry];
                rank_position_t const rank = ballots.ranks ? ballots.ranks[entry] : entry - first;
                for (ballot_offset_t other = first; other < last; ++other)
                    if (rank < (ballots.ranks ? ballots.ranks[other] : other - first))
                        counts[preferred * n + ballots.candidates[other]] += weight;
                if (unranked == unranked_t::worse_k)
                    for (candidate_index_t other = 0; other < n; ++other)
                        if (labels[other] == unranked_position_k) counts[preferred * n + other] += weight;
            }
        }
#pragma omp critical
        for (std::size_t cell = 0; cell < output_cells; ++cell) preferences(cell / n, cell % n) += counts[cell];
    }
    throw_if_cancelled_(cancelled);
}

#if defined(SCALING_ELECTIONS_WITH_GPU)

template <tally_arithmetic arithmetic_type_>
__global__ void tally_ragged_ballots_cuda_(ragged_ballots ballots, arithmetic_type_* counts,
                                           pairwise_relation_t relation, rank_position_t* rank_scratch) {
    using arithmetic_t = arithmetic_type_;
    extern __shared__ std::uint32_t present[];
    std::size_t const n = ballots.num_candidates;
    std::size_t const words = divide_round_up(n, 32);
    for (std::size_t ballot = blockIdx.x; ballot < ballots.num_ballots; ballot += gridDim.x) {
        ballot_offset_t const first = ballots.offsets[ballot];
        ballot_offset_t const last = ballots.offsets[ballot + 1];
        arithmetic_t const weight(ballots.weights ? ballots.weights[ballot] : 1);
        if (weight == arithmetic_t {0}) continue;
        if (relation != pairwise_relation_t::preference_k) {
            rank_position_t* const labels = rank_scratch + std::size_t(blockIdx.x) * n;
            for (std::size_t candidate = threadIdx.x; candidate < n; candidate += blockDim.x)
                labels[candidate] = unranked_position_k;
            __syncthreads();
            for (ballot_offset_t entry = first + threadIdx.x; entry < last; entry += blockDim.x)
                labels[ballots.candidates[entry]] = ballots.ranks ? ballots.ranks[entry] : entry - first;
            __syncthreads();
            for (std::size_t candidate = threadIdx.x; candidate < n; candidate += blockDim.x)
                for (std::size_t opponent = 0; opponent < n; ++opponent) {
                    if (candidate == opponent) continue;
                    auto const actual = classify_relation_(labels[candidate], labels[opponent],
                                                           ballots.unranked_at(ballot));
                    if (actual == pairwise_relation_t::preference_k && labels[candidate] >= labels[opponent]) continue;
                    if (relation != pairwise_relation_t::all_k && relation != actual) continue;
                    std::size_t const plane = relation == pairwise_relation_t::all_k ? static_cast<std::size_t>(actual)
                                                                                     : 0;
                    atomic_add_relaxed<atomic_scope_t::device_k>(counts + (plane * n + candidate) * n + opponent,
                                                                 weight);
                }
            __syncthreads();
            continue;
        }
        if (ballots.unranked_at(ballot) == unranked_t::worse_k) {
            for (std::size_t word = threadIdx.x; word < words; word += blockDim.x) present[word] = 0;
            __syncthreads();
            for (ballot_offset_t entry = first + threadIdx.x; entry < last; entry += blockDim.x) {
                candidate_index_t const candidate = ballots.candidates[entry];
                atomic_or_relaxed<atomic_scope_t::block_k>(present + candidate / 32,
                                                           std::uint32_t {1} << (candidate % 32));
            }
            __syncthreads();
        }
        for (ballot_offset_t entry = first + threadIdx.x; entry < last; entry += blockDim.x) {
            std::size_t const preferred = ballots.candidates[entry];
            rank_position_t const rank = ballots.ranks ? ballots.ranks[entry] : entry - first;
            for (ballot_offset_t other = first; other < last; ++other)
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
                              pairwise_relation_t relation) {
    using arithmetic_t = arithmetic_type_;
    if (!ballots.num_ballots) return;
    std::size_t const entries = ballots.offsets[ballots.num_ballots];
    std::size_t const cells = checked_product(checked_product(ballots.num_candidates, ballots.num_candidates),
                                              relation_planes_(relation));
    std::size_t const shared_bytes = relation == pairwise_relation_t::preference_k &&
                                             (ballots.policies || ballots.unranked == unranked_t::worse_k)
                                         ? divide_round_up(std::size_t(ballots.num_candidates), 32) * 4
                                         : 0;
    if (shared_bytes > static_cast<std::size_t>(current_device_properties_().sharedMemPerBlock))
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
    // One block walks one ballot at a time, so a resident grid is enough, and its label scratch scales with it.
    int minimum_grid = 0;
    int block_size = 0;
    if (cudaOccupancyMaxPotentialBlockSize(&minimum_grid, &block_size, tally_ragged_ballots_cuda_<arithmetic_t>,
                                           shared_bytes) != cudaSuccess)
        throw std::runtime_error("Failed to size the tally kernel");
    unsigned const blocks = static_cast<unsigned>(std::min<std::size_t>(
        ballots.num_ballots, resident_blocks_(tally_ragged_ballots_cuda_<arithmetic_t>, block_size, shared_bytes)));
    managed_vector<rank_position_t> rank_scratch(
        relation == pairwise_relation_t::preference_k ? 0 : checked_product(blocks, ballots.num_candidates));
    tally_ragged_ballots_cuda_<<<blocks, block_size, shared_bytes>>>(device_ballots, counts.data(), relation,
                                                                     rank_scratch.data());
    cudaError_t const error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA tally did not complete");
    for (std::size_t cell = 0; cell < cells; ++cell)
        preferences(cell / ballots.num_candidates, cell % ballots.num_candidates) += counts[cell];
}

#else

template <tally_arithmetic arithmetic_type_>
inline void tally_ballots_gpu(ragged_ballots, strided_matrix<arithmetic_type_>, pairwise_relation_t) {
    throw std::runtime_error("This build has no GPU support compiled in");
}

#endif // defined(SCALING_ELECTIONS_WITH_GPU)

/** Adds weighted ragged ballots into an existing matrix, on whichever processor was named. */
template <tally_arithmetic arithmetic_type_>
inline void tally_ballots(ragged_ballots ballots, strided_matrix<arithmetic_type_> preferences, backend_t backend,
                          pairwise_relation_t relation, cancellation_t cancelled = nullptr) {
    switch (backend) {
    case backend_t::cpu_k: return tally_ballots_cpu(ballots, preferences, relation, cancelled);
    case backend_t::gpu_k:
        if constexpr (device_tally_arithmetic<arithmetic_type_>)
            return tally_ballots_gpu(ballots, preferences, relation);
        else throw std::invalid_argument("A GPU tally counts in 32 bits or wider");
    }
}

#pragma endregion Ragged Tally
