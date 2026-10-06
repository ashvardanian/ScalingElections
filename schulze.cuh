/**
 *  @brief Block-parallel Schulze strongest-paths kernels for CUDA, HIP, and OpenMP.
 *  @file schulze.cuh
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#pragma once
#include "types.cuh"

#pragma region Graphs

/** Which graph the strongest-paths sweep closes over. */
enum class seed_graph_t : std::uint8_t {
    /** Winning votes, the variant Schulze runs on here. */
    winning_votes_k,
    /** Positive margins, which is what Split Cycle is defined on. */
    positive_margins_k,
};

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
inline void winning_votes_graph_(strided_matrix<stored_count_type_ const> preferences,
                                 strided_matrix<arithmetic_type_> graph) {

    candidate_index_t const num_candidates = preferences.rows;
#pragma omp parallel for collapse(2)
    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++)
            graph(row, column) = row != column && preferences(row, column) > preferences(column, row)
                                     ? static_cast<arithmetic_type_>(preferences(row, column))
                                     : arithmetic_type_ {0};
}

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
inline void positive_margins_graph_(strided_matrix<stored_count_type_ const> preferences,
                                    strided_matrix<arithmetic_type_> graph) {

    candidate_index_t const num_candidates = preferences.rows;
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
template <seed_graph_t seed_, typename stored_count_type_, tally_arithmetic arithmetic_type_>
inline void seed_graph_(strided_matrix<stored_count_type_ const> preferences, strided_matrix<arithmetic_type_> graph) {
    if constexpr (seed_ == seed_graph_t::winning_votes_k) winning_votes_graph_(preferences, graph);
    else positive_margins_graph_(preferences, graph);
}

/**
 *  @brief The widest edge the seeded graph will hold, which bounds every path.
 *
 *  The max-min recurrence never widens an edge. Winning votes are bounded by the raw counts,
 *  positive margins by the counts' differences.
 */
template <seed_graph_t seed_, typename stored_count_type_>
inline voter_weight_t schulze_widest_edge_(strided_matrix<stored_count_type_ const> preferences) {
    candidate_index_t const num_candidates = preferences.rows;
    voter_weight_t widest = 0;
    for (candidate_index_t row = 0; row < num_candidates; row++)
        for (candidate_index_t column = 0; column < num_candidates; column++) {
            if (row == column) continue;
            voter_weight_t const forward = preferences(row, column);
            voter_weight_t const backward = preferences(column, row);
            if constexpr (seed_ == seed_graph_t::winning_votes_k) widest = std::max(widest, forward);
            else widest = std::max(widest, forward > backward ? forward - backward : 0);
        }
    return widest;
}

/** Resolves the automatic path type from the widest seeded edge, and validates an explicit request. */
template <seed_graph_t seed_, typename stored_count_type_>
inline score_type_t schulze_resolve_score_type(strided_matrix<stored_count_type_ const> preferences,
                                               score_type_t requested) {
    candidate_index_t const num_candidates = preferences.rows;
    if (requested == score_type_t::saturated64_k)
        for (candidate_index_t row = 0; row < num_candidates; row++)
            for (candidate_index_t column = 0; column < num_candidates; column++)
                if (preferences(row, column) == std::numeric_limits<std::uint64_t>::max())
                    throw std::overflow_error("Schulze input reaches the overflow sentinel");
    voter_weight_t const widest = schulze_widest_edge_<seed_>(preferences);
    voter_weight_t limit = 0;
    switch (requested) {
    case score_type_t::auto_k:
        return widest <= std::numeric_limits<std::uint32_t>::max() ? score_type_t::uint32_k : score_type_t::uint64_k;
    case score_type_t::saturated64_k:
    case score_type_t::uint64_k: return requested;
    case score_type_t::uint32_k: limit = std::numeric_limits<std::uint32_t>::max(); break;
    case score_type_t::uint16_k: limit = std::numeric_limits<std::uint16_t>::max(); break;
    }
    if (widest > limit) throw std::overflow_error("Schulze edge exceeds the selected arithmetic type");
    return requested;
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
 *
 *  @see https://arxiv.org/abs/2004.02350
 */
template <typename stored_count_type_, tally_arithmetic arithmetic_type_>
inline std::vector<candidate_index_t> select_split_cycle_winners(strided_matrix<stored_count_type_ const> preferences,
                                                                 strided_matrix<arithmetic_type_> margin_paths) {

    candidate_index_t const num_candidates = preferences.rows;
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

#pragma endregion Graphs

template <std::uint32_t tile_size_, typename element_type_>
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

/** The cells of tile (@p tile_row, @p tile_column), clipped where the tile hangs past the matrix edge. */
template <std::uint32_t tile_size_, typename element_type_>
inline strided_matrix<element_type_> tile_view_(strided_matrix<element_type_> graph, candidate_index_t tile_row,
                                                candidate_index_t tile_column) noexcept {
    std::size_t const row = static_cast<std::size_t>(tile_row) * tile_size_;
    std::size_t const column = static_cast<std::size_t>(tile_column) * tile_size_;
    return {&graph(row, column), std::min<std::size_t>(graph.rows - row, tile_size_),
            std::min<std::size_t>(graph.columns - column, tile_size_), graph.stride};
}

#pragma region CUDA

#if defined(SCALING_ELECTIONS_WITH_GPU)

#if defined(SCALING_ELECTIONS_WITH_CUDA)
namespace cde = cuda::device::experimental;
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#endif

#if defined(SCALING_ELECTIONS_KEPLER)

/**
 *  @brief Processes a tile of the preferences matrix for the block-parallel Schulze voting algorithm
 *      in CUDA on Nvidia @b Kepler GPUs and newer.
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
            // A funnel shift clamped to 0 or 32 bits picks one operand without a branch.
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
 *  @brief Performs the partially dependent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
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

    // The tile in the pivot column, relaxed through the pivot tile.
    paths[row][column] = graph(tile_index * tile_size_ + row, pivot_tile * tile_size_ + column);
    from_pivot[row][column] = graph(pivot_tile * tile_size_ + row, pivot_tile * tile_size_ + column);

    __syncthreads();
    process_tile_cuda_<tile_size_, tile_phase_t::aliased_k>( //
        paths, paths, from_pivot, tile_origin_t {tile_index, pivot_tile, pivot_tile}, row, column);

    // The tile in the pivot row, relaxed through the pivot tile.
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
 *  @brief Performs the independent step of the block-parallel Schulze voting algorithm in CUDA or @b HIP.
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

#if defined(SCALING_ELECTIONS_WITH_CUDA)

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

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#pragma endregion Packed Sixteen Bit

/**
 *  @brief Performs the independent step of the block-parallel Schulze voting algorithm on NVIDIA Hopper and newer.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] pivot_tile The index of the current tile being processed.
 *  @param[inout] graph The graph of strongest paths, as a @c CUtensorMap .
 *
 *  @note Loads and stores go through the Tensor Memory Accelerator, which AMD GPUs lack.
 */
#if defined(SCALING_ELECTIONS_WITH_CUDA)
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
    __shared__ barrier_t tile_barrier;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&tile_barrier, tile_size_ * tile_size_);
        // The bulk copies arrive through the async proxy, which must see the initialized barrier.
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    barrier_t::arrival_token token;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // The first coordinate is the column, as dimension 0 is the contiguous one.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&paths, &graph, tile_column * tile_size_, tile_row * tile_size_,
                                                      tile_barrier);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&to_pivot, &graph, pivot_tile * tile_size_, tile_row * tile_size_,
                                                      tile_barrier);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&from_pivot, &graph, tile_column * tile_size_,
                                                      pivot_tile * tile_size_, tile_barrier);
        token = cuda::device::barrier_arrive_tx(tile_barrier, 1, sizeof(paths) + sizeof(to_pivot) + sizeof(from_pivot));
    }
    else { token = tile_barrier.arrive(1); }
    tile_barrier.wait(std::move(token));

    tile_origin_t const origin {tile_row, tile_column, pivot_tile};
    if (tile_row == tile_column)
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
            paths, to_pivot, from_pivot, origin, row, column);
    else
        process_tile_cuda_<tile_size_, tile_phase_t::distinct_independent_k>( //
            paths, to_pivot, from_pivot, origin, row, column);

    // Generic-proxy writes must be fenced before the bulk store reads them back out.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&graph, tile_column * tile_size_, tile_row * tile_size_, &paths);
        // Shared memory must outlive the bulk store's reads of it.
        cde::cp_async_bulk_commit_group();
        cde::cp_async_bulk_wait_group_read<0>();
    }
#else
    if (tile_row == 0 && tile_column == 0 && row == 0 && column == 0)
        printf("This kernel is only supported on Hopper and newer GPUs\n");
#endif
}
#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#if defined(SCALING_ELECTIONS_WITH_CUDA)

/** Resolves the CUDA 12.0 `cuTensorMapEncodeTiled` through the runtime, sparing a link against the driver. */
inline PFN_cuTensorMapEncodeTiled_v12000 tensor_map_encoder_() {
    cudaDriverEntryPointQueryResult driver_status;
    void* get_proc_address = nullptr;
    cudaError_t error = cudaGetDriverEntryPoint("cuGetProcAddress", &get_proc_address, cudaEnableDefault,
                                                &driver_status);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get cuGetProcAddress");
    if (driver_status != cudaDriverEntryPointSuccess)
        throw std::runtime_error("Failed to get cuGetProcAddress entry point");
    auto const cuGetProcAddress = reinterpret_cast<PFN_cuGetProcAddress_v12000>(get_proc_address);

    CUdriverProcAddressQueryResult symbol_status;
    void* encoder = nullptr;
    CUresult encode_status = cuGetProcAddress("cuTensorMapEncodeTiled", &encoder, 12000, CU_GET_PROC_ADDRESS_DEFAULT,
                                              &symbol_status);
    if (encode_status != CUDA_SUCCESS || symbol_status != CU_GET_PROC_ADDRESS_SUCCESS)
        throw std::runtime_error("Failed to get cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(encoder);
}

/** Tensor map for the Hopper bulk-tensor path, absent when the device or the layout rules it out. */
using tma_descriptor_t = std::optional<CUtensorMap>;

/**
 *  @brief Builds the tensor map describing the padded strongest-paths matrix.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @param[in] graph The padded matrix of strongest paths.
 *  @param[in] device_properties Properties of the device the kernels will run on.
 *  @return The descriptor, having thrown if the device or the layout cannot supply one.
 *
 *  @see https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
 */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(strided_matrix<std::uint32_t> graph, cudaDeviceProp const& device_properties) {
    if (device_properties.major < 9)
        throw std::runtime_error(std::format("The Hopper kernel needs compute capability 9.0, found {}.{}",
                                             device_properties.major, device_properties.minor));

    CUtensorMap descriptor_map {};
    candidate_index_t const graph_stride = graph.stride;
    constexpr std::uint32_t rank_k = 2;
    uint64_t size[rank_k] = {graph_stride, graph_stride};
    // Row strides must be a multiple of 16 bytes.
    uint64_t stride[rank_k - 1] = {graph_stride * sizeof(std::uint32_t)};
    std::uint32_t box_size[rank_k] = {tile_size_, tile_size_};
    std::uint32_t element_stride[rank_k] = {1, 1};

    PFN_cuTensorMapEncodeTiled_v12000 encode = tensor_map_encoder_();
    CUresult encode_status = encode(                                                              //
        &descriptor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32, rank_k, graph.data, //
        size, stride, box_size, element_stride,                                                   //
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
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
void launch_independent_(dim3 grid, dim3 block, candidate_index_t pivot_tile, strided_matrix<std::uint32_t> graph,
                         tma_descriptor_t const& tma) {
    if (tma) schulze_independent_hopper_cuda_<tile_size_><<<grid, block>>>(pivot_tile, *tma);
    else schulze_independent_cuda_<tile_size_><<<grid, block>>>(pivot_tile, graph);
}

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
 *  @brief Whether the current device runs the packed 16-bit min-max natively.
 *
 *  Below sm_90 the packed intrinsics are emulated and lose to the 32-bit path, so both the
 *  compiled architectures and the running device have to offer the real instruction.
 */
template <std::uint32_t tile_size_>
bool packed_min_max_available_() {
    if constexpr (tile_size_ % 4 != 0) return false;
    else {
        cudaDeviceProp const device_properties = current_device_properties_();
        return carries_native_sm90_code_(device_properties) && device_properties.major >= 9;
    }
}

/** Runs every pivot step on a 16-bit graph, with the independent phase packed two per word. */
template <std::uint32_t tile_size_>
void sweep_narrow_(strided_matrix<std::uint16_t> graph) {
    candidate_index_t const graph_stride = graph.rows;
    candidate_index_t const tiles_count = graph_stride / tile_size_;
    strided_matrix<std::uint32_t> const packed = strided_view<std::uint32_t>(
        reinterpret_cast<std::uint32_t*>(graph.data), graph_stride, graph_stride / 2, graph_stride / 2);
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

/** HIP has no bulk-tensor engine, so the descriptor type can never hold one. */
using tma_descriptor_t = std::nullopt_t;

template <std::uint32_t tile_size_>
void launch_independent_(dim3 grid, dim3 block, candidate_index_t pivot_tile, strided_matrix<std::uint32_t> graph,
                         tma_descriptor_t const&) {
    schulze_independent_cuda_<tile_size_><<<grid, block>>>(pivot_tile, graph);
}

/** HIP has no bulk-tensor engine, so the descriptor can never be built. */
template <std::uint32_t tile_size_>
tma_descriptor_t require_tma_(strided_matrix<std::uint32_t>, cudaDeviceProp const&) {
    throw std::runtime_error("The Hopper kernel is unavailable in a HIP build");
}

/** HIP never carries NVIDIA's sm_90 instructions. */
constexpr bool carries_native_sm90_code_(cudaDeviceProp const&) { return false; }

/** HIP has no packed 16-bit min-max. */
template <std::uint32_t tile_size_>
bool packed_min_max_available_() {
    return false;
}

/** HIP has no packed 16-bit min-max, so the narrow sweep can never run. */
template <std::uint32_t tile_size_>
void sweep_narrow_(strided_matrix<std::uint16_t>) {
    throw std::runtime_error("The packed 16-bit path is unavailable in a HIP build");
}

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

/** Runs every pivot step on the full-width graph, using TMA when a tensor map is available. */
template <std::uint32_t tile_size_, tally_arithmetic arithmetic_type_>
void sweep_wide_(strided_matrix<arithmetic_type_> graph, tma_descriptor_t const& tma) {
    candidate_index_t const tiles_count = graph.rows / tile_size_;
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

/**
 *  @brief Picks the engine for one padded graph, then runs every pivot step over it.
 *
 *  A 16-bit graph uses the packed min-max where the device runs it natively, and a 32-bit graph
 *  uses TMA on sm_90. Other widths and devices use the ordinary tiled sweep.
 */
template <std::uint32_t tile_size_, tally_arithmetic arithmetic_type_>
void sweep_graph_(strided_matrix<arithmetic_type_> graph) {
    using arithmetic_t = arithmetic_type_;
    if constexpr (std::is_same_v<arithmetic_t, std::uint16_t>)
        if (packed_min_max_available_<tile_size_>()) return sweep_narrow_<tile_size_>(graph);

    tma_descriptor_t tma {std::nullopt};
    if constexpr (std::is_same_v<arithmetic_t, std::uint32_t>) {
        cudaDeviceProp const device_properties = current_device_properties_();
        if (carries_native_sm90_code_(device_properties) && device_properties.major >= 9)
            tma = require_tma_<tile_size_>(graph, device_properties);
    }
    sweep_wide_<tile_size_>(graph, tma);
}

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm in CUDA or @b HIP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam seed_ Which graph the sweep closes over.
 *  @tparam arithmetic_type_ The width the device sweeps in, which may be narrower than @p paths .
 *  @param[in] preferences The preferences matrix.
 *  @param[out] paths The matrix of strongest paths, one cell per candidate pair.
 *
 *  The sweep runs on a managed copy padded to a whole number of tiles, so the kernels need no tail
 *  checks: the padding is zero, which is the identity of the max-min semiring.
 */
template <std::uint32_t tile_size_, seed_graph_t seed_, tally_arithmetic arithmetic_type_, typename stored_count_type_,
          typename output_count_type_>
void compute_strongest_paths_gpu(strided_matrix<stored_count_type_ const> preferences,
                                 strided_matrix<output_count_type_> paths) {
    using arithmetic_t = arithmetic_type_;
    using output_count_t = output_count_type_;
    candidate_index_t const num_candidates = preferences.rows;
    candidate_index_t const stride = divide_round_up(num_candidates, tile_size_) * tile_size_;
    managed_vector<arithmetic_t> graph(checked_product(stride, stride));
    if (cudaMemset(graph.data(), 0, checked_product(graph.size(), sizeof(arithmetic_t))) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("Failed to clear device memory");

    seed_graph_<seed_>(preferences, square_view(graph.data(), num_candidates, stride));
    sweep_graph_<tile_size_>(square_view(graph.data(), stride, stride));
    if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA solver did not complete");

    if constexpr (std::is_same_v<arithmetic_t, output_count_t>) {
        if (cudaMemcpy2D(paths.data, paths.stride * sizeof(output_count_t), graph.data(),
                         std::size_t(stride) * sizeof(arithmetic_t), std::size_t(num_candidates) * sizeof(arithmetic_t),
                         num_candidates, cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Failed to copy paths from device");
    }
    else {
        // The managed pages fault back to the host, and each cell widens on its way out.
        strided_matrix<arithmetic_t const> const swept = square_view<arithmetic_t const>(graph.data(), num_candidates,
                                                                                         stride);
        for (candidate_index_t row = 0; row < num_candidates; row++)
            std::copy_n(&swept(row, 0), num_candidates, &paths(row, 0));
    }
}

#else

template <std::uint32_t tile_size_>
bool packed_min_max_available_() {
    return false;
}

template <std::uint32_t tile_size_, seed_graph_t seed_, tally_arithmetic arithmetic_type_, typename stored_count_type_,
          typename output_count_type_>
void compute_strongest_paths_gpu(strided_matrix<stored_count_type_ const>, strided_matrix<output_count_type_>) {
    throw std::runtime_error("This build has no GPU support compiled in");
}

#endif // defined(SCALING_ELECTIONS_WITH_GPU)

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

/** Stages the clipped tile @p source into @p target , zero-filling whatever the clip leaves out. */
template <std::uint32_t tile_size_, tally_arithmetic arithmetic_type_>
void load_tile_(strided_matrix<arithmetic_type_> source, votes_count_tile<tile_size_, arithmetic_type_>& target) {
    using arithmetic_t = arithmetic_type_;
    for (std::size_t row = 0; row < source.rows; row++) {
        std::copy_n(&source(row, 0), source.columns, target[row]);
        std::fill(target[row] + source.columns, target[row] + tile_size_, arithmetic_t {0});
    }
    for (std::size_t row = source.rows; row < tile_size_; row++)
        std::fill(target[row], target[row] + tile_size_, arithmetic_t {0});
}

/** Writes back the part of @p source that the clipped tile @p target covers. */
template <std::uint32_t tile_size_, tally_arithmetic arithmetic_type_>
void store_tile_(votes_count_tile<tile_size_, arithmetic_type_> const& source,
                 strided_matrix<arithmetic_type_> target) {
    for (std::size_t row = 0; row < target.rows; row++) std::copy_n(source[row], target.columns, &target(row, 0));
}

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm using @b OpenMP.
 *
 *  @tparam tile_size_ The size of the tile to be processed.
 *  @tparam seed_ Which graph the sweep closes over.
 *  @param[in] preferences The preferences matrix.
 *  @param[out] graph The output matrix of strongest paths, packed to the candidate count per row.
 *  @param[in] cancelled Polled between pivots, aborting the run once it reads non-zero.
 */
template <std::uint32_t tile_size_, seed_graph_t seed_, tally_arithmetic arithmetic_type_,
          typename stored_count_type_>
void compute_strongest_paths_tiled_cpu( //
    strided_matrix<stored_count_type_ const> preferences, strided_matrix<arithmetic_type_> graph,
    cancellation_t cancelled = nullptr) {
    using arithmetic_t = arithmetic_type_;

    seed_graph_<seed_>(preferences, graph);

    candidate_index_t const num_candidates = preferences.rows;
    candidate_index_t const tiles_count = divide_round_up(num_candidates, tile_size_);
    for (candidate_index_t pivot_tile = 0; pivot_tile < tiles_count; pivot_tile++) {
        throw_if_cancelled_(cancelled);

        // Dependent phase: the pivot tile itself.
        {
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
            load_tile_<tile_size_>(tile_view_<tile_size_>(graph, pivot_tile, pivot_tile), paths);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                paths, paths, paths, tile_origin_t {pivot_tile, pivot_tile, pivot_tile});
            store_tile_<tile_size_>(paths, tile_view_<tile_size_>(graph, pivot_tile, pivot_tile));
        }
        // Partially dependent phase: the pivot column.
#pragma omp parallel for schedule(dynamic)
        for (candidate_index_t tile_row = 0; tile_row < tiles_count; tile_row++) {
            if (tile_row == pivot_tile) continue;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> from_pivot;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
            load_tile_<tile_size_>(tile_view_<tile_size_>(graph, tile_row, pivot_tile), paths);
            load_tile_<tile_size_>(tile_view_<tile_size_>(graph, pivot_tile, pivot_tile), from_pivot);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                paths, paths, from_pivot, tile_origin_t {tile_row, pivot_tile, pivot_tile});
            store_tile_<tile_size_>(paths, tile_view_<tile_size_>(graph, tile_row, pivot_tile));
        }
        // Partially dependent phase: the pivot row.
#pragma omp parallel for schedule(dynamic)
        for (candidate_index_t tile_column = 0; tile_column < tiles_count; tile_column++) {
            if (tile_column == pivot_tile) continue;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> to_pivot;
            alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
            load_tile_<tile_size_>(tile_view_<tile_size_>(graph, pivot_tile, tile_column), paths);
            load_tile_<tile_size_>(tile_view_<tile_size_>(graph, pivot_tile, pivot_tile), to_pivot);
            process_tile_openmp_<tile_size_, tile_phase_t::aliased_k>( //
                paths, to_pivot, paths, tile_origin_t {pivot_tile, tile_column, pivot_tile});
            store_tile_<tile_size_>(paths, tile_view_<tile_size_>(graph, pivot_tile, tile_column));
        }
        // Independent phase: every tile off the pivot row and column.
#pragma omp parallel for schedule(dynamic) collapse(2)
        for (candidate_index_t tile_row = 0; tile_row < tiles_count; tile_row++) {
            for (candidate_index_t tile_column = 0; tile_column < tiles_count; tile_column++) {
                if (tile_row == pivot_tile || tile_column == pivot_tile) continue;
                alignas(64) votes_count_tile<tile_size_, arithmetic_t> to_pivot;
                alignas(64) votes_count_tile<tile_size_, arithmetic_t> from_pivot;
                alignas(64) votes_count_tile<tile_size_, arithmetic_t> paths;
                load_tile_<tile_size_>(tile_view_<tile_size_>(graph, tile_row, tile_column), paths);
                load_tile_<tile_size_>(tile_view_<tile_size_>(graph, tile_row, pivot_tile), to_pivot);
                load_tile_<tile_size_>(tile_view_<tile_size_>(graph, pivot_tile, tile_column), from_pivot);
                tile_origin_t const origin {tile_row, tile_column, pivot_tile};
                if (tile_row != tile_column)
                    process_tile_openmp_<tile_size_, tile_phase_t::distinct_independent_k>( //
                        paths, to_pivot, from_pivot, origin);
                else
                    process_tile_openmp_<tile_size_, tile_phase_t::distinct_diagonal_k>( //
                        paths, to_pivot, from_pivot, origin);
                store_tile_<tile_size_>(paths, tile_view_<tile_size_>(graph, tile_row, tile_column));
            }
        }
    }
}

#pragma endregion OpenMP
