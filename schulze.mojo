"""
Schulze strongest paths, from a serial reference up to a three-phase tiled GPU sweep.

The Schulze method is a Condorcet system: the winner is whoever beats every rival along the
widest chain of pairwise wins, which is a max-min closure over the winning-votes graph and so a
Floyd-Warshall in the `(max, min)` semiring rather than `(min, +)`.

Every backend here computes the same closure and differs only in how it walks it. The serial
version is the parity oracle. The tiled CPU versions block the sweep into three dependency
phases so the working set fits cache, once scalar and once with the diagonal handled by SIMD
masking. The GPU version runs those same phases as three kernels per diagonal tile, mirroring
`scalingelections.cu`.
"""

from std.builtin.sort import sort
from std.math import iota
from std.memory import AddressSpace, stack_allocation, unsafe_memcpy, unsafe_memset_zero

from max.algorithm import parallelize
from max.gpu import barrier, block_idx, thread_idx
from max.gpu.host import DeviceContext

from ballots import (
    Backend,
    PreferenceMatrix,
    ScoreType,
    SeedGraph,
    StrongestPathsMatrix,
    VoteMatrix,
    seed_graph,
)


# region Types

comptime TILE_SIZE = 32
"""The tile edge every backend compiles for, matching a warp; a tile wider than the electorate is zero-filled, not an error."""


@fieldwise_init
struct TilePhase(Copyable, Equatable, Movable):
    """Which tile phase a processor runs, fixing aliasing and diagonal handling together.

    The three states are the only combinations the recurrence produces.
    """

    var value: UInt8
    comptime aliased = Self(0)
    comptime distinct_diagonal = Self(1)
    comptime distinct_independent = Self(2)

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


@fieldwise_init
struct IndexedScore(Comparable, Copyable, Equatable, Movable):
    """Pairs a candidate index with their win count, ordered by descending score then ascending index."""

    var index: Int
    var score: Int

    def __lt__(self, other: Self) -> Bool:
        if self.score != other.score:
            return self.score > other.score
        return self.index < other.index

    def __le__(self, other: Self) -> Bool:
        return self < other or self == other

    def __eq__(self, other: Self) -> Bool:
        return self.score == other.score and self.index == other.index

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def __gt__(self, other: Self) -> Bool:
        return other < self

    def __ge__(self, other: Self) -> Bool:
        return other < self or self == other


# endregion Types


# region Serial Reference


def compute_strongest_paths_serial[
    score_dtype: DType = DType.uint32, seed: SeedGraph = SeedGraph.winning_votes
](preferences: PreferenceMatrix) raises -> VoteMatrix[score_dtype]:
    """
    Serial implementation of Schulze strongest paths computation.

    Parameters:
        seed: Which graph the closure runs over, since Split Cycle wants margins.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var strongest_paths = VoteMatrix[score_dtype](num_candidates)

    # Step 1: Initialize strongest paths
    seed_graph(preferences, strongest_paths.data, num_candidates, seed)

    # Step 2: Floyd-Warshall-like algorithm for strongest paths
    for pivot in range(num_candidates):
        for row in range(num_candidates):
            if pivot != row:
                for column in range(num_candidates):
                    if pivot != column and row != column:
                        var to_pivot = strongest_paths[row, pivot]
                        var from_pivot = strongest_paths[pivot, column]
                        var direct = strongest_paths[row, column]
                        var through_pivot = min(to_pivot, from_pivot)
                        strongest_paths[row, column] = max(direct, through_pivot)

    return strongest_paths^


# endregion Serial Reference


# region CPU Tiles


def process_tile_cpu[
    score_dtype: DType, tile_size: Int
](
    output: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    left: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    right: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    output_row: Int,
    output_column: Int,
    left_column: Int,
    right_column: Int,
    num_candidates: Int,
    tile_stride: Int,
):
    """
    CPU-optimized tile processing for blocked Schulze algorithm.

    Args:
        output: Output tile.
        left: First input tile.
        right: Second input tile.
        output_row: Row index of output tile.
        output_column: Column index of output tile.
        left_column: Column index of first input tile.
        right_column: Column index of second input tile.
        num_candidates: Total number of candidates.
        tile_stride: Stride for accessing tiles.
    """
    for step in range(tile_size):
        for tile_row in range(tile_size):
            for tile_column in range(tile_size):
                # Check bounds
                var global_row = output_row + tile_row
                var global_column = output_column + tile_column
                var global_step = left_column + step

                if global_row >= num_candidates or global_column >= num_candidates or global_step >= num_candidates:
                    continue

                # Skip diagonal elements
                if global_row == global_column or global_row == global_step or global_step == global_column:
                    continue

                var left_value = left[unsafe_offset=tile_row * tile_stride + step]
                var right_value = right[unsafe_offset=step * tile_stride + tile_column]
                var output_offset = tile_row * tile_stride + tile_column
                var output_value = output[unsafe_offset=output_offset]
                var relaxed = min(left_value, right_value)

                if relaxed > output_value:
                    output[unsafe_offset=output_offset] = relaxed


def process_tile_cpu_simd_independent[
    score_dtype: DType, tile_size: Int, simd_width: Int
](
    output: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    left: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    right: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    tile_stride: Int,
):
    """
    SIMD-vectorized tile processor for independent tiles (no diagonal checking needed).
    Processes several columns at once with SIMD vectors.

    Args:
        output: Output tile.
        left: First input tile.
        right: Second input tile.
        tile_stride: Stride for accessing tiles.
    """
    # Walk the intermediate candidate
    for step in range(tile_size):
        # Process each row
        for tile_row in range(tile_size):
            var left_value = left[unsafe_offset=tile_row * tile_stride + step]

            comptime num_simd_chunks = tile_size // simd_width

            # Process all elements with SIMD
            for chunk in range(num_simd_chunks):
                var tile_column = chunk * simd_width
                var output_offset = tile_row * tile_stride + tile_column
                var right_offset = step * tile_stride + tile_column

                # Load SIMD vectors
                var output_lanes = output.unsafe_load[width=simd_width](output_offset)
                var right_lanes = right.unsafe_load[width=simd_width](right_offset)

                # The left operand is one cell, shared by every lane
                var left_lanes = SIMD[score_dtype, simd_width](left_value)

                var narrowed = min(left_lanes, right_lanes)
                var widened = max(output_lanes, narrowed)

                # Store result
                output.unsafe_store[width=simd_width](output_offset, widened)


def process_tile_cpu_simd_diagonal[
    score_dtype: DType, tile_size: Int, simd_width: Int
](
    output: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    left: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    right: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    output_row: Int,
    output_column: Int,
    left_column: Int,
    num_candidates: Int,
    tile_stride: Int,
):
    """
    SIMD-vectorized tile processor for diagonal tiles (requires diagonal avoidance).
    Uses masking and select operations to avoid branches.

    Args:
        output: Output tile.
        left: First input tile.
        right: Second input tile.
        output_row: Global row index of output tile.
        output_column: Global column index of output tile.
        left_column: Global column index for intermediate dimension.
        num_candidates: Total number of candidates.
        tile_stride: Stride for accessing tiles.
    """
    # Walk the intermediate candidate
    for step in range(tile_size):
        var global_step = left_column + step

        # Process each row
        for tile_row in range(tile_size):
            var global_row = output_row + tile_row
            var left_value = left[unsafe_offset=tile_row * tile_stride + step]

            # Vectorized processing with diagonal masking
            comptime num_simd_chunks = tile_size // simd_width

            # Process all elements with SIMD
            for chunk in range(num_simd_chunks):
                var tile_column = chunk * simd_width
                var output_offset = tile_row * tile_stride + tile_column
                var right_offset = step * tile_stride + tile_column

                # Load SIMD vectors
                var output_lanes = output.unsafe_load[width=simd_width](output_offset)
                var right_lanes = right.unsafe_load[width=simd_width](right_offset)
                var left_lanes = SIMD[score_dtype, simd_width](left_value)

                var narrowed = min(left_lanes, right_lanes)

                # Lane `lane` covers candidate `output_column + tile_column + lane`; skip the three diagonals.
                var global_column = iota[DType.int32, simd_width]() + Int32(output_column + tile_column)
                var mask = (
                    global_column.ne(Int32(global_row))
                    & global_column.ne(Int32(global_step))
                    & narrowed.gt(output_lanes)
                    & SIMD[DType.bool, simd_width](fill=global_row != global_step)
                )
                output.unsafe_store[width=simd_width](output_offset, mask.select(narrowed, output_lanes))


def copy_tile_to_buffer[
    score_dtype: DType
](
    source: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    dest: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    start_row: Int,
    start_column: Int,
    tile_size: Int,
    num_candidates: Int,
):
    """Copy a tile from the global matrix to a local buffer."""
    for tile_row in range(tile_size):
        for tile_column in range(tile_size):
            var row = start_row + tile_row
            var column = start_column + tile_column
            if row < num_candidates and column < num_candidates:
                dest[unsafe_offset=tile_row * tile_size + tile_column] = source[
                    unsafe_offset=row * num_candidates + column
                ]
            else:
                dest[unsafe_offset=tile_row * tile_size + tile_column] = 0


def copy_buffer_to_tile[
    score_dtype: DType
](
    source: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    dest: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    start_row: Int,
    start_column: Int,
    tile_size: Int,
    num_candidates: Int,
):
    """Copy a tile from a local buffer back to the global matrix."""
    for tile_row in range(tile_size):
        for tile_column in range(tile_size):
            var row = start_row + tile_row
            var column = start_column + tile_column
            if row < num_candidates and column < num_candidates:
                dest[unsafe_offset=row * num_candidates + column] = source[
                    unsafe_offset=tile_row * tile_size + tile_column
                ]


@always_inline
def tile_origin(tile_index: Int, tile_size: Int) -> Int:
    """
    The global index a tile starts at.

    Args:
        tile_index: Index of the tile.
        tile_size: Size of each tile.

    Returns:
        The tile's first global index.
    """
    return tile_index * tile_size


# endregion CPU Tiles


# region CPU Drivers


def compute_strongest_paths_tiled_cpu[
    score_dtype: DType = DType.uint32, tile_size: Int = TILE_SIZE, seed: SeedGraph = SeedGraph.winning_votes
](preferences: PreferenceMatrix) raises -> VoteMatrix[score_dtype]:
    """
    Tiled CPU implementation of Schulze strongest paths computation.
    Uses blocking for better cache utilization.

    Parameters:
        tile_size: Compile-time tile size for CPU processing (default: 32).
        seed: Which graph the closure runs over, since Split Cycle wants margins.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var strongest_paths = VoteMatrix[score_dtype](num_candidates)

    # Step 1: Initialize strongest paths
    seed_graph(preferences, strongest_paths.data, num_candidates, seed)

    # Step 2: Tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for pivot in range(num_tiles):
        # Copied because a `parallelize` closure capturing the induction variable faults at -O1.
        var pivot_index = pivot
        var pivot_start = tile_origin(pivot, tile_size)

        # Dependent phase: process diagonal tile
        var diagonal_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

        copy_tile_to_buffer(
            strongest_paths.data,
            diagonal_tile,
            pivot_start,
            pivot_start,
            tile_size,
            num_candidates,
        )

        process_tile_cpu[score_dtype, tile_size](
            diagonal_tile,
            diagonal_tile,
            diagonal_tile,
            pivot_start,
            pivot_start,
            pivot_start,
            pivot_start,
            num_candidates,
            tile_size,
        )

        copy_buffer_to_tile(
            diagonal_tile,
            strongest_paths.data,
            pivot_start,
            pivot_start,
            tile_size,
            num_candidates,
        )

        # Partially dependent phases - row tiles
        def process_row_tiles(tile: Int) {imm}:
            if tile == pivot_index:
                return

            var tile_start = tile_origin(tile, tile_size)

            var output_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var right_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

            copy_tile_to_buffer(
                strongest_paths.data,
                output_tile,
                tile_start,
                pivot_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                right_tile,
                pivot_start,
                pivot_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu[score_dtype, tile_size](
                output_tile,
                output_tile,
                right_tile,
                tile_start,
                pivot_start,
                pivot_start,
                pivot_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                output_tile,
                strongest_paths.data,
                tile_start,
                pivot_start,
                tile_size,
                num_candidates,
            )

        parallelize(process_row_tiles, num_tiles)

        # Partially dependent phases - column tiles
        def process_col_tiles(tile: Int) {imm}:
            if tile == pivot_index:
                return

            var tile_start = tile_origin(tile, tile_size)

            var output_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var left_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

            copy_tile_to_buffer(
                strongest_paths.data,
                output_tile,
                pivot_start,
                tile_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                left_tile,
                pivot_start,
                pivot_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu[score_dtype, tile_size](
                output_tile,
                left_tile,
                output_tile,
                pivot_start,
                tile_start,
                pivot_start,
                tile_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                output_tile,
                strongest_paths.data,
                pivot_start,
                tile_start,
                tile_size,
                num_candidates,
            )

        parallelize(process_col_tiles, num_tiles)

        # Independent phase
        def process_independent_tiles(flat_index: Int) {imm}:
            var row_tile = flat_index // num_tiles
            var column_tile = flat_index % num_tiles

            if row_tile == pivot_index or column_tile == pivot_index:
                return

            var row_start = tile_origin(row_tile, tile_size)

            var column_start = tile_origin(column_tile, tile_size)

            var output_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var left_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var right_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

            copy_tile_to_buffer(
                strongest_paths.data,
                output_tile,
                row_start,
                column_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                left_tile,
                row_start,
                pivot_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                right_tile,
                pivot_start,
                column_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu[score_dtype, tile_size](
                output_tile,
                left_tile,
                right_tile,
                row_start,
                column_start,
                pivot_start,
                column_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                output_tile,
                strongest_paths.data,
                row_start,
                column_start,
                tile_size,
                num_candidates,
            )

        parallelize(process_independent_tiles, num_tiles * num_tiles)

    return strongest_paths^


def compute_strongest_paths_tiled_cpu_simd[
    score_dtype: DType = DType.uint32, tile_size: Int = TILE_SIZE, seed: SeedGraph = SeedGraph.winning_votes
](preferences: PreferenceMatrix) raises -> VoteMatrix[score_dtype]:
    """
    SIMD-vectorized tiled CPU implementation of Schulze strongest paths computation.
    Uses phase-specific SIMD tile processors for optimal vectorization and minimal branching.

    Parameters:
        tile_size: Compile-time tile size for CPU processing (default: 32).
        seed: Which graph the closure runs over, since Split Cycle wants margins.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    # Largest power-of-two SIMD width dividing the tile, capped at 16 for AVX-512.
    comptime simd_width = (
        16 if tile_size % 16
        == 0 else 8 if tile_size % 8
        == 0 else 4 if tile_size % 4
        == 0 else 2 if tile_size % 2
        == 0 else 1
    )
    var num_candidates = preferences.num_candidates
    var strongest_paths = VoteMatrix[score_dtype](num_candidates)

    # Step 1: Initialize strongest paths
    seed_graph(preferences, strongest_paths.data, num_candidates, seed)

    # Step 2: SIMD-vectorized tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for pivot in range(num_tiles):
        # Copied because a `parallelize` closure capturing the induction variable faults at -O1.
        var pivot_index = pivot
        var pivot_start = tile_origin(pivot, tile_size)

        # Diagonal phase: uses diagonal-aware SIMD processor
        var diagonal_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

        copy_tile_to_buffer(
            strongest_paths.data,
            diagonal_tile,
            pivot_start,
            pivot_start,
            tile_size,
            num_candidates,
        )

        process_tile_cpu_simd_diagonal[score_dtype, tile_size, simd_width](
            diagonal_tile,
            diagonal_tile,
            diagonal_tile,
            pivot_start,
            pivot_start,
            pivot_start,
            num_candidates,
            tile_size,
        )

        copy_buffer_to_tile(
            diagonal_tile,
            strongest_paths.data,
            pivot_start,
            pivot_start,
            tile_size,
            num_candidates,
        )

        # Partially dependent phases - row and column tiles
        def process_row_col_tiles(tile: Int) {imm}:
            if tile == pivot_index:
                return

            var tile_start = tile_origin(tile, tile_size)

            # Row tile, left of the diagonal tile
            var output_row_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var right_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

            copy_tile_to_buffer(
                strongest_paths.data,
                output_row_tile,
                tile_start,
                pivot_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                right_tile,
                pivot_start,
                pivot_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu_simd_diagonal[score_dtype, tile_size, simd_width](
                output_row_tile,
                output_row_tile,
                right_tile,
                tile_start,
                pivot_start,
                pivot_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                output_row_tile,
                strongest_paths.data,
                tile_start,
                pivot_start,
                tile_size,
                num_candidates,
            )

            # Column tile, above the diagonal tile
            var output_column_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var left_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

            copy_tile_to_buffer(
                strongest_paths.data,
                output_column_tile,
                pivot_start,
                tile_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                left_tile,
                pivot_start,
                pivot_start,
                tile_size,
                num_candidates,
            )

            process_tile_cpu_simd_diagonal[score_dtype, tile_size, simd_width](
                output_column_tile,
                left_tile,
                output_column_tile,
                pivot_start,
                tile_start,
                pivot_start,
                num_candidates,
                tile_size,
            )

            copy_buffer_to_tile(
                output_column_tile,
                strongest_paths.data,
                pivot_start,
                tile_start,
                tile_size,
                num_candidates,
            )

        parallelize(process_row_col_tiles, num_tiles)

        # Independent phase: uses fast SIMD processor (no diagonal checks)
        def process_independent_tiles(flat_index: Int) {imm}:
            var row_tile = flat_index // num_tiles
            var column_tile = flat_index % num_tiles

            if row_tile == pivot_index or column_tile == pivot_index:
                return

            var row_start = tile_origin(row_tile, tile_size)

            var column_start = tile_origin(column_tile, tile_size)

            var output_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var left_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()
            var right_tile = stack_allocation[tile_size * tile_size, SIMD[score_dtype, 1], alignment=64]()

            copy_tile_to_buffer(
                strongest_paths.data,
                output_tile,
                row_start,
                column_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                left_tile,
                row_start,
                pivot_start,
                tile_size,
                num_candidates,
            )
            copy_tile_to_buffer(
                strongest_paths.data,
                right_tile,
                pivot_start,
                column_start,
                tile_size,
                num_candidates,
            )

            # Use independent processor if not on diagonal, otherwise use diagonal processor
            if row_tile == column_tile:
                process_tile_cpu_simd_diagonal[score_dtype, tile_size, simd_width](
                    output_tile,
                    left_tile,
                    right_tile,
                    row_start,
                    column_start,
                    pivot_start,
                    num_candidates,
                    tile_size,
                )
            else:
                process_tile_cpu_simd_independent[score_dtype, tile_size, simd_width](
                    output_tile, left_tile, right_tile, tile_size
                )

            copy_buffer_to_tile(
                output_tile,
                strongest_paths.data,
                row_start,
                column_start,
                tile_size,
                num_candidates,
            )

        parallelize(process_independent_tiles, num_tiles * num_tiles)

    return strongest_paths^


# endregion CPU Drivers


# region GPU Kernels


@always_inline
def process_tile_gpu_device[
    score_dtype: DType, tile_size: Int, phase: TilePhase
](
    output_shared: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    left_shared: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    right_shared: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    output_row: Int,
    output_column: Int,
    left_row: Int,
    left_column: Int,
    right_row: Int,
    right_column: Int,
):
    """
    Core tile processing logic for GPU - runs on each thread.
    Processes one cell of the tile through every intermediate candidate.

    This matches the CUDA process_tile_cuda_ template function.
    """
    var tile_row = Int(thread_idx.y)
    var tile_column = Int(thread_idx.x)
    var output_offset = tile_row * tile_size + tile_column

    # Each thread processes one cell of the output tile
    var output_value = output_shared[unsafe_offset=output_offset]

    # Floyd-Warshall inner loop over the intermediate candidate
    for step in range(tile_size):
        var global_step = left_column + step

        var left_offset = tile_row * tile_size + step
        var right_offset = step * tile_size + tile_column

        var left_value = left_shared[unsafe_offset=left_offset]
        var right_value = right_shared[unsafe_offset=right_offset]
        var smallest = min(left_value, right_value)

        comptime if phase != TilePhase.distinct_independent:
            var global_row = output_row + tile_row
            var global_column = output_column + tile_column

            # Diagonal avoidance using branchless bit operations
            var is_not_diagonal_output = UInt32(1) if global_row != global_column else UInt32(0)
            var is_not_diagonal_left = UInt32(1) if global_row != global_step else UInt32(0)
            var is_not_diagonal_right = UInt32(1) if global_step != global_column else UInt32(0)
            var is_bigger = UInt32(1) if smallest > output_value else UInt32(0)
            var will_replace = is_not_diagonal_output & is_not_diagonal_left & is_not_diagonal_right & is_bigger

            if will_replace == 1:
                output_value = smallest
        else:
            # Non-diagonal case - simple max
            output_value = max(output_value, smallest)

        # Write back IMMEDIATELY after update - critical for correctness!
        # When left_shared/right_shared/output_shared point to the same buffer (diagonal phase),
        # threads must see updated values from earlier iterations.
        output_shared[unsafe_offset=output_offset] = output_value

        comptime if phase == TilePhase.aliased:
            barrier()


def gpu_diagonal_kernel[
    score_dtype: DType, tile_size: Int
](graph: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin], padded_edge: Int32, pivot_tile: Int32):
    """
    GPU kernel for diagonal phase - processes tile (pivot, pivot).
    Matches cuda_diagonal_ from CUDA implementation.
    """
    var stride = Int(padded_edge)
    var pivot = Int(pivot_tile)
    var tile_row = Int(thread_idx.y)
    var tile_column = Int(thread_idx.x)

    # Allocate shared memory for one tile
    var output_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()

    # Load tile from global memory
    output_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=pivot * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ]

    # Synchronize after load
    barrier()

    # Process tile (all three inputs are the same tile, need synchronization)
    process_tile_gpu_device[score_dtype, tile_size, TilePhase.aliased](
        output_shared,
        output_shared,
        output_shared,
        tile_size * pivot,
        tile_size * pivot,
        tile_size * pivot,
        tile_size * pivot,
        tile_size * pivot,
        tile_size * pivot,
    )

    # Synchronize before store
    barrier()

    # Write back to global memory
    graph[
        unsafe_offset=pivot * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ] = output_shared[unsafe_offset=tile_row * tile_size + tile_column]


def gpu_partially_independent_kernel[
    score_dtype: DType, tile_size: Int
](graph: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin], padded_edge: Int32, pivot_tile: Int32):
    """
    GPU kernel for partially independent phase.
    Processes row and column tiles relative to the diagonal tile.
    Matches cuda_partially_independent_ from CUDA.
    """
    var stride = Int(padded_edge)
    var pivot = Int(pivot_tile)
    var tile = Int(block_idx.x)
    var tile_row = Int(thread_idx.y)
    var tile_column = Int(thread_idx.x)

    if tile == pivot:
        return

    # Allocate shared memory for three tiles
    var left_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()
    var right_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()
    var output_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()

    # Phase 1: the row tile, relaxed through the diagonal tile
    output_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=tile * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ]
    right_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=pivot * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ]

    barrier()

    process_tile_gpu_device[score_dtype, tile_size, TilePhase.aliased](
        output_shared,
        output_shared,
        right_shared,
        tile * tile_size,
        pivot * tile_size,
        tile * tile_size,
        pivot * tile_size,
        pivot * tile_size,
        pivot * tile_size,
    )

    barrier()

    # Store phase 1 result
    graph[
        unsafe_offset=tile * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ] = output_shared[unsafe_offset=tile_row * tile_size + tile_column]

    # Phase 2: the column tile, relaxed through the diagonal tile
    output_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=pivot * tile_size * stride + tile * tile_size + tile_row * stride + tile_column
    ]
    left_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=pivot * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ]

    barrier()

    process_tile_gpu_device[score_dtype, tile_size, TilePhase.aliased](
        output_shared,
        left_shared,
        output_shared,
        pivot * tile_size,
        tile * tile_size,
        pivot * tile_size,
        pivot * tile_size,
        pivot * tile_size,
        tile * tile_size,
    )

    barrier()

    # Store phase 2 result
    graph[
        unsafe_offset=pivot * tile_size * stride + tile * tile_size + tile_row * stride + tile_column
    ] = output_shared[unsafe_offset=tile_row * tile_size + tile_column]


def gpu_independent_kernel[
    score_dtype: DType, tile_size: Int
](graph: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin], padded_edge: Int32, pivot_tile: Int32):
    """
    GPU kernel for independent phase - processes every tile off the pivot's row and column.
    Matches cuda_independent_ from CUDA implementation.
    """
    var stride = Int(padded_edge)
    var pivot = Int(pivot_tile)
    var column_tile = Int(block_idx.x)
    var row_tile = Int(block_idx.y)
    var tile_row = Int(thread_idx.y)
    var tile_column = Int(thread_idx.x)

    if row_tile == pivot and column_tile == pivot:
        return

    # Allocate shared memory for three tiles
    var left_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()
    var right_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()
    var output_shared = stack_allocation[
        tile_size * tile_size,
        SIMD[score_dtype, 1],
        address_space=AddressSpace.SHARED,
    ]()

    # Load the output tile and the two it relaxes through
    output_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=row_tile * tile_size * stride + column_tile * tile_size + tile_row * stride + tile_column
    ]
    left_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=row_tile * tile_size * stride + pivot * tile_size + tile_row * stride + tile_column
    ]
    right_shared[unsafe_offset=tile_row * tile_size + tile_column] = graph[
        unsafe_offset=pivot * tile_size * stride + column_tile * tile_size + tile_row * stride + tile_column
    ]

    barrier()

    # Process tile - use diagonal check if row_tile == column_tile, no synchronization needed (different tiles)
    if row_tile == column_tile:
        process_tile_gpu_device[score_dtype, tile_size, TilePhase.distinct_diagonal](
            output_shared,
            left_shared,
            right_shared,
            row_tile * tile_size,
            column_tile * tile_size,
            row_tile * tile_size,
            pivot * tile_size,
            pivot * tile_size,
            column_tile * tile_size,
        )
    else:
        process_tile_gpu_device[score_dtype, tile_size, TilePhase.distinct_independent](
            output_shared,
            left_shared,
            right_shared,
            row_tile * tile_size,
            column_tile * tile_size,
            row_tile * tile_size,
            pivot * tile_size,
            pivot * tile_size,
            column_tile * tile_size,
        )

    # No barrier needed - independent tiles write to different locations

    # Write back result
    graph[
        unsafe_offset=row_tile * tile_size * stride + column_tile * tile_size + tile_row * stride + tile_column
    ] = output_shared[unsafe_offset=tile_row * tile_size + tile_column]


# endregion GPU Kernels


# region GPU Driver


def compute_strongest_paths_gpu[
    score_dtype: DType, tile_size: Int = TILE_SIZE, seed: SeedGraph = SeedGraph.winning_votes
](preferences: PreferenceMatrix) raises -> VoteMatrix[score_dtype]:
    """
    Pure Mojo GPU implementation of Schulze strongest paths computation.

    Implements three-phase tiled Floyd-Warshall algorithm on GPU using native
    Mojo GPU kernels. Matches the CUDA implementation in scalingelections.cu.

    Parameters:
        tile_size: Compile-time tile size for GPU processing (default: 32).
        seed: Which graph the closure runs over, since Split Cycle wants margins.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var result = VoteMatrix[score_dtype](num_candidates)

    # Rounding up to a whole number of tiles keeps the kernels free of tail checks: the
    # padding is zero, which is the identity of the max-min semiring.
    var num_tiles = (num_candidates + tile_size - 1) // tile_size
    var padded = num_tiles * tile_size

    var ctx = DeviceContext()
    var host_graph = ctx.enqueue_create_host_buffer[score_dtype](padded * padded)
    var device_graph = ctx.enqueue_create_buffer[score_dtype](padded * padded)
    ctx.synchronize()
    var host_ptr = host_graph.unsafe_ptr()
    unsafe_memset_zero(host_ptr, padded * padded)

    seed_graph(preferences, host_ptr, padded, seed)

    host_graph.enqueue_copy_to(device_graph)

    var graph_ptr = device_graph.unsafe_ptr()
    var block_dim_tuple = (tile_size, tile_size, 1)

    for pivot in range(num_tiles):
        # Phase 1: Diagonal tile (sequential, 1 block)
        ctx.enqueue_function[gpu_diagonal_kernel[score_dtype, tile_size]](
            graph_ptr,
            Int32(padded),
            Int32(pivot),
            grid_dim=(1, 1, 1),
            block_dim=block_dim_tuple,
        )

        # Phase 2: Partially independent tiles (num_tiles blocks)
        ctx.enqueue_function[gpu_partially_independent_kernel[score_dtype, tile_size]](
            graph_ptr,
            Int32(padded),
            Int32(pivot),
            grid_dim=(num_tiles, 1, 1),
            block_dim=block_dim_tuple,
        )

        # Phase 3: Independent tiles (num_tiles x num_tiles blocks)
        ctx.enqueue_function[gpu_independent_kernel[score_dtype, tile_size]](
            graph_ptr,
            Int32(padded),
            Int32(pivot),
            grid_dim=(num_tiles, num_tiles, 1),
            block_dim=block_dim_tuple,
        )

    device_graph.enqueue_copy_to(host_graph)
    ctx.synchronize()

    # The answer is the leading sub-block of the padded matrix, copied a row at a time.
    for row in range(num_candidates):
        unsafe_memcpy(
            dest=result.data.unsafe_offset(row * num_candidates),
            src=host_ptr.unsafe_offset(row * padded),
            count=num_candidates,
        )
    # The untracked pointer does not keep its host allocation alive.
    deinit(host_graph^)

    return result^


# endregion GPU Driver


def resolve_score_type[
    seed: SeedGraph = SeedGraph.winning_votes
](preferences: PreferenceMatrix, requested_type: ScoreType = ScoreType.auto) raises -> ScoreType:
    """Resolves the path type before converting the graph to its storage width."""
    if requested_type == ScoreType.auto:
        return ScoreType.uint32
    if requested_type == ScoreType.uint16:
        for row in range(preferences.num_candidates):
            for column in range(preferences.num_candidates):
                var edge = preferences[row, column]
                comptime if seed == SeedGraph.positive_margins:
                    var reverse = preferences[column, row]
                    edge = edge - reverse if edge > reverse else UInt32(0)
                if edge > UInt32(UInt16.MAX):
                    raise Error("Schulze paths exceed the selected arithmetic type")
    return requested_type


def strongest_paths_typed[
    score_dtype: DType, seed: SeedGraph
](preferences: PreferenceMatrix, backend: Backend) raises -> StrongestPathsMatrix:
    var paths = compute_strongest_paths_gpu[score_dtype, TILE_SIZE, seed](
        preferences
    ) if backend == Backend.gpu else compute_strongest_paths_tiled_cpu_simd[score_dtype, TILE_SIZE, seed](preferences)
    var result = StrongestPathsMatrix(preferences.num_candidates)
    for cell in range(preferences.num_candidates * preferences.num_candidates):
        result.data[unsafe_offset=cell] = paths.data[unsafe_offset=cell].cast[DType.uint32]()
    return result^


def compute_strongest_paths[
    seed: SeedGraph = SeedGraph.winning_votes
](
    preferences: PreferenceMatrix, *, backend: Backend = Backend.cpu, score_type: ScoreType = ScoreType.auto
) raises -> StrongestPathsMatrix:
    """Computes strongest paths using the selected device and arithmetic type."""
    var resolved = resolve_score_type[seed](preferences, score_type)
    if resolved == ScoreType.uint16:
        return strongest_paths_typed[DType.uint16, seed](preferences, backend)
    if resolved == ScoreType.uint32:
        return compute_strongest_paths_gpu[DType.uint32, TILE_SIZE, seed](
            preferences
        ) if backend == Backend.gpu else compute_strongest_paths_tiled_cpu_simd[DType.uint32, TILE_SIZE, seed](
            preferences
        )
    if resolved == ScoreType.uint64:
        return strongest_paths_typed[DType.uint64, seed](preferences, backend)
    raise Error("Invalid score type")


# region Results


def compute_split_cycle_winners(
    preferences: PreferenceMatrix, *, backend: Backend = Backend.cpu, score_type: ScoreType = ScoreType.auto
) raises -> List[Int]:
    """
    Names the candidates nobody defeats, which is the Split Cycle winning set.

    Holliday and Pacuit's Lemma 3.17: one candidate defeats another when its margin is positive
    and exceeds the widest path running back the other way. The set is irresolute by Theorem 4.7,
    so it can name several winners where Schulze names one.

    Args:
        preferences: Pairwise vote counts.

    Returns:
        The undefeated candidates, in increasing order.
    """
    var margin_paths = compute_strongest_paths[SeedGraph.positive_margins](
        preferences, backend=backend, score_type=score_type
    )
    var num_candidates = preferences.num_candidates
    var undefeated = List[Int]()

    for candidate in range(num_candidates):
        var defeated = False
        for rival in range(num_candidates):
            if rival == candidate:
                continue
            var forward = preferences[rival, candidate]
            var backward = preferences[candidate, rival]
            if forward <= backward:
                continue
            if (forward - backward) > margin_paths[candidate, rival]:
                defeated = True
                break
        if not defeated:
            undefeated.append(candidate)

    return undefeated^


@fieldwise_init
struct ElectionOutcome(Movable):
    """One sweep's verdict: who won and the order everyone else finished in."""

    var winner: Int
    """The candidate at the head of the ranking."""
    var ranking: List[Int]
    """Every candidate, most preferred first, ties broken by ascending index."""


def compute_election_results(
    strongest_paths: StrongestPathsMatrix,
) -> ElectionOutcome:
    """
    Determines the winner and ranking based on strongest paths matrix.

    Args:
        strongest_paths: Computed strongest paths matrix.

    Returns:
        The winner and the full ranking behind them.
    """
    var num_candidates = strongest_paths.num_candidates
    var wins = List[Int]()
    wins.resize(num_candidates, 0)

    for candidate in range(num_candidates):
        var win_count = 0
        for rival in range(num_candidates):
            if candidate != rival and strongest_paths[candidate, rival] > strongest_paths[rival, candidate]:
                win_count += 1
        wins[candidate] = win_count

    var scored_candidates = List[IndexedScore]()
    for candidate in range(num_candidates):
        scored_candidates.append(IndexedScore(candidate, wins[candidate]))

    sort(scored_candidates)

    var ranking = List[Int]()
    for position in range(len(scored_candidates)):
        ranking.append(scored_candidates[position].index)

    var winner = ranking[0]
    return ElectionOutcome(winner, ranking^)


# endregion Results
