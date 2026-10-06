"""
Schulze strongest paths and Split Cycle winners, as a three-phase tiled sweep on CPU and GPU.

The Schulze method is a Condorcet system: the winner is whoever beats every rival along the
widest chain of pairwise wins, which is a max-min closure over the winning-votes graph and so a
Floyd-Warshall in the `(max, min)` semiring rather than `(min, +)`.

Both backends block the closure into three dependency phases per diagonal tile, so the working
set of one step fits in cache or in shared memory. Split Cycle runs the same closure over
positive margins instead of winning votes.
"""

from std.builtin.sort import sort
from std.math import iota
from std.memory import (
    AddressSpace,
    stack_allocation,
    unsafe_memcpy,
    unsafe_memset_zero,
)

from max.algorithm import parallelize
from max.gpu import barrier, block_idx, thread_idx
from max.gpu.host import DeviceContext

from ballots import (
    Arithmetic,
    Backend,
    CandidateIndex,
    ScoreType,
    VoteMatrix,
    VoteMatrixView,
    VoterWeight,
    divide_round_up,
    with_score_type,
)


# region Types

comptime TILE_SIZE = 32
"""The tile edge every backend compiles for: a tile of `TILE_SIZE` squared threads fills the 1024-thread block limit.

A field narrower than a tile is zero-padded.
"""


@fieldwise_init
struct TilePhase(Copyable, Equatable, Movable):
    """Which tile phase a processor runs, fixing aliasing and diagonal handling together.

    The three states are the only combinations the recurrence produces.
    """

    var value: UInt8
    comptime aliased = Self(0)
    """Output and inputs share one tile, so every step needs a barrier."""
    comptime distinct_diagonal = Self(1)
    """Separate tiles, but the output may straddle the matrix diagonal."""
    comptime distinct_independent = Self(2)
    """Separate tiles, provably off the diagonal, so the update is a plain maximum."""

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


@fieldwise_init
struct SeedGraph(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Which graph the strongest-paths sweep closes over."""

    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    comptime winning_votes = Self(0)
    """Winning votes, the variant Schulze runs on here."""
    comptime positive_margins = Self(1)
    """Positive margins, which is what Split Cycle is defined on."""


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


# region Graph


def winning_votes_graph[
    ArithmeticDataType: DType, StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    row_stride: Int,
):
    """
    Seeds a strongest-paths graph with the winning side of each pairwise contest.

    Entry `(row, column)` keeps the winner's votes when the row's candidate took the pair, and
    ties, losses and the diagonal read as zero, the identity of the max-min semiring. Only the
    leading `num_candidates` columns of each row are written, so a padded tail keeps its contents.

    Args:
        preferences: Input preference matrix.
        graph: Destination graph, at least `num_candidates` rows of `row_stride` entries.
        row_stride: Distance in entries between consecutive rows of the destination.
    """
    var num_candidates = preferences.num_candidates

    def fill_row(row: Int) {imm}:
        for column in range(num_candidates):
            var forward = preferences[row, column]
            var won = row != column and forward > preferences[column, row]
            graph[unsafe_offset=row * row_stride + column] = forward.cast[ArithmeticDataType]() if won else 0

    parallelize(fill_row, num_candidates)


def positive_margins_graph[
    ArithmeticDataType: DType, StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    row_stride: Int,
):
    """
    Seeds a strongest-paths graph with each pair's positive margin.

    Args:
        preferences: Pairwise vote counts.
        graph: Destination, which may be padded wider than the electorate.
        row_stride: The destination's row stride.
    """
    var num_candidates = preferences.num_candidates

    def fill_row(row: Int) {imm}:
        for column in range(num_candidates):
            var forward = preferences[row, column]
            var backward = preferences[column, row]
            var margin = forward - backward if row != column and forward > backward else SIMD[StoredCountDataType, 1](0)
            graph[unsafe_offset=row * row_stride + column] = margin.cast[ArithmeticDataType]()

    parallelize(fill_row, num_candidates)


def seed_graph[
    SeedMode: SeedGraph, ArithmeticDataType: DType, StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    row_stride: Int,
):
    """Seeds the matrix with whichever graph the method is defined on."""
    comptime if SeedMode == SeedGraph.positive_margins:
        positive_margins_graph(preferences, graph, row_stride)
    else:
        winning_votes_graph(preferences, graph, row_stride)


# endregion Graph


# region CPU


def process_tile_cpu[
    ArithmeticDataType: DType, TileSize: Int, VectorWidth: Int, Phase: TilePhase
](
    paths: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    to_pivot: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    from_pivot: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    paths_row: Int,
    paths_column: Int,
    pivot_origin: Int,
):
    """
    Relaxes the `paths` tile through every pivot of the tile pair, `VectorWidth` columns at a time.

    Tiles that may straddle the diagonal mask their lanes, leaving the three diagonals at the
    semiring identity; `paths_row`, `paths_column` and `pivot_origin` place them in the graph.
    """
    for step in range(TileSize):
        var pivot = pivot_origin + step
        for row in range(TileSize):
            var to_pivot_lanes = SIMD[ArithmeticDataType, VectorWidth](to_pivot[unsafe_offset=row * TileSize + step])
            for column in range(0, TileSize, VectorWidth):
                var paths_offset = row * TileSize + column
                var paths_lanes = paths.unsafe_load[width=VectorWidth](paths_offset)
                var from_pivot_lanes = from_pivot.unsafe_load[width=VectorWidth](step * TileSize + column)
                var smallest = min(to_pivot_lanes, from_pivot_lanes)
                comptime if Phase == TilePhase.distinct_independent:
                    paths.unsafe_store[width=VectorWidth](paths_offset, max(paths_lanes, smallest))
                else:
                    var global_row = paths_row + row
                    var global_column = iota[DType.int32, VectorWidth]() + Int32(paths_column + column)
                    var mask = (
                        global_column.ne(Int32(global_row))
                        & global_column.ne(Int32(pivot))
                        & smallest.gt(paths_lanes)
                        & SIMD[DType.bool, VectorWidth](fill=global_row != pivot)
                    )
                    paths.unsafe_store[width=VectorWidth](paths_offset, mask.select(smallest, paths_lanes))


def load_tile[
    ArithmeticDataType: DType, TileSize: Int
](
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    tile: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    first_row: Int,
    first_column: Int,
    num_candidates: Int,
):
    """Stages one tile of the dense graph, zero-filling whatever lies past its edge."""
    for row in range(TileSize):
        for column in range(TileSize):
            var graph_row = first_row + row
            var graph_column = first_column + column
            var inside = graph_row < num_candidates and graph_column < num_candidates
            tile[unsafe_offset=row * TileSize + column] = graph[
                unsafe_offset=graph_row * num_candidates + graph_column
            ] if inside else 0


def store_tile[
    ArithmeticDataType: DType, TileSize: Int
](
    tile: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    first_row: Int,
    first_column: Int,
    num_candidates: Int,
):
    """Writes a staged tile back over the dense graph, dropping whatever lies past its edge."""
    for row in range(min(TileSize, num_candidates - first_row)):
        for column in range(min(TileSize, num_candidates - first_column)):
            graph[unsafe_offset=(first_row + row) * num_candidates + first_column + column] = tile[
                unsafe_offset=row * TileSize + column
            ]


def compute_strongest_paths_tiled_cpu[
    StoredCountDataType: DType,
    //,
    ArithmeticDataType: DType,
    TileSize: Int,
    SeedMode: SeedGraph,
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    paths: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
):
    """
    Closes the seeded graph into strongest paths across the CPU cores, one tile per task.

    Parameters:
        TileSize: The tile edge, which the vector width divides.
        SeedMode: Which graph the closure runs over, since Split Cycle wants margins.

    Args:
        preferences: Input preference matrix.
        paths: The dense `num_candidates` square output, overwritten.
    """
    # Largest power-of-two SIMD width dividing the tile, capped at 16 for AVX-512.
    comptime vector_width = (
        16 if TileSize % 16
        == 0 else 8 if TileSize % 8
        == 0 else 4 if TileSize % 4
        == 0 else 2 if TileSize % 2
        == 0 else 1
    )
    comptime Cell = SIMD[ArithmeticDataType, 1]
    var num_candidates = preferences.num_candidates
    seed_graph[SeedMode](preferences, paths, num_candidates)
    var num_tiles = divide_round_up(num_candidates, TileSize)

    for pivot_tile in range(num_tiles):
        # Copied because a `parallelize` closure capturing the induction variable faults at -O1.
        var pivot_index = pivot_tile
        var pivot_start = pivot_tile * TileSize

        var diagonal = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
        load_tile[ArithmeticDataType, TileSize](paths, diagonal, pivot_start, pivot_start, num_candidates)
        process_tile_cpu[ArithmeticDataType, TileSize, vector_width, TilePhase.aliased](
            diagonal, diagonal, diagonal, pivot_start, pivot_start, pivot_start
        )
        store_tile[ArithmeticDataType, TileSize](diagonal, paths, pivot_start, pivot_start, num_candidates)

        def process_partial_tiles(tile_index: Int) {imm}:
            if tile_index == pivot_index:
                return
            var tile_start = tile_index * TileSize
            var pivot = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
            var row_tile = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
            var column_tile = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
            load_tile[ArithmeticDataType, TileSize](paths, pivot, pivot_start, pivot_start, num_candidates)
            load_tile[ArithmeticDataType, TileSize](paths, row_tile, tile_start, pivot_start, num_candidates)
            load_tile[ArithmeticDataType, TileSize](paths, column_tile, pivot_start, tile_start, num_candidates)
            process_tile_cpu[ArithmeticDataType, TileSize, vector_width, TilePhase.aliased](
                row_tile, row_tile, pivot, tile_start, pivot_start, pivot_start
            )
            process_tile_cpu[ArithmeticDataType, TileSize, vector_width, TilePhase.aliased](
                column_tile, pivot, column_tile, pivot_start, tile_start, pivot_start
            )
            store_tile[ArithmeticDataType, TileSize](row_tile, paths, tile_start, pivot_start, num_candidates)
            store_tile[ArithmeticDataType, TileSize](column_tile, paths, pivot_start, tile_start, num_candidates)

        parallelize(process_partial_tiles, num_tiles)

        def process_independent_tiles(flat_index: Int) {imm}:
            var tile_row = flat_index // num_tiles
            var tile_column = flat_index % num_tiles
            if tile_row == pivot_index or tile_column == pivot_index:
                return
            var row_start = tile_row * TileSize
            var column_start = tile_column * TileSize
            var tile = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
            var to_pivot = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
            var from_pivot = stack_allocation[TileSize * TileSize, Cell, alignment=64]()
            load_tile[ArithmeticDataType, TileSize](paths, tile, row_start, column_start, num_candidates)
            load_tile[ArithmeticDataType, TileSize](paths, to_pivot, row_start, pivot_start, num_candidates)
            load_tile[ArithmeticDataType, TileSize](paths, from_pivot, pivot_start, column_start, num_candidates)
            if tile_row == tile_column:
                process_tile_cpu[ArithmeticDataType, TileSize, vector_width, TilePhase.distinct_diagonal](
                    tile, to_pivot, from_pivot, row_start, column_start, pivot_start
                )
            else:
                process_tile_cpu[ArithmeticDataType, TileSize, vector_width, TilePhase.distinct_independent](
                    tile, to_pivot, from_pivot, row_start, column_start, pivot_start
                )
            store_tile[ArithmeticDataType, TileSize](tile, paths, row_start, column_start, num_candidates)

        parallelize(process_independent_tiles, num_tiles * num_tiles)


# endregion CPU


# region GPU


comptime SharedTile[ArithmeticDataType: DType] = Pointer[
    SIMD[ArithmeticDataType, 1], MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@always_inline
def shared_tile[ArithmeticDataType: DType, TileSize: Int]() -> SharedTile[ArithmeticDataType]:
    return stack_allocation[TileSize * TileSize, SIMD[ArithmeticDataType, 1], address_space=AddressSpace.SHARED]()


@always_inline
def process_tile_gpu[
    ArithmeticDataType: DType, TileSize: Int, Phase: TilePhase
](
    paths: SharedTile[ArithmeticDataType],
    to_pivot: SharedTile[ArithmeticDataType],
    from_pivot: SharedTile[ArithmeticDataType],
    paths_row: Int,
    paths_column: Int,
    pivot_origin: Int,
):
    """Relaxes the calling thread's cell of the `paths` tile through every pivot of the tile pair."""
    var row = Int(thread_idx.y)
    var column = Int(thread_idx.x)
    var paths_offset = row * TileSize + column
    var paths_cell = paths[unsafe_offset=paths_offset]
    var global_row = paths_row + row
    var global_column = paths_column + column

    for step in range(TileSize):
        var smallest = min(
            to_pivot[unsafe_offset=row * TileSize + step], from_pivot[unsafe_offset=step * TileSize + column]
        )
        comptime if Phase == TilePhase.distinct_independent:
            paths_cell = max(paths_cell, smallest)
        else:
            var pivot = pivot_origin + step
            var replace = (
                (global_row != global_column)
                & (global_row != pivot)
                & (pivot != global_column)
                & (smallest > paths_cell)
            )
            paths_cell = smallest if replace else paths_cell
        comptime if Phase == TilePhase.aliased:
            # Every thread reads this cell back as an input of the next step.
            paths[unsafe_offset=paths_offset] = paths_cell
            barrier()

    comptime if Phase != TilePhase.aliased:
        paths[unsafe_offset=paths_offset] = paths_cell


def schulze_diagonal_kernel[
    ArithmeticDataType: DType, TileSize: Int
](
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    padded_edge: CandidateIndex,
    pivot_tile: CandidateIndex,
):
    """Closes the pivot tile over its own candidates."""
    var stride = Int(padded_edge)
    var pivot_start = Int(pivot_tile) * TileSize
    var row = Int(thread_idx.y)
    var column = Int(thread_idx.x)
    var cell = (pivot_start + row) * stride + pivot_start + column

    var paths = shared_tile[ArithmeticDataType, TileSize]()
    paths[unsafe_offset=row * TileSize + column] = graph[unsafe_offset=cell]
    barrier()
    process_tile_gpu[ArithmeticDataType, TileSize, TilePhase.aliased](
        paths, paths, paths, pivot_start, pivot_start, pivot_start
    )
    graph[unsafe_offset=cell] = paths[unsafe_offset=row * TileSize + column]


def schulze_partial_kernel[
    ArithmeticDataType: DType, TileSize: Int
](
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    padded_edge: CandidateIndex,
    pivot_tile: CandidateIndex,
):
    """Relaxes one tile of the pivot column, then one of the pivot row, through the pivot tile."""
    var stride = Int(padded_edge)
    var pivot_start = Int(pivot_tile) * TileSize
    var tile_start = Int(block_idx.x) * TileSize
    var row = Int(thread_idx.y)
    var column = Int(thread_idx.x)
    if tile_start == pivot_start:
        return

    var pivot = shared_tile[ArithmeticDataType, TileSize]()
    var paths = shared_tile[ArithmeticDataType, TileSize]()
    var offset = row * TileSize + column
    var column_cell = (tile_start + row) * stride + pivot_start + column
    var row_cell = (pivot_start + row) * stride + tile_start + column

    pivot[unsafe_offset=offset] = graph[unsafe_offset=(pivot_start + row) * stride + pivot_start + column]
    paths[unsafe_offset=offset] = graph[unsafe_offset=column_cell]
    barrier()
    process_tile_gpu[ArithmeticDataType, TileSize, TilePhase.aliased](
        paths, paths, pivot, tile_start, pivot_start, pivot_start
    )
    barrier()
    graph[unsafe_offset=column_cell] = paths[unsafe_offset=offset]

    paths[unsafe_offset=offset] = graph[unsafe_offset=row_cell]
    barrier()
    process_tile_gpu[ArithmeticDataType, TileSize, TilePhase.aliased](
        paths, pivot, paths, pivot_start, tile_start, pivot_start
    )
    graph[unsafe_offset=row_cell] = paths[unsafe_offset=offset]


def schulze_independent_kernel[
    ArithmeticDataType: DType, TileSize: Int
](
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    padded_edge: CandidateIndex,
    pivot_tile: CandidateIndex,
):
    """Relaxes every tile off the pivot's row and column through the two pivot tiles beside it."""
    var stride = Int(padded_edge)
    var pivot_start = Int(pivot_tile) * TileSize
    var row_start = Int(block_idx.y) * TileSize
    var column_start = Int(block_idx.x) * TileSize
    var row = Int(thread_idx.y)
    var column = Int(thread_idx.x)
    if row_start == pivot_start and column_start == pivot_start:
        return

    var paths = shared_tile[ArithmeticDataType, TileSize]()
    var to_pivot = shared_tile[ArithmeticDataType, TileSize]()
    var from_pivot = shared_tile[ArithmeticDataType, TileSize]()
    var offset = row * TileSize + column
    var cell = (row_start + row) * stride + column_start + column

    paths[unsafe_offset=offset] = graph[unsafe_offset=cell]
    to_pivot[unsafe_offset=offset] = graph[unsafe_offset=(row_start + row) * stride + pivot_start + column]
    from_pivot[unsafe_offset=offset] = graph[unsafe_offset=(pivot_start + row) * stride + column_start + column]
    barrier()
    if row_start == column_start:
        process_tile_gpu[ArithmeticDataType, TileSize, TilePhase.distinct_diagonal](
            paths, to_pivot, from_pivot, row_start, column_start, pivot_start
        )
    else:
        process_tile_gpu[ArithmeticDataType, TileSize, TilePhase.distinct_independent](
            paths, to_pivot, from_pivot, row_start, column_start, pivot_start
        )
    graph[unsafe_offset=cell] = paths[unsafe_offset=offset]


def compute_strongest_paths_gpu[
    StoredCountDataType: DType,
    //,
    ArithmeticDataType: DType,
    TileSize: Int,
    SeedMode: SeedGraph,
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    paths: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
) raises:
    """
    Closes the seeded graph into strongest paths on the GPU, three kernels per pivot tile.

    Parameters:
        TileSize: The tile edge, which is also the block edge.
        SeedMode: Which graph the closure runs over, since Split Cycle wants margins.

    Args:
        preferences: Input preference matrix.
        paths: The dense `num_candidates` square output, overwritten.
    """
    var num_candidates = preferences.num_candidates

    # Rounding up to a whole number of tiles keeps the kernels free of tail checks: the
    # padding is zero, which is the identity of the max-min semiring.
    var num_tiles = divide_round_up(num_candidates, TileSize)
    var padded = num_tiles * TileSize

    var ctx = DeviceContext()
    var host_graph = ctx.enqueue_create_host_buffer[ArithmeticDataType](padded * padded)
    var device_graph = ctx.enqueue_create_buffer[ArithmeticDataType](padded * padded)
    ctx.synchronize()
    var host_ptr = host_graph.unsafe_ptr()
    unsafe_memset_zero(host_ptr, padded * padded)
    seed_graph[SeedMode](preferences, host_ptr, padded)
    host_graph.enqueue_copy_to(device_graph)

    var graph_ptr = device_graph.unsafe_ptr()
    var tile_shape = (TileSize, TileSize, 1)
    for pivot_tile in range(num_tiles):
        ctx.enqueue_function[schulze_diagonal_kernel[ArithmeticDataType, TileSize]](
            graph_ptr,
            CandidateIndex(padded),
            CandidateIndex(pivot_tile),
            grid_dim=(1, 1, 1),
            block_dim=tile_shape,
        )
        ctx.enqueue_function[schulze_partial_kernel[ArithmeticDataType, TileSize]](
            graph_ptr,
            CandidateIndex(padded),
            CandidateIndex(pivot_tile),
            grid_dim=(num_tiles, 1, 1),
            block_dim=tile_shape,
        )
        ctx.enqueue_function[schulze_independent_kernel[ArithmeticDataType, TileSize]](
            graph_ptr,
            CandidateIndex(padded),
            CandidateIndex(pivot_tile),
            grid_dim=(num_tiles, num_tiles, 1),
            block_dim=tile_shape,
        )

    device_graph.enqueue_copy_to(host_graph)
    ctx.synchronize()
    for row in range(num_candidates):
        unsafe_memcpy(
            dest=paths.unsafe_offset(row * num_candidates),
            src=host_ptr.unsafe_offset(row * padded),
            count=num_candidates,
        )
    # The untracked pointer does not keep its host allocation alive.
    deinit(host_graph^)


# endregion GPU


# region Dispatch


def resolve_score_type[
    StoredCountDataType: DType, //, SeedMode: SeedGraph
](preferences: VoteMatrixView[StoredCountDataType, _], requested_type: ScoreType = ScoreType.auto) raises -> ScoreType:
    """Resolves the path type without narrowing any input count; max-min never grows an edge."""
    var largest = VoterWeight(0)
    for row in range(preferences.num_candidates):
        for column in range(preferences.num_candidates):
            var edge = VoterWeight(preferences[row, column])
            if requested_type == ScoreType.saturated64 and edge == VoterWeight.MAX:
                raise Error("Schulze input reaches the overflow sentinel")
            if row == column:
                continue
            comptime if SeedMode == SeedGraph.positive_margins:
                var reverse = VoterWeight(preferences[column, row])
                edge = edge - reverse if edge > reverse else VoterWeight(0)
            largest = max(largest, edge)
    if requested_type == ScoreType.auto:
        return ScoreType.uint32 if largest <= VoterWeight(UInt32.MAX) else ScoreType.uint64
    if (requested_type == ScoreType.uint32 and largest > VoterWeight(UInt32.MAX)) or (
        requested_type == ScoreType.uint16 and largest > VoterWeight(UInt16.MAX)
    ):
        raise Error("Schulze edge exceeds the selected arithmetic type")
    return requested_type


def strongest_paths_typed[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, SeedMode: SeedGraph
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    backend: Backend,
    paths: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
) raises:
    """Computes paths into `paths` in the selected arithmetic width, on the selected device."""
    if backend == Backend.gpu:
        return compute_strongest_paths_gpu[ArithmeticDataType, TILE_SIZE, SeedMode](preferences, paths)
    compute_strongest_paths_tiled_cpu[ArithmeticDataType, TILE_SIZE, SeedMode](preferences, paths)


def compute_strongest_paths[
    StoredCountDataType: DType, //, SeedMode: SeedGraph
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    *,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) raises -> VoteMatrix[StoredCountDataType]:
    """Computes strongest paths using the selected device and arithmetic type, stored like the input."""
    var num_candidates = preferences.num_candidates
    var result = VoteMatrix[StoredCountDataType](num_candidates)
    var stored = result.data

    def solve[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> None:
        comptime if ArithmeticDataType == StoredCountDataType:
            return strongest_paths_typed[ArithmeticDataType, SeedMode](
                preferences, backend, rebind[Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin]](stored)
            )
        else:
            var paths = VoteMatrix[ArithmeticDataType](num_candidates)
            strongest_paths_typed[ArithmeticDataType, SeedMode](preferences, backend, paths.data)
            for cell in range(num_candidates * num_candidates):
                stored[unsafe_offset=cell] = paths.data[unsafe_offset=cell].cast[StoredCountDataType]()

    with_score_type(resolve_score_type[SeedMode](preferences, score_type), solve)
    return result^


# endregion Dispatch


# region Results


def compute_split_cycle_winners[
    StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    *,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
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

    def select[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> List[Int]:
        return split_cycle_winners_typed[ArithmeticDataType](preferences, backend)

    return with_score_type(resolve_score_type[SeedGraph.positive_margins](preferences, score_type), select)


def split_cycle_winners_typed[
    StoredCountDataType: DType, //, ArithmeticDataType: DType
](preferences: VoteMatrixView[StoredCountDataType, _], backend: Backend) raises -> List[Int]:
    """Selects Split Cycle winners without converting the intermediate path matrix."""
    var num_candidates = preferences.num_candidates
    var margin_paths = VoteMatrix[ArithmeticDataType](num_candidates)
    strongest_paths_typed[ArithmeticDataType, SeedGraph.positive_margins](preferences, backend, margin_paths.data)
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
            # A path never exceeds the largest margin, so converting it to the input type is lossless.
            if forward - backward > margin_paths[candidate, rival].cast[StoredCountDataType]():
                defeated = True
                break
        if not defeated:
            undefeated.append(candidate)

    return undefeated^


@fieldwise_init
struct ElectionOutcome(Movable):
    """One sweep's verdict: who won and the order everyone else finished in."""

    var winners: List[Int]
    """Every candidate undefeated by the strongest-path relation."""
    var ranking: List[Int]
    """Every candidate, most preferred first, ties broken by ascending index."""


def compute_election_results[
    StoredCountDataType: DType
](strongest_paths: VoteMatrixView[StoredCountDataType, _],) -> ElectionOutcome:
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

    var winners = List[Int]()
    for candidate in range(num_candidates):
        var defeated = False
        for rival in range(num_candidates):
            if strongest_paths[rival, candidate] > strongest_paths[candidate, rival]:
                defeated = True
                break
        if not defeated:
            winners.append(candidate)
    return ElectionOutcome(winners^, ranking^)


# endregion Results
