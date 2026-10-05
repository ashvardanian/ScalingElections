"""
Ballot storage shared by the Schulze and Kemeny-Young solvers.

Both stages pass around one square `UInt32` matrix. Read as preferences, entry `(row, column)`
counts the voters ranking the row's candidate above the column's; read as strongest paths, it
holds the widest bottleneck between them. The two meanings share a layout, an allocation and an
accessor, so `VoteMatrix` is the storage and the aliases below name the reading.

`winning_votes_graph` turns the first into the seed of the second, keeping only the winning side
of each pairwise contest, which is where every Schulze backend starts.
"""

from std.atomic import Atomic
from std.memory import AddressSpace, Layout, alloc, stack_allocation, unsafe_memset_zero
from std.random.philox import Random

from max.algorithm import parallelize
from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext

# region Matrix


@fieldwise_init
struct VoteMatrix(Movable):
    """Dense square matrix of pairwise vote counts, indexed by a pair of candidates."""

    var data: Pointer[UInt32, MutUntrackedOrigin]
    var num_candidates: Int

    def __init__(out self, num_candidates: Int):
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = alloc(Layout[UInt32](count=size)).unsafe_leak()
        unsafe_memset_zero(self.data, size)

    def __getitem__(self, row: Int, column: Int) -> UInt32:
        return self.data[unsafe_offset=row * self.num_candidates + column]

    def __setitem__(mut self, row: Int, column: Int, value: UInt32):
        self.data[unsafe_offset=row * self.num_candidates + column] = value

    def __deinit__(deinit self):
        self.data.unsafe_free()


comptime PreferenceMatrix = VoteMatrix
"""The two names read differently at a call site and denote the same storage."""
comptime StrongestPathsMatrix = VoteMatrix

# endregion Matrix

# region Preferences


def populate_preferences_from_ranking(mut preferences: PreferenceMatrix, ranking: List[Int]):
    """
    Populates the preference matrix based on a ranking of candidates.

    Args:
        preferences: The preference matrix to populate.
        ranking: List of candidate indices in order of preference.
    """
    var num_ranked = len(ranking)
    for position in range(num_ranked):
        var preferred = ranking[position]
        for later in range(position + 1, num_ranked):
            var opponent = ranking[later]
            var current_count = preferences[preferred, opponent]
            preferences[preferred, opponent] = current_count + 1


def generate_random_preferences(num_candidates: Int, num_voters: Int, seed_value: Int) -> PreferenceMatrix:
    """
    Draws a preference matrix for a synthetic election of the requested shape.

    Args:
        num_candidates: Number of candidates.
        num_voters: Number of voters. If 0, draws the counts themselves at random.
        seed_value: Seeds the counter-based generator, so a run reproduces exactly.

    Returns:
        Random preference matrix.
    """
    var preferences = PreferenceMatrix(num_candidates)

    if num_voters == 0:

        def fill_row(row: Int) {imm}:
            # Seeded per row, so parallel workers share no state to race on.
            var generator = Random(seed=UInt64(seed_value), offset=UInt64(row))
            var bound = UInt32(num_candidates)
            var column = 0
            while column < num_candidates:
                # Every lane of the draw is spent, rather than three in four discarded.
                var draws = generator.step()
                var lanes = min(len(draws), num_candidates - column)
                for lane in range(lanes):
                    preferences.data[unsafe_offset=row * num_candidates + column + lane] = draws[lane] % bound
                column += lanes

        parallelize(fill_row, num_candidates)
        return preferences^

    var ranking = List[Int]()
    ranking.resize(num_candidates, 0)
    var generator = Random(seed=UInt64(seed_value))

    for _ in range(num_voters):
        for candidate in range(num_candidates):
            ranking[candidate] = candidate

        # Fisher-Yates, drawing an index in `[0, upper]` inclusive.
        for upper in range(num_candidates - 1, 0, -1):
            var draws = generator.step()
            var chosen = Int(draws[0] % UInt32(upper + 1))
            var held = ranking[upper]
            ranking[upper] = ranking[chosen]
            ranking[chosen] = held

        populate_preferences_from_ranking(preferences, ranking)

    return preferences^


# endregion Preferences

# region Graph


def winning_votes_graph(
    preferences: PreferenceMatrix,
    graph: Pointer[UInt32, MutUntrackedOrigin],
    row_stride: Int,
):
    """
    Seeds a strongest-paths graph with the winning side of each pairwise contest.

    Entry `(row, column)` keeps the winner's votes when the row's candidate took the pair, which
    is the direct-comparison step every Schulze backend runs before its Floyd-Warshall sweep.
    Only the leading `num_candidates` columns of each row are written, so a padded destination
    keeps whatever its tail already held.

    Args:
        preferences: Input preference matrix.
        graph: Destination graph, at least `num_candidates` rows of `row_stride` entries.
        row_stride: Distance in entries between consecutive rows of the destination.
    """
    var num_candidates = preferences.num_candidates

    def fill_row(row: Int) {imm}:
        for column in range(num_candidates):
            if row != column:
                var forward = preferences[row, column]
                var backward = preferences[column, row]
                if forward > backward:
                    graph[unsafe_offset=row * row_stride + column] = forward
                else:
                    graph[unsafe_offset=row * row_stride + column] = 0

    parallelize(fill_row, num_candidates)


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


def positive_margins_graph(
    preferences: PreferenceMatrix,
    graph: Pointer[UInt32, MutUntrackedOrigin],
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
            var margin = forward - backward if row != column and forward > backward else UInt32(0)
            graph[unsafe_offset=row * row_stride + column] = margin

    parallelize(fill_row, num_candidates)


def seed_graph(
    preferences: PreferenceMatrix,
    graph: Pointer[UInt32, MutUntrackedOrigin],
    row_stride: Int,
    which: SeedGraph,
):
    """Seeds the matrix with whichever graph the method is defined on."""
    if which == SeedGraph.positive_margins:
        positive_margins_graph(preferences, graph, row_stride)
    else:
        winning_votes_graph(preferences, graph, row_stride)


# endregion Graph


comptime TALLY_MAX_CANDIDATES = 64
"""The widest field the shared counter matrix holds, at four bytes a cell."""

comptime TALLY_BLOCK_SIZE = 256
"""Threads per block for the tally, one ballot to a thread."""


def gpu_tally_kernel[
    max_candidates: Int
](
    rankings: Pointer[UInt32, MutUntrackedOrigin],
    num_ballots_arg: Int32,
    num_candidates_arg: Int32,
    preferences: Pointer[UInt32, MutUntrackedOrigin],
):
    """Accumulates each block's ballots into a shared matrix, merging into global once at exit."""
    var num_ballots = Int(num_ballots_arg)
    var num_candidates = Int(num_candidates_arg)
    var cells = num_candidates * num_candidates
    var thread = Int(thread_idx.x)
    var threads = Int(block_dim.x)

    var counters = stack_allocation[
        max_candidates * max_candidates,
        UInt32,
        address_space=AddressSpace.SHARED,
    ]()
    var cell = thread
    while cell < cells:
        counters[unsafe_offset=cell] = 0
        cell += threads
    barrier()

    var stride = Int(grid_dim.x) * threads
    var ballot = Int(block_idx.x) * threads + thread
    while ballot < num_ballots:
        var base = ballot * num_candidates
        for position in range(num_candidates - 1):
            var preferred = Int(rankings[unsafe_offset=base + position])
            for later in range(position + 1, num_candidates):
                var opponent = Int(rankings[unsafe_offset=base + later])
                _ = Atomic.fetch_add(counters.unsafe_offset(preferred * num_candidates + opponent), UInt32(1))
        ballot += stride
    barrier()

    cell = thread
    while cell < cells:
        var counted = counters[unsafe_offset=cell]
        if counted != 0:
            _ = Atomic.fetch_add(preferences.unsafe_offset(cell), counted)
        cell += threads


def tally_ballots_gpu(rankings: List[UInt32], num_ballots: Int, num_candidates: Int) raises -> PreferenceMatrix:
    """
    Counts complete rankings into a pairwise matrix, one private matrix per block.

    Args:
        rankings: Row-major complete rankings, `num_ballots` by `num_candidates`, best first.
        num_ballots: How many rankings the chunk holds.
        num_candidates: The number of candidates, at most `TALLY_MAX_CANDIDATES`.

    Returns:
        The square matrix counting, per ordered pair, the ballots preferring the first.
    """
    if num_candidates > TALLY_MAX_CANDIDATES:
        raise Error("The GPU tally holds at most " + String(TALLY_MAX_CANDIDATES) + " candidates")

    var preferences = PreferenceMatrix(num_candidates)
    var cells = num_candidates * num_candidates
    var total = num_ballots * num_candidates

    var ctx = DeviceContext()
    var host_rankings = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var device_rankings = ctx.enqueue_create_buffer[DType.uint32](total)
    var host_counts = ctx.enqueue_create_host_buffer[DType.uint32](cells)
    var device_counts = ctx.enqueue_create_buffer[DType.uint32](cells)
    ctx.synchronize()

    var rankings_ptr = host_rankings.unsafe_ptr()
    for index in range(total):
        rankings_ptr[unsafe_offset=index] = rankings[index]
    unsafe_memset_zero(host_counts.unsafe_ptr(), cells)

    host_rankings.enqueue_copy_to(device_rankings)
    host_counts.enqueue_copy_to(device_counts)

    var blocks = min((num_ballots + TALLY_BLOCK_SIZE - 1) // TALLY_BLOCK_SIZE, 65535)
    ctx.enqueue_function[gpu_tally_kernel[TALLY_MAX_CANDIDATES]](
        device_rankings.unsafe_ptr(),
        Int32(num_ballots),
        Int32(num_candidates),
        device_counts.unsafe_ptr(),
        grid_dim=(blocks, 1, 1),
        block_dim=(TALLY_BLOCK_SIZE, 1, 1),
    )

    device_counts.enqueue_copy_to(host_counts)
    ctx.synchronize()

    var counts_ptr = host_counts.unsafe_ptr()
    for cell in range(cells):
        preferences.data[unsafe_offset=cell] = counts_ptr[unsafe_offset=cell]
    deinit(host_counts^)
    return preferences^
