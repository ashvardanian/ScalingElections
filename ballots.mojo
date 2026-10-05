"""
Ballot storage shared by the Schulze and Kemeny-Young solvers.

Entries count voters preferring the row candidate to the column candidate.
The same generic storage holds strongest path capacities after solving Schulze.
"""

from std.atomic import Atomic
from std.memory import (
    AddressSpace,
    Layout,
    alloc,
    stack_allocation,
    unsafe_memset_zero,
)
from std.random.philox import Random
from std.sys import num_logical_cores

from max.algorithm import parallelize
from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext


@fieldwise_init
struct Backend(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def name(self) -> String:
        return String("CPU") if self == Self.cpu else String("GPU")

    comptime cpu = Self(0)
    comptime gpu = Self(1)


@fieldwise_init
struct ScoreType(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    comptime auto = Self(0)
    comptime uint16 = Self(16)
    comptime uint32 = Self(32)
    comptime uint64 = Self(64)
    comptime saturated64 = Self(65)

    def width(self) -> Int:
        return 64 if self == Self.saturated64 else Int(self.value)

    @staticmethod
    def parse(value: String) raises -> Self:
        if value == "auto":
            return Self.auto
        if value == "uint16":
            return Self.uint16
        if value == "uint32":
            return Self.uint32
        if value == "uint64":
            return Self.uint64
        if value == "saturated64":
            return Self.saturated64
        raise Error("score_type must be auto, uint16, uint32, uint64, or saturated64")


@fieldwise_init
struct Arithmetic(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    comptime exact = Self(0)
    comptime saturated = Self(1)


@fieldwise_init
struct Unranked(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    comptime unknown = Self(0)
    comptime worse = Self(1)


@fieldwise_init
struct Saturated[dtype: DType](Copyable, Movable):
    """An unsigned counter whose maximum is the saturation sentinel."""

    comptime count_t = SIMD[Self.dtype, 1]

    var value: Self.count_t

    @always_inline
    def __add__(self, other: Self) -> Self:
        return Self(min(self.value, Self.count_t.MAX - other.value) + other.value)


@always_inline
def add_counts[arithmetic: Arithmetic, dtype: DType](left: SIMD[dtype, 1], right: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    comptime if arithmetic == Arithmetic.saturated:
        return (Saturated[dtype](left) + Saturated[dtype](right)).value
    else:
        return left + right


# region Matrix


@fieldwise_init
struct VoteMatrix[stored_count_dtype: DType = DType.uint32](Movable):
    """Dense square matrix of pairwise vote counts, indexed by a pair of candidates."""

    comptime stored_count_t = SIMD[Self.stored_count_dtype, 1]

    var data: Pointer[Self.stored_count_t, MutUntrackedOrigin]
    var num_candidates: Int

    def __init__(out self, num_candidates: Int) raises:
        if num_candidates < 0 or (num_candidates != 0 and num_candidates > Int.MAX // num_candidates // 8):
            raise Error("Matrix size exceeds the addressable range")
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = alloc(Layout[Self.stored_count_t](count=size)).unsafe_leak()
        unsafe_memset_zero(self.data, size)

    def __getitem__(self, row: Int, column: Int) -> Self.stored_count_t:
        return self.data[unsafe_offset=row * self.num_candidates + column]

    def __setitem__(mut self, row: Int, column: Int, value: Self.stored_count_t):
        self.data[unsafe_offset=row * self.num_candidates + column] = value

    def __deinit__(deinit self):
        self.data.unsafe_free()


comptime UInt32VoteMatrix = VoteMatrix[DType.uint32]

# endregion Matrix

# region Preferences


def populate_preferences_from_ranking(mut preferences: UInt32VoteMatrix, ranking: List[Int]):
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


def generate_random_preferences(num_candidates: Int, num_voters: Int, seed_value: Int) raises -> UInt32VoteMatrix:
    """
    Draws a preference matrix for a synthetic election of the requested shape.

    Args:
        num_candidates: Number of candidates.
        num_voters: Number of voters. If 0, draws the counts themselves at random.
        seed_value: Seeds the counter-based generator, so a run reproduces exactly.

    Returns:
        Random preference matrix.
    """
    var preferences = UInt32VoteMatrix(num_candidates)

    if num_voters == 0:

        def fill_row(row: Int) {imm}:
            # Seeded per row, so parallel workers share no state to race on.
            var generator = Random(seed=UInt64(seed_value), offset=UInt64(row))
            var bound = UInt32(350_000_001)
            var column = 0
            while column < num_candidates:
                # Every lane of the draw is spent, rather than three in four discarded.
                var draws = generator.step()
                var lanes = min(len(draws), num_candidates - column)
                for lane in range(lanes):
                    preferences.data[unsafe_offset=row * num_candidates + column + lane] = draws[
                        lane
                    ] % bound if row != column + lane else UInt32(0)
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


def winning_votes_graph[
    arithmetic_dtype: DType, stored_count_dtype: DType
](
    preferences: VoteMatrix[stored_count_dtype],
    graph: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
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
                    graph[unsafe_offset=row * row_stride + column] = forward.cast[arithmetic_dtype]()
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


def positive_margins_graph[
    arithmetic_dtype: DType, stored_count_dtype: DType
](
    preferences: VoteMatrix[stored_count_dtype],
    graph: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
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
            var margin = forward - backward if row != column and forward > backward else SIMD[stored_count_dtype, 1](0)
            graph[unsafe_offset=row * row_stride + column] = margin.cast[arithmetic_dtype]()

    parallelize(fill_row, num_candidates)


def seed_graph[
    arithmetic_dtype: DType, stored_count_dtype: DType
](
    preferences: VoteMatrix[stored_count_dtype],
    graph: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    row_stride: Int,
    which: SeedGraph,
):
    """Seeds the matrix with whichever graph the method is defined on."""
    if which == SeedGraph.positive_margins:
        positive_margins_graph(preferences, graph, row_stride)
    else:
        winning_votes_graph(preferences, graph, row_stride)


# endregion Graph


def tally_ballots_cpu(rankings: List[UInt32], num_ballots: Int, num_candidates: Int) raises -> UInt32VoteMatrix:
    """Counts complete rankings in parallel with one private matrix per worker."""
    var preferences = UInt32VoteMatrix(num_candidates)
    if num_ballots == 0:
        return preferences^
    var workers = min(num_ballots, num_logical_cores())
    var cells = num_candidates * num_candidates
    var counts = List[UInt32]()
    counts.resize(workers * cells, 0)
    var counts_ptr = counts.unsafe_ptr()

    def count_chunk(worker: Int) {imm}:
        var private_counts = counts_ptr.unsafe_offset(worker * cells)
        for ballot in range(
            num_ballots * worker // workers,
            num_ballots * (worker + 1) // workers,
        ):
            var base = ballot * num_candidates
            for position in range(num_candidates - 1):
                var preferred = Int(rankings[base + position])
                for later in range(position + 1, num_candidates):
                    var cell = preferred * num_candidates + Int(rankings[base + later])
                    private_counts[unsafe_offset=cell] += 1

    parallelize(count_chunk, workers)
    for worker in range(workers):
        for cell in range(cells):
            preferences.data[unsafe_offset=cell] += counts[worker * cells + cell]
    return preferences^


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
                _ = Atomic.fetch_add(
                    counters.unsafe_offset(preferred * num_candidates + opponent),
                    UInt32(1),
                )
        ballot += stride
    barrier()

    cell = thread
    while cell < cells:
        var counted = counters[unsafe_offset=cell]
        if counted != 0:
            _ = Atomic.fetch_add(preferences.unsafe_offset(cell), counted)
        cell += threads


def tally_ballots_gpu(rankings: List[UInt32], num_ballots: Int, num_candidates: Int) raises -> UInt32VoteMatrix:
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

    var preferences = UInt32VoteMatrix(num_candidates)
    if num_ballots == 0:
        return preferences^
    var cells = num_candidates * num_candidates
    var total = num_ballots * num_candidates

    var ctx = DeviceContext()
    var host_rankings = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var device_rankings = ctx.enqueue_create_buffer[DType.uint32](total)
    var host_counts = ctx.enqueue_create_host_buffer[DType.uint32](cells)
    var device_counts = ctx.enqueue_create_buffer[DType.uint32](cells)
    ctx.synchronize()

    for index in range(total):
        host_rankings[index] = rankings[index]
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

    var counts = host_counts.as_span()
    for cell in range(cells):
        preferences.data[unsafe_offset=cell] = counts[cell]
    return preferences^


def tally_ballots(
    rankings: List[UInt32],
    num_ballots: Int,
    num_candidates: Int,
    *,
    backend: Backend = Backend.cpu,
) raises -> UInt32VoteMatrix:
    """Counts complete rankings using the selected device's default kernel."""
    return tally_ballots_gpu(rankings, num_ballots, num_candidates) if backend == Backend.gpu else tally_ballots_cpu(
        rankings, num_ballots, num_candidates
    )


@always_inline
def tally_ragged_row[
    arithmetic_dtype: DType, arithmetic: Arithmetic
](
    rankings: Pointer[UInt32, ImmUntrackedOrigin],
    offsets: Pointer[UInt64, ImmUntrackedOrigin],
    ranks: Pointer[UInt32, ImmUntrackedOrigin],
    weights: Pointer[UInt64, ImmUntrackedOrigin],
    counts: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    candidate: Int,
    num_candidates: Int,
    first: Int,
    last: Int,
    unranked: Unranked,
):
    for ballot in range(first, last):
        var begin = Int(offsets[unsafe_offset=ballot])
        var end = Int(offsets[unsafe_offset=ballot + 1])
        var position = begin
        while position < end and Int(rankings[unsafe_offset=position]) != candidate:
            position += 1
        if position == end:
            continue
        var weight = weights[unsafe_offset=ballot].cast[arithmetic_dtype]()
        var rank = ranks[unsafe_offset=position]
        for other in range(begin, end):
            if rank < ranks[unsafe_offset=other]:
                var opponent = Int(rankings[unsafe_offset=other])
                counts[unsafe_offset=opponent] = add_counts[arithmetic](counts[unsafe_offset=opponent], weight)
        if unranked == Unranked.worse:
            for opponent in range(num_candidates):
                var other = begin
                while other < end and Int(rankings[unsafe_offset=other]) != opponent:
                    other += 1
                if other == end:
                    counts[unsafe_offset=opponent] = add_counts[arithmetic](counts[unsafe_offset=opponent], weight)


def tally_ragged_gpu[
    arithmetic_dtype: DType, arithmetic: Arithmetic
](
    rankings: Pointer[UInt32, MutUntrackedOrigin],
    offsets: Pointer[UInt64, MutUntrackedOrigin],
    ranks: Pointer[UInt32, MutUntrackedOrigin],
    weights: Pointer[UInt64, MutUntrackedOrigin],
    counts: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    num_candidates_arg: Int64,
    num_ballots_arg: Int64,
    chunks_arg: Int64,
    unranked_value: UInt8,
):
    var num_candidates = Int(num_candidates_arg)
    var num_ballots = Int(num_ballots_arg)
    var chunks = Int(chunks_arg)
    var task = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if task >= chunks * num_candidates:
        return
    var chunk = task // num_candidates
    var candidate = task % num_candidates
    tally_ragged_row[arithmetic_dtype, arithmetic](
        rankings,
        offsets,
        ranks,
        weights,
        counts.unsafe_offset(task * num_candidates),
        candidate,
        num_candidates,
        num_ballots * chunk // chunks,
        num_ballots * (chunk + 1) // chunks,
        Unranked(unranked_value),
    )


def reduce_tally_gpu[
    arithmetic_dtype: DType, arithmetic: Arithmetic
](
    counts: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    result: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    cells_arg: Int64,
    chunks_arg: Int64,
):
    var cells = Int(cells_arg)
    var chunks = Int(chunks_arg)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= cells:
        return
    var total = SIMD[arithmetic_dtype, 1](0)
    for chunk in range(chunks):
        total = add_counts[arithmetic](total, counts[unsafe_offset=chunk * cells + cell])
    result[unsafe_offset=cell] = total


def tally_ragged_typed[
    arithmetic_dtype: DType, arithmetic: Arithmetic
](
    rankings: List[UInt32],
    offsets: List[UInt64],
    ranks: List[UInt32],
    weights: List[UInt64],
    num_candidates: Int,
    unranked: Unranked,
    backend: Backend,
) raises -> VoteMatrix[DType.uint64]:
    var result = VoteMatrix[DType.uint64](num_candidates)
    var num_ballots = len(weights)
    if num_ballots == 0:
        return result^
    var cells = num_candidates * num_candidates
    if backend == Backend.cpu:
        var counts = List[SIMD[arithmetic_dtype, 1]]()
        counts.resize(cells, 0)
        var counts_ptr = counts.unsafe_ptr()

        def count_row(candidate: Int) {imm}:
            tally_ragged_row[arithmetic_dtype, arithmetic](
                rankings.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
                offsets.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
                ranks.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
                weights.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
                counts_ptr.unsafe_offset(candidate * num_candidates).unsafe_origin_cast[MutUntrackedOrigin](),
                candidate,
                num_candidates,
                0,
                num_ballots,
                unranked,
            )

        parallelize(count_row, num_candidates)
        for cell in range(cells):
            result.data[unsafe_offset=cell] = UInt64(counts[cell])
    else:
        # Private rows avoid unsupported 64-bit atomics on Metal.
        var chunks = min(num_ballots, max(1, min(4096 // num_candidates, (64 * 1024 * 1024) // (cells * 8))))
        var ctx = DeviceContext()
        var host_rankings = ctx.enqueue_create_host_buffer[DType.uint32](max(1, len(rankings)))
        var host_offsets = ctx.enqueue_create_host_buffer[DType.uint64](len(offsets))
        var host_ranks = ctx.enqueue_create_host_buffer[DType.uint32](max(1, len(ranks)))
        var host_weights = ctx.enqueue_create_host_buffer[DType.uint64](num_ballots)
        var device_rankings = ctx.enqueue_create_buffer[DType.uint32](max(1, len(rankings)))
        var device_offsets = ctx.enqueue_create_buffer[DType.uint64](len(offsets))
        var device_ranks = ctx.enqueue_create_buffer[DType.uint32](max(1, len(ranks)))
        var device_weights = ctx.enqueue_create_buffer[DType.uint64](num_ballots)
        var counts = ctx.enqueue_create_buffer[arithmetic_dtype](chunks * cells)
        var device_result = ctx.enqueue_create_buffer[arithmetic_dtype](cells)
        var host_result = ctx.enqueue_create_host_buffer[arithmetic_dtype](cells)
        ctx.synchronize()
        for entry in range(len(rankings)):
            host_rankings[entry] = rankings[entry]
            host_ranks[entry] = ranks[entry]
        for ballot in range(num_ballots):
            host_weights[ballot] = weights[ballot]
        for offset in range(len(offsets)):
            host_offsets[offset] = offsets[offset]
        host_rankings.enqueue_copy_to(device_rankings)
        host_offsets.enqueue_copy_to(device_offsets)
        host_ranks.enqueue_copy_to(device_ranks)
        host_weights.enqueue_copy_to(device_weights)
        counts.enqueue_fill(0)
        ctx.enqueue_function[tally_ragged_gpu[arithmetic_dtype, arithmetic]](
            device_rankings.unsafe_ptr(),
            device_offsets.unsafe_ptr(),
            device_ranks.unsafe_ptr(),
            device_weights.unsafe_ptr(),
            counts.unsafe_ptr(),
            Int64(num_candidates),
            Int64(num_ballots),
            Int64(chunks),
            unranked.value,
            grid_dim=((chunks * num_candidates + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[reduce_tally_gpu[arithmetic_dtype, arithmetic]](
            counts.unsafe_ptr(),
            device_result.unsafe_ptr(),
            Int64(cells),
            Int64(chunks),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        device_result.enqueue_copy_to(host_result)
        ctx.synchronize()
        for cell in range(cells):
            result.data[unsafe_offset=cell] = UInt64(host_result[cell])
    return result^


def resolve_tally_score_type(bound: UInt64, requested: ScoreType) raises -> ScoreType:
    if requested == ScoreType.auto:
        if bound <= UInt64(UInt32.MAX):
            return ScoreType.uint32
        return ScoreType.saturated64 if bound == UInt64.MAX else ScoreType.uint64
    if (
        (requested == ScoreType.uint16 and bound > UInt64(UInt16.MAX))
        or (requested == ScoreType.uint32 and bound > UInt64(UInt32.MAX))
        or (requested == ScoreType.uint64 and bound == UInt64.MAX)
    ):
        raise Error("Tally bound exceeds the selected arithmetic type")
    return requested


def tally_ragged_ballots(
    rankings: List[UInt32],
    offsets: List[UInt64],
    ranks: List[UInt32],
    weights: List[UInt64],
    num_candidates: Int,
    *,
    unranked: Unranked = Unranked.unknown,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) raises -> VoteMatrix[DType.uint64]:
    """Tallies weighted CSR ballots; UInt64.MAX cells report unrepresentable counts."""
    if num_candidates < 1 or num_candidates > Int.MAX // num_candidates // 8:
        raise Error("Invalid candidate count or matrix size")
    if len(offsets) != len(weights) + 1 or len(offsets) == 0:
        raise Error("Offsets must have one more entry than weights")
    if offsets[0] != 0 or offsets[len(offsets) - 1] != UInt64(len(rankings)) or len(ranks) != len(rankings):
        raise Error("Offsets and ranks must cover all entries")
    var seen = List[Int]()
    seen.resize(num_candidates, -1)
    var bound = UInt64(0)
    for ballot in range(len(weights)):
        if offsets[ballot] > offsets[ballot + 1] or offsets[ballot + 1] > UInt64(len(rankings)):
            raise Error("Offsets must be monotone and within the rankings")
        bound = add_counts[Arithmetic.saturated](bound, weights[ballot])
        for position in range(Int(offsets[ballot]), Int(offsets[ballot + 1])):
            var candidate = Int(rankings[position])
            if candidate >= num_candidates or seen[candidate] == ballot:
                raise Error("Candidate IDs must be in range and unique within each ballot")
            seen[candidate] = ballot
    var resolved = resolve_tally_score_type(bound, score_type)
    if resolved == ScoreType.uint16:
        return tally_ragged_typed[DType.uint16, Arithmetic.exact](
            rankings, offsets, ranks, weights, num_candidates, unranked, backend
        )
    if resolved == ScoreType.uint32:
        return tally_ragged_typed[DType.uint32, Arithmetic.exact](
            rankings, offsets, ranks, weights, num_candidates, unranked, backend
        )
    if resolved == ScoreType.saturated64 or bound == UInt64.MAX:
        return tally_ragged_typed[DType.uint64, Arithmetic.saturated](
            rankings, offsets, ranks, weights, num_candidates, unranked, backend
        )
    return tally_ragged_typed[DType.uint64, Arithmetic.exact](
        rankings, offsets, ranks, weights, num_candidates, unranked, backend
    )
