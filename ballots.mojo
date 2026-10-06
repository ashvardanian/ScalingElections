"""
Ballot storage shared by the Schulze and Kemeny-Young solvers.

Entries count voters preferring the row candidate to the column candidate.
The same generic storage holds strongest path capacities after solving Schulze.
"""

from std.atomic import Atomic, Ordering
from std.collections import Span
from std.memory import (
    AddressSpace,
    Layout,
    alloc,
    stack_allocation,
    unsafe_memset_zero,
    unsafe_memcpy,
)
from std.random.philox import Random
from std.sys import num_logical_cores

from max.algorithm import parallelize
from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext


comptime CandidateIndex = UInt32
comptime RankLabel = UInt32
comptime BallotOffset = UInt64
comptime VoterWeight = UInt64
comptime PolicyCode = UInt8


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

    @staticmethod
    def parse(value: String) raises -> Self:
        if value == "unknown":
            return Self.unknown
        if value == "worse":
            return Self.worse
        raise Error("unranked must be unknown or worse")


@fieldwise_init
struct Saturated[CountDataType: DType](Copyable, Movable):
    """An unsigned counter whose maximum is the saturation sentinel."""

    comptime Count = SIMD[Self.CountDataType, 1]

    var value: Self.Count

    @always_inline
    def __add__(self, other: Self) -> Self:
        return Self(min(self.value, Self.Count.MAX - other.value) + other.value)


@always_inline
def add_counts[
    ArithmeticMode: Arithmetic, ArithmeticDataType: DType
](left: SIMD[ArithmeticDataType, 1], right: SIMD[ArithmeticDataType, 1]) -> SIMD[ArithmeticDataType, 1]:
    comptime if ArithmeticMode == Arithmetic.saturated:
        return (Saturated[ArithmeticDataType](left) + Saturated[ArithmeticDataType](right)).value
    else:
        return left + right


# region Matrix


@fieldwise_init
struct VoteMatrix[StoredCountDataType: DType = DType.uint32](Movable):
    """Dense square matrix of pairwise vote counts, indexed by a pair of candidates."""

    comptime StoredCount = SIMD[Self.StoredCountDataType, 1]

    var data: Pointer[Self.StoredCount, MutUntrackedOrigin]
    var num_candidates: Int

    def __init__(out self, num_candidates: Int) raises:
        if num_candidates < 0 or (num_candidates != 0 and num_candidates > Int.MAX // num_candidates // 8):
            raise Error("Matrix size exceeds the addressable range")
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = alloc(Layout[Self.StoredCount](count=size)).unsafe_leak()
        unsafe_memset_zero(self.data, size)

    def __getitem__(self, row: Int, column: Int) -> Self.StoredCount:
        return self.data[unsafe_offset=row * self.num_candidates + column]

    def __setitem__(mut self, row: Int, column: Int, value: Self.StoredCount):
        self.data[unsafe_offset=row * self.num_candidates + column] = value

    def view(self) -> VoteMatrixView[Self.StoredCountDataType, origin_of(self)]:
        """Borrow immutable counts while keeping this matrix alive through the solve."""
        var values = Span[Self.StoredCount, ImmUntrackedOrigin](
            unsafe_ptr=self.data,
            length=self.num_candidates * self.num_candidates,
        )
        return VoteMatrixView(
            Span(unsafe_ptr=values.unsafe_ptr().unsafe_origin_cast[origin_of(self)](), length=len(values)),
            self.num_candidates,
        )

    def __deinit__(deinit self):
        self.data.unsafe_free()


@fieldwise_init
struct VoteMatrixView[StoredCountDataType: DType, InputOrigin: ImmOrigin](Copyable, Movable):
    """Read-only contiguous counts whose owner remains alive through the solve."""

    comptime StoredCount = SIMD[Self.StoredCountDataType, 1]

    var values: Span[Self.StoredCount, Self.InputOrigin]
    """Row-major matrix entries borrowed from the owning allocation."""
    var num_candidates: Int
    """Number of rows and columns in the square matrix."""

    def __getitem__(self, row: Int, column: Int) -> Self.StoredCount:
        return self.values[row * self.num_candidates + column]


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
    ArithmeticDataType: DType, StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
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
                    graph[unsafe_offset=row * row_stride + column] = forward.cast[ArithmeticDataType]()
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
    ArithmeticDataType: DType, StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    graph: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    row_stride: Int,
    which: SeedGraph,
):
    """Seeds the matrix with whichever graph the method is defined on."""
    if which == SeedGraph.positive_margins:
        positive_margins_graph(preferences, graph, row_stride)
    else:
        winning_votes_graph(preferences, graph, row_stride)


# endregion Graph


def tally_ballots_cpu(
    candidates: Span[CandidateIndex, ImmUntrackedOrigin], num_ballots: Int, num_candidates: Int
) raises -> UInt32VoteMatrix:
    """Counts complete candidates in parallel with one private matrix per worker."""
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
                var preferred = Int(candidates[base + position])
                for later in range(position + 1, num_candidates):
                    var cell = preferred * num_candidates + Int(candidates[base + later])
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
    CandidateCapacity: Int
](
    candidates: Pointer[CandidateIndex, MutUntrackedOrigin],
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
        CandidateCapacity * CandidateCapacity,
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
            var preferred = Int(candidates[unsafe_offset=base + position])
            for later in range(position + 1, num_candidates):
                var opponent = Int(candidates[unsafe_offset=base + later])
                _ = Atomic.fetch_add[ordering=Ordering.RELAXED](
                    counters.unsafe_offset(preferred * num_candidates + opponent),
                    UInt32(1),
                )
        ballot += stride
    barrier()

    cell = thread
    while cell < cells:
        var counted = counters[unsafe_offset=cell]
        if counted != 0:
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](preferences.unsafe_offset(cell), counted)
        cell += threads


def tally_ballots_gpu(
    candidates: Span[CandidateIndex, ImmUntrackedOrigin], num_ballots: Int, num_candidates: Int
) raises -> UInt32VoteMatrix:
    """
    Counts complete candidates into a pairwise matrix, one private matrix per block.

    Args:
        candidates: Row-major complete candidates, `num_ballots` by `num_candidates`, best first.
        num_ballots: Number of ballots in the chunk.
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
    var host_candidates = ctx.enqueue_create_host_buffer[DType.uint32](total)
    var device_candidates = ctx.enqueue_create_buffer[DType.uint32](total)
    var host_counts = ctx.enqueue_create_host_buffer[DType.uint32](cells)
    var device_counts = ctx.enqueue_create_buffer[DType.uint32](cells)
    ctx.synchronize()

    unsafe_memcpy(dest=host_candidates.unsafe_ptr(), src=candidates.unsafe_ptr(), count=total)
    host_candidates.enqueue_copy_to(device_candidates)
    device_counts.enqueue_fill(0)

    var blocks = min(divide_round_up(num_ballots, TALLY_BLOCK_SIZE), 65535)
    ctx.enqueue_function[gpu_tally_kernel[TALLY_MAX_CANDIDATES]](
        device_candidates.unsafe_ptr(),
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
    candidates: Span[CandidateIndex, ImmUntrackedOrigin],
    num_ballots: Int,
    num_candidates: Int,
    *,
    backend: Backend = Backend.cpu,
) raises -> UInt32VoteMatrix:
    """Counts complete candidates using the selected device's default kernel."""
    return tally_ballots_gpu(candidates, num_ballots, num_candidates) if backend == Backend.gpu else tally_ballots_cpu(
        candidates, num_ballots, num_candidates
    )


@fieldwise_init
struct PairwiseRelation(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """The comparison counted for each ordered candidate pair."""

    var value: UInt8
    """The encoded relation case."""
    comptime preference = Self(0)
    """The row candidate is strictly preferred."""
    comptime indifference = Self(1)
    """The candidates share a rank."""
    comptime unknown = Self(2)
    """The ballot leaves the comparison unspecified."""

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


def ballot_span[Element: Movable](values: List[Element]) -> Span[Element, ImmUntrackedOrigin]:
    """Borrow a list for a synchronous tally while its caller retains ownership."""
    return Span(unsafe_ptr=values.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](), length=len(values))


@fieldwise_init
struct RaggedBallots(Copyable, Movable):
    """Borrowed ballot arrays whose owners remain alive until tallying completes."""

    var candidates: Span[CandidateIndex, ImmUntrackedOrigin]
    """Listed candidate identifiers, concatenated across ballots."""
    var offsets: Span[BallotOffset, ImmUntrackedOrigin]
    """Ballot boundaries, including the final entry count."""
    var ranks: Span[RankLabel, ImmUntrackedOrigin]
    """Optional rank labels; an empty span uses positions within each ballot."""
    var weights: Span[VoterWeight, ImmUntrackedOrigin]
    """Optional voter weights; an empty span gives every ballot unit weight."""
    var policies: Span[PolicyCode, ImmUntrackedOrigin]
    """Optional per-voter omission policies overriding the scalar policy."""
    var num_candidates: Int
    """Size of the candidate universe, including candidates never listed."""
    var unranked: Unranked
    """Omission policy when per-voter policies are absent."""

    @always_inline
    def rank_at(self, entry: Int, begin: Int) -> RankLabel:
        return self.ranks[entry] if len(self.ranks) else RankLabel(entry - begin)

    @always_inline
    def weight_at(self, ballot: Int) -> VoterWeight:
        return self.weights[ballot] if len(self.weights) else VoterWeight(1)

    @always_inline
    def unranked_at(self, ballot: Int) -> Unranked:
        return Unranked(self.policies[ballot]) if len(self.policies) else self.unranked


def divide_round_up(value: Int, divisor: Int) -> Int:
    """Count fixed-width groups without overflowing the numerator."""
    return value // divisor + Int(value % divisor != 0)


@always_inline
def tally_ragged_row[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, RelationCount: Int = 1
](
    ballots: RaggedBallots,
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    labels: Pointer[UInt64, MutUntrackedOrigin],
    candidate: Int,
    num_candidates: Int,
    first: Int,
    last: Int,
    relation: PairwiseRelation,
):
    for ballot in range(first, last):
        var unranked = ballots.unranked_at(ballot)
        var begin = Int(ballots.offsets[ballot])
        var end = Int(ballots.offsets[ballot + 1])
        var weight = ballots.weight_at(ballot).cast[ArithmeticDataType]()
        if weight == 0:
            continue
        if RelationCount == 3 or relation != PairwiseRelation.preference or unranked == Unranked.worse:
            for opponent in range(num_candidates):
                labels[unsafe_offset=opponent] = UInt64.MAX
            for entry in range(begin, end):
                labels[unsafe_offset=Int(ballots.candidates[entry])] = UInt64(ballots.rank_at(entry, begin))
            var left = labels[unsafe_offset=candidate]
            for opponent in range(num_candidates):
                if candidate == opponent:
                    continue
                var right = labels[unsafe_offset=opponent]
                var output = -1
                if left == UInt64.MAX or right == UInt64.MAX:
                    if unranked == Unranked.unknown:
                        output = 2
                    elif left == right:
                        output = 1
                    elif left != UInt64.MAX:
                        output = 0
                elif left == right:
                    output = 1
                elif left < right:
                    output = 0
                if output >= 0 and (RelationCount == 3 or output == Int(relation.value)):
                    var cell = (output * num_candidates if RelationCount == 3 else 0) + opponent
                    counts[unsafe_offset=cell] = add_counts[ArithmeticMode](counts[unsafe_offset=cell], weight)
            continue
        var position = begin
        while position < end and Int(ballots.candidates[position]) != candidate:
            position += 1
        if position == end:
            continue
        var rank = ballots.rank_at(position, begin)
        for other in range(begin, end):
            if rank < ballots.rank_at(other, begin):
                var opponent = Int(ballots.candidates[other])
                counts[unsafe_offset=opponent] = add_counts[ArithmeticMode](counts[unsafe_offset=opponent], weight)


def tally_ragged_gpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, RelationCount: Int = 1
](
    inputs: Pointer[UInt64, ImmUntrackedOrigin],
    offset_start: Int64,
    rank_start: Int64,
    weight_start: Int64,
    policy_start: Int64,
    entry_count: Int64,
    rank_count: Int64,
    weight_count: Int64,
    policy_count: Int64,
    unranked_value: UInt8,
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    labels: Pointer[UInt64, MutUntrackedOrigin],
    scratch_width_arg: Int64,
    num_candidates_arg: Int64,
    num_ballots_arg: Int64,
    chunks_arg: Int64,
    relation_value: UInt8,
):
    var num_candidates = Int(num_candidates_arg)
    var num_ballots = Int(num_ballots_arg)
    var chunks = Int(chunks_arg)
    var task = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if task >= chunks * num_candidates:
        return
    var chunk = task // num_candidates
    var candidate = task % num_candidates
    tally_ragged_row[ArithmeticDataType, ArithmeticMode, RelationCount](
        RaggedBallots(
            Span(unsafe_ptr=inputs.unsafe_bitcast[CandidateIndex](), length=Int(entry_count)),
            Span(unsafe_ptr=inputs.unsafe_offset(Int(offset_start)), length=num_ballots + 1),
            Span(unsafe_ptr=inputs.unsafe_offset(Int(rank_start)).unsafe_bitcast[RankLabel](), length=Int(rank_count)),
            Span(unsafe_ptr=inputs.unsafe_offset(Int(weight_start)), length=Int(weight_count)),
            Span(
                unsafe_ptr=inputs.unsafe_offset(Int(policy_start)).unsafe_bitcast[PolicyCode](),
                length=Int(policy_count),
            ),
            num_candidates,
            Unranked(unranked_value),
        ),
        counts.unsafe_offset(task * num_candidates * RelationCount),
        labels.unsafe_offset(task * Int(scratch_width_arg)),
        candidate,
        num_candidates,
        num_ballots * chunk // chunks,
        num_ballots * (chunk + 1) // chunks,
        PairwiseRelation(relation_value),
    )


def reduce_tally_gpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, RelationCount: Int = 1
](
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    result: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    cells_arg: Int64,
    chunks_arg: Int64,
):
    var cells = Int(cells_arg)
    var chunks = Int(chunks_arg)
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if cell >= cells:
        return
    var total = SIMD[ArithmeticDataType, 1](0)
    for chunk in range(chunks):
        total = add_counts[ArithmeticMode](total, counts[unsafe_offset=chunk * cells + cell])
    result[unsafe_offset=cell] = total


def tally_ragged_typed[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, RelationCount: Int = 1
](ballots: RaggedBallots, relation: PairwiseRelation, backend: Backend,) raises -> List[VoteMatrix[ArithmeticDataType]]:
    var num_candidates = ballots.num_candidates
    var result = List[VoteMatrix[ArithmeticDataType]]()
    for _ in range(RelationCount):
        result.append(VoteMatrix[ArithmeticDataType](num_candidates))
    var num_ballots = len(ballots.offsets) - 1
    if num_ballots == 0:
        return result^
    var cells = num_candidates * num_candidates * RelationCount
    var scratch_width = num_candidates if RelationCount == 3 or relation != PairwiseRelation.preference else 0
    if ballots.unranked == Unranked.worse:
        scratch_width = num_candidates
    for policy in ballots.policies:
        if policy == Unranked.worse.value:
            scratch_width = num_candidates
    if backend == Backend.cpu:
        var counts = List[SIMD[ArithmeticDataType, 1]]()
        counts.resize(cells, 0)
        var counts_ptr = counts.unsafe_ptr()

        def count_row(candidate: Int) {imm}:
            var labels = List[UInt64]()
            labels.resize(scratch_width, UInt64.MAX)
            tally_ragged_row[ArithmeticDataType, ArithmeticMode, RelationCount](
                ballots,
                counts_ptr.unsafe_offset(candidate * num_candidates * RelationCount).unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                labels.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                candidate,
                num_candidates,
                0,
                num_ballots,
                relation,
            )
            # The untracked kernel pointer must not outlive its scratch allocation.
            _ = labels^

        parallelize(count_row, num_candidates)
        for cell in range(cells):
            var row = cell // (num_candidates * RelationCount)
            var output = (cell // num_candidates) % RelationCount
            result[output].data[unsafe_offset=row * num_candidates + cell % num_candidates] = counts[cell]
    else:
        # Private rows avoid unsupported 64-bit atomics on Metal.
        var chunks = min(
            num_ballots,
            max(1, min(4096 // num_candidates, (64 * 1024 * 1024) // (cells * 8 + num_candidates * scratch_width * 8))),
        )
        var ctx = DeviceContext()
        var offset_start = divide_round_up(len(ballots.candidates), 2)
        var rank_start = offset_start + len(ballots.offsets)
        var weight_start = rank_start + divide_round_up(len(ballots.ranks), 2)
        var policy_start = weight_start + len(ballots.weights)
        var input_words = policy_start + divide_round_up(len(ballots.policies), 8)
        var host_inputs = ctx.enqueue_create_host_buffer[DType.uint64](input_words)
        var device_inputs = ctx.enqueue_create_buffer[DType.uint64](input_words)
        var labels = ctx.enqueue_create_buffer[DType.uint64](max(1, chunks * num_candidates * scratch_width))
        var counts = ctx.enqueue_create_buffer[ArithmeticDataType](chunks * cells)
        var device_result = ctx.enqueue_create_buffer[ArithmeticDataType](cells)
        var host_result = ctx.enqueue_create_host_buffer[ArithmeticDataType](cells)
        ctx.synchronize()
        unsafe_memcpy(
            dest=host_inputs.unsafe_ptr().unsafe_bitcast[CandidateIndex](),
            src=ballots.candidates.unsafe_ptr(),
            count=len(ballots.candidates),
        )
        unsafe_memcpy(
            dest=host_inputs.unsafe_ptr().unsafe_offset(offset_start),
            src=ballots.offsets.unsafe_ptr(),
            count=len(ballots.offsets),
        )
        unsafe_memcpy(
            dest=host_inputs.unsafe_ptr().unsafe_offset(rank_start).unsafe_bitcast[RankLabel](),
            src=ballots.ranks.unsafe_ptr(),
            count=len(ballots.ranks),
        )
        unsafe_memcpy(
            dest=host_inputs.unsafe_ptr().unsafe_offset(weight_start),
            src=ballots.weights.unsafe_ptr(),
            count=len(ballots.weights),
        )
        unsafe_memcpy(
            dest=host_inputs.unsafe_ptr().unsafe_offset(policy_start).unsafe_bitcast[PolicyCode](),
            src=ballots.policies.unsafe_ptr(),
            count=len(ballots.policies),
        )
        host_inputs.enqueue_copy_to(device_inputs)
        counts.enqueue_fill(0)
        ctx.enqueue_function[tally_ragged_gpu[ArithmeticDataType, ArithmeticMode, RelationCount]](
            device_inputs.unsafe_ptr(),
            Int64(offset_start),
            Int64(rank_start),
            Int64(weight_start),
            Int64(policy_start),
            Int64(len(ballots.candidates)),
            Int64(len(ballots.ranks)),
            Int64(len(ballots.weights)),
            Int64(len(ballots.policies)),
            ballots.unranked.value,
            counts.unsafe_ptr(),
            labels.unsafe_ptr(),
            Int64(scratch_width),
            Int64(num_candidates),
            Int64(num_ballots),
            Int64(chunks),
            relation.value,
            grid_dim=(divide_round_up(chunks * num_candidates, 256), 1, 1),
            block_dim=(256, 1, 1),
        )
        ctx.enqueue_function[reduce_tally_gpu[ArithmeticDataType, ArithmeticMode]](
            counts.unsafe_ptr(),
            device_result.unsafe_ptr(),
            Int64(cells),
            Int64(chunks),
            grid_dim=(divide_round_up(cells, 256), 1, 1),
            block_dim=(256, 1, 1),
        )
        device_result.enqueue_copy_to(host_result)
        ctx.synchronize()
        for cell in range(cells):
            var row = cell // (num_candidates * RelationCount)
            var output = (cell // num_candidates) % RelationCount
            result[output].data[unsafe_offset=row * num_candidates + cell % num_candidates] = host_result[cell]
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


def tally_ragged_relations[
    ArithmeticDataType: DType = DType.uint64,
    ArithmeticMode: Arithmetic = Arithmetic.exact,
    RelationCount: Int = 3,
](
    ballots: RaggedBallots,
    *,
    relation: PairwiseRelation = PairwiseRelation.preference,
    backend: Backend = Backend.cpu,
) raises -> List[VoteMatrix[ArithmeticDataType]]:
    """Tally borrowed partial ballots in the selected arithmetic and storage type."""
    var num_candidates = ballots.num_candidates
    if num_candidates < 1 or num_candidates > Int.MAX // num_candidates // (8 * RelationCount):
        raise Error("Invalid candidate count or matrix size")
    if len(ballots.offsets) == 0:
        raise Error("Offsets must include their initial zero")
    var ballot_count = len(ballots.offsets) - 1
    if len(ballots.weights) not in (0, ballot_count) or len(ballots.policies) not in (0, ballot_count):
        raise Error("Weights and policies must match the ballot count")
    if len(ballots.ranks) not in (0, len(ballots.candidates)):
        raise Error("Ranks must cover all entries")
    if ballots.offsets[0] != 0 or ballots.offsets[ballot_count] != UInt64(len(ballots.candidates)):
        raise Error("Offsets must cover all entries")
    for policy in ballots.policies:
        if policy > 1:
            raise Error("Unknown unranked policy")
    var seen = List[Int](length=num_candidates, fill=-1)
    var bound = UInt64(0)
    for ballot in range(ballot_count):
        if ballots.offsets[ballot] > ballots.offsets[ballot + 1] or ballots.offsets[ballot + 1] > UInt64(
            len(ballots.candidates)
        ):
            raise Error("Offsets must be monotone and within the candidates")
        bound = add_counts[Arithmetic.saturated](bound, ballots.weight_at(ballot))
        for position in range(Int(ballots.offsets[ballot]), Int(ballots.offsets[ballot + 1])):
            var candidate = Int(ballots.candidates[position])
            if candidate >= num_candidates or seen[candidate] == ballot:
                raise Error("Candidate IDs must be in range and unique within each ballot")
            seen[candidate] = ballot
    comptime if ArithmeticMode == Arithmetic.exact:
        if bound > UInt64(SIMD[ArithmeticDataType, 1].MAX) or bound == UInt64.MAX:
            raise Error("Tally bound exceeds the selected arithmetic type")
    return tally_ragged_typed[ArithmeticDataType, ArithmeticMode, RelationCount](ballots, relation, backend)


def tally_ragged_ballots[
    ArithmeticDataType: DType = DType.uint64, ArithmeticMode: Arithmetic = Arithmetic.exact
](
    ballots: RaggedBallots,
    *,
    relation: PairwiseRelation = PairwiseRelation.preference,
    backend: Backend = Backend.cpu,
) raises -> VoteMatrix[ArithmeticDataType]:
    """Tally one relation without allocating the other relation matrices."""
    var results = tally_ragged_relations[ArithmeticDataType, ArithmeticMode, 1](
        ballots,
        relation=relation,
        backend=backend,
    )
    return results.pop()
