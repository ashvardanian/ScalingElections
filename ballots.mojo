"""
Ballot storage and the tallies that fill it, shared by the Schulze, Split Cycle and Kemeny-Young solvers.

Entries count voters preferring the row candidate to the column candidate.
The same generic storage holds strongest path capacities after solving Schulze.
"""

from std.atomic import Atomic, Ordering
from std.collections import Span
from std.memory import (
    AddressSpace,
    Layout,
    alloc,
    unsafe_memset_zero,
    unsafe_memcpy,
)
from std.random.philox import Random
from std.sys import num_logical_cores, size_of

from max.algorithm import parallelize
from max.gpu import barrier, block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceAttribute, DeviceContext
from max.gpu.memory import external_memory


comptime CandidateIndex = UInt32
comptime RankLabel = UInt32
comptime BallotOffset = UInt64
comptime VoterWeight = UInt64
comptime RankPosition = UInt64
"""A rank label or in-ballot offset, whose maximum marks a candidate the ballot leaves unranked."""
comptime PolicyCode = UInt8
comptime RelationCode = UInt8


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
    comptime saturated64 = Self(65)
    comptime uint64 = Self(64)
    comptime uint32 = Self(32)
    comptime uint16 = Self(16)

    def bits(self) -> Int:
        return 64 if self == Self.saturated64 else Int(self.value)

    @staticmethod
    def parse(value: String) raises -> Self:
        if value == "auto":
            return Self.auto
        if value == "saturated64":
            return Self.saturated64
        if value == "uint64":
            return Self.uint64
        if value == "uint32":
            return Self.uint32
        if value == "uint16":
            return Self.uint16
        raise Error("score_type must be auto, saturated64, uint64, uint32, or uint16")


@fieldwise_init
struct Arithmetic(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    comptime exact = Self(0)
    comptime saturated = Self(1)


def with_score_type[
    Result: Movable, Body: def[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises -> Result
](score_type: ScoreType, body: Body) raises -> Result:
    """Runs `body` with the storage type and addition semantics a resolved score type names."""
    if score_type == ScoreType.saturated64:
        return body[DType.uint64, Arithmetic.saturated]()
    if score_type == ScoreType.uint64:
        return body[DType.uint64, Arithmetic.exact]()
    if score_type == ScoreType.uint32:
        return body[DType.uint32, Arithmetic.exact]()
    if score_type == ScoreType.uint16:
        return body[DType.uint16, Arithmetic.exact]()
    raise Error("Resolve the automatic score type first")


comptime TALLY_DENSE_COUNTER = DType.uint32
"""The dense GPU tally's shared counters, the only width Metal adds atomically.

It is the one widening the counts ever take: 16-bit counts accumulate here and narrow on copy-out,
while 64-bit and saturated counts take the private-row ragged kernel instead.
"""


def tally_dense_serves_gpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](ctx: DeviceContext, num_candidates: Int) raises -> Bool:
    """Whether the dense GPU kernel can count this field in this arithmetic within one block's shared memory."""
    comptime if ArithmeticMode == Arithmetic.saturated or size_of[Scalar[ArithmeticDataType]]() > size_of[
        Scalar[TALLY_DENSE_COUNTER]
    ]():
        return False
    return tally_dense_shared_bytes(num_candidates) <= ctx.get_attribute(DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK)


def tally_dense_shared_bytes(num_candidates: Int) -> Int:
    """Shared memory one dense tally block needs: a private counter matrix."""
    return num_candidates * num_candidates * size_of[Scalar[TALLY_DENSE_COUNTER]]()


comptime GPU_BLOCK_SIZE = 256
"""Threads per block for every one-dimensional launch, a tuning knob: on the M5 Pro, 128 to 1024 threads
time within 2% on Kemeny over 25 candidates and within 8% on a 200-candidate ragged tally, 256 at the low end.
"""


def gpu_grid_limit(ctx: DeviceContext, blocks: Int) raises -> Int:
    """`blocks`, capped at the widest one-dimensional grid the device launches."""
    return min(blocks, Int(ctx.get_attribute(DeviceAttribute.MAX_GRID_DIM_X)))


@fieldwise_init
struct Unranked(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    var value: PolicyCode

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
struct VoteMatrix[StoredCountDataType: DType](Movable):
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


# endregion Matrix

# region Preferences


comptime NATIONAL_ELECTORATE = 350_000_000
"""Voters in a national electorate, the widest count a synthetic preference cell draws."""


def populate_preferences_from_ranking[
    StoredCountDataType: DType
](mut preferences: VoteMatrix[StoredCountDataType], ranking: List[Int]):
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


def generate_random_preferences[
    StoredCountDataType: DType
](num_candidates: Int, num_voters: Int, seed_value: Int) raises -> VoteMatrix[StoredCountDataType]:
    """
    Draws a preference matrix for a synthetic election of the requested shape.

    Args:
        num_candidates: Number of candidates.
        num_voters: Number of voters. If 0, draws the counts themselves at random.
        seed_value: Seeds the counter-based generator, so a run reproduces exactly.

    Returns:
        Random preference matrix.
    """
    var preferences = VoteMatrix[StoredCountDataType](num_candidates)

    if num_voters == 0:

        def fill_row(row: Int) {imm}:
            # Seeded per row, so parallel workers share no state to race on.
            var generator = Random(seed=UInt64(seed_value), offset=UInt64(row))
            var bound = UInt32(NATIONAL_ELECTORATE + 1)
            var column = 0
            while column < num_candidates:
                # Every lane of the draw is spent, rather than three in four discarded.
                var draws = generator.step()
                var lanes = min(len(draws), num_candidates - column)
                for lane in range(lanes):
                    var count = draws[lane] % bound if row != column + lane else UInt32(0)
                    preferences.data[unsafe_offset=row * num_candidates + column + lane] = count.cast[
                        StoredCountDataType
                    ]()
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

# region Dense Tally


def tally_ballots_cpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    candidates: Span[CandidateIndex, ImmUntrackedOrigin],
    num_ballots: Int,
    num_candidates: Int,
    preferences: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
):
    """Counts complete rankings in parallel with one private matrix per worker, overwriting `preferences`."""
    comptime Count = SIMD[ArithmeticDataType, 1]
    var cells = num_candidates * num_candidates
    var workers = max(1, min(num_ballots, num_logical_cores()))
    var counts = List[Count](length=workers * cells, fill=0)
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
                    private_counts[unsafe_offset=cell] = add_counts[ArithmeticMode](
                        private_counts[unsafe_offset=cell], Count(1)
                    )

    parallelize(count_chunk, workers)
    for cell in range(cells):
        var total = Count(0)
        for worker in range(workers):
            total = add_counts[ArithmeticMode](total, counts[worker * cells + cell])
        preferences[unsafe_offset=cell] = total


def tally_dense_kernel(
    candidates: Pointer[CandidateIndex, MutUntrackedOrigin],
    num_ballots_arg: BallotOffset,
    num_candidates_arg: CandidateIndex,
    preferences: Pointer[Scalar[TALLY_DENSE_COUNTER], MutUntrackedOrigin],
):
    """Accumulates each block's ballots into a shared matrix, one ballot to a thread, merging once at exit."""
    var num_ballots = Int(num_ballots_arg)
    var num_candidates = Int(num_candidates_arg)
    var cells = num_candidates * num_candidates
    var thread = Int(thread_idx.x)
    var threads = Int(block_dim.x)

    var counters = external_memory[
        Scalar[TALLY_DENSE_COUNTER], address_space=AddressSpace.SHARED, alignment=size_of[Scalar[TALLY_DENSE_COUNTER]]()
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
                    Scalar[TALLY_DENSE_COUNTER](1),
                )
        ballot += stride
    barrier()

    cell = thread
    while cell < cells:
        var counted = counters[unsafe_offset=cell]
        if counted != 0:
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](preferences.unsafe_offset(cell), counted)
        cell += threads


def tally_ballots_gpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    candidates: Span[CandidateIndex, ImmUntrackedOrigin],
    num_ballots: Int,
    num_candidates: Int,
    preferences: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
) raises:
    """
    Counts complete rankings into a pairwise matrix, one private matrix per block.

    Args:
        candidates: Row-major complete rankings, `num_ballots` by `num_candidates`, best first.
        num_ballots: Number of ballots in the chunk.
        num_candidates: The number of candidates, which `tally_dense_serves_gpu` must accept.
        preferences: Host matrix of `num_candidates` squared counts, overwritten.
    """
    var ctx = DeviceContext()
    if not tally_dense_serves_gpu[ArithmeticDataType, ArithmeticMode](ctx, num_candidates):
        raise Error("The dense GPU tally cannot count this field in this arithmetic")

    var cells = num_candidates * num_candidates
    if num_ballots == 0:
        unsafe_memset_zero(preferences, cells)
        return
    var total = num_ballots * num_candidates

    var device_candidates = ctx.enqueue_create_buffer[CandidateIndex.dtype](total)
    var device_counts = ctx.enqueue_create_buffer[TALLY_DENSE_COUNTER](cells)
    ctx.enqueue_copy(dst_buf=device_candidates, src_ptr=candidates.unsafe_ptr())
    device_counts.enqueue_fill(0)

    ctx.enqueue_function[tally_dense_kernel](
        device_candidates.unsafe_ptr(),
        BallotOffset(num_ballots),
        CandidateIndex(num_candidates),
        device_counts.unsafe_ptr(),
        grid_dim=(gpu_grid_limit(ctx, divide_round_up(num_ballots, GPU_BLOCK_SIZE)), 1, 1),
        block_dim=(GPU_BLOCK_SIZE, 1, 1),
        shared_mem_bytes=tally_dense_shared_bytes(num_candidates),
    )
    comptime if ArithmeticDataType == TALLY_DENSE_COUNTER:
        ctx.enqueue_copy(
            dst_ptr=rebind[Pointer[Scalar[TALLY_DENSE_COUNTER], MutUntrackedOrigin]](preferences), src_buf=device_counts
        )
        ctx.synchronize()
    else:
        var host_counts = ctx.enqueue_create_host_buffer[TALLY_DENSE_COUNTER](cells)
        device_counts.enqueue_copy_to(host_counts)
        ctx.synchronize()
        for cell in range(cells):
            preferences[unsafe_offset=cell] = host_counts[cell].cast[ArithmeticDataType]()


def tally_ballots[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    candidates: Span[CandidateIndex, ImmUntrackedOrigin],
    num_ballots: Int,
    num_candidates: Int,
    preferences: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    *,
    backend: Backend,
) raises:
    """Counts complete rankings into `preferences` using the selected device's dense kernel."""
    if backend == Backend.gpu:
        return tally_ballots_gpu[ArithmeticDataType, ArithmeticMode](
            candidates, num_ballots, num_candidates, preferences
        )
    tally_ballots_cpu[ArithmeticDataType, ArithmeticMode](candidates, num_ballots, num_candidates, preferences)


# endregion Dense Tally

# region Ragged Tally


@fieldwise_init
struct PairwiseRelation(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """The comparison counted for each ordered candidate pair."""

    var value: RelationCode
    """The encoded relation case, which is also its plane among the three relation matrices."""
    comptime preference = Self(0)
    """The row candidate is strictly preferred."""
    comptime indifference = Self(1)
    """The candidates share a rank."""
    comptime unknown = Self(2)
    """The ballot leaves the comparison unspecified."""
    comptime all = Self(3)
    """All three comparison matrices in preference, indifference, unknown order."""

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def planes(self) -> Int:
        """How many relation matrices a tally of this relation fills."""
        return 3 if self == Self.all else 1


@always_inline
def classify_relation(left: RankPosition, right: RankPosition, unranked: Unranked) -> PairwiseRelation:
    """How a ballot relates two candidates' positions, where `RankPosition.MAX` marks an omitted candidate."""
    if unranked == Unranked.unknown and (left == RankPosition.MAX or right == RankPosition.MAX):
        return PairwiseRelation.unknown
    if left == right:
        return PairwiseRelation.indifference
    return PairwiseRelation.preference


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
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, Relation: PairwiseRelation
](
    ballots: RaggedBallots,
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    labels: Pointer[RankPosition, MutUntrackedOrigin],
    candidate: Int,
    num_candidates: Int,
    first: Int,
    last: Int,
):
    """Adds ballots `[first, last)` to `candidate`'s row; relation planes sit `num_candidates` squared apart."""
    var plane_stride = num_candidates * num_candidates
    for ballot in range(first, last):
        var unranked = ballots.unranked_at(ballot)
        var begin = Int(ballots.offsets[ballot])
        var end = Int(ballots.offsets[ballot + 1])
        var weight = ballots.weight_at(ballot).cast[ArithmeticDataType]()
        if weight == 0:
            continue
        if Relation != PairwiseRelation.preference or unranked == Unranked.worse:
            for opponent in range(num_candidates):
                labels[unsafe_offset=opponent] = RankPosition.MAX
            for entry in range(begin, end):
                labels[unsafe_offset=Int(ballots.candidates[entry])] = RankPosition(ballots.rank_at(entry, begin))
            var left = labels[unsafe_offset=candidate]
            for opponent in range(num_candidates):
                if candidate == opponent:
                    continue
                var right = labels[unsafe_offset=opponent]
                var actual = classify_relation(left, right, unranked)
                if actual == PairwiseRelation.preference and left >= right:
                    continue
                if Relation != PairwiseRelation.all and actual != Relation:
                    continue
                var cell = (Int(actual.value) * plane_stride if Relation == PairwiseRelation.all else 0) + opponent
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


def tally_ragged_kernel[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, Relation: PairwiseRelation
](
    inputs: Pointer[UInt64, ImmUntrackedOrigin],
    offset_start: BallotOffset,
    rank_start: BallotOffset,
    weight_start: BallotOffset,
    policy_start: BallotOffset,
    entry_count: BallotOffset,
    rank_count: BallotOffset,
    weight_count: BallotOffset,
    policy_count: BallotOffset,
    unranked_value: PolicyCode,
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    labels: Pointer[RankPosition, MutUntrackedOrigin],
    scratch_width_arg: CandidateIndex,
    num_candidates_arg: CandidateIndex,
    num_ballots_arg: BallotOffset,
    chunks_arg: BallotOffset,
):
    """Tallies one ballot chunk into one candidate's row of that chunk's private planes."""
    comptime planes = Relation.planes()
    var num_candidates = Int(num_candidates_arg)
    var num_ballots = Int(num_ballots_arg)
    var chunks = Int(chunks_arg)
    var task = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if task >= chunks * num_candidates:
        return
    var chunk = task // num_candidates
    var candidate = task % num_candidates
    tally_ragged_row[ArithmeticDataType, ArithmeticMode, Relation](
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
        counts.unsafe_offset((chunk * num_candidates * planes + candidate) * num_candidates),
        labels.unsafe_offset(task * Int(scratch_width_arg)),
        candidate,
        num_candidates,
        num_ballots * chunk // chunks,
        num_ballots * (chunk + 1) // chunks,
    )


def tally_reduce_kernel[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    result: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    cells_arg: BallotOffset,
    chunks_arg: BallotOffset,
):
    """Sums every chunk's private planes into one set."""
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
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic, Relation: PairwiseRelation
](ballots: RaggedBallots, backend: Backend, counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],) raises:
    """Overwrites `counts` with the relation's planes of `num_candidates` squared cells, without validating."""
    var num_candidates = ballots.num_candidates
    var cells = num_candidates * num_candidates * Relation.planes()
    var num_ballots = len(ballots.offsets) - 1
    unsafe_memset_zero(counts, cells)
    if num_ballots == 0:
        return
    var scratch_width = num_candidates if Relation != PairwiseRelation.preference else 0
    if ballots.unranked == Unranked.worse:
        scratch_width = num_candidates
    for policy in ballots.policies:
        if policy == Unranked.worse.value:
            scratch_width = num_candidates

    if backend == Backend.cpu:

        def count_row(candidate: Int) {imm}:
            var labels = List[RankPosition](length=scratch_width, fill=RankPosition.MAX)
            tally_ragged_row[ArithmeticDataType, ArithmeticMode, Relation](
                ballots,
                counts.unsafe_offset(candidate * num_candidates),
                labels.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                candidate,
                num_candidates,
                0,
                num_ballots,
            )
            # The untracked kernel pointer must not outlive its scratch allocation.
            _ = labels^

        parallelize(count_row, num_candidates)
        return

    # Private rows avoid unsupported 64-bit atomics on Metal. One chunk of ballots per resident row
    # of threads saturates the device; more chunks would only multiply the private planes.
    var ctx = DeviceContext()
    var resident_threads = Int(ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)) * Int(
        ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK)
    )
    var chunk_bytes = (
        cells * size_of[SIMD[ArithmeticDataType, 1]]() + num_candidates * scratch_width * size_of[RankPosition]()
    )
    var free_bytes = Int(ctx.get_memory_info()[0])
    var chunks = max(1, min(num_ballots, resident_threads // num_candidates, free_bytes // chunk_bytes))
    var offset_start = divide_round_up(len(ballots.candidates), 2)
    var rank_start = offset_start + len(ballots.offsets)
    var weight_start = rank_start + divide_round_up(len(ballots.ranks), 2)
    var policy_start = weight_start + len(ballots.weights)
    var input_words = policy_start + divide_round_up(len(ballots.policies), 8)
    var host_inputs = ctx.enqueue_create_host_buffer[DType.uint64](input_words)
    var device_inputs = ctx.enqueue_create_buffer[DType.uint64](input_words)
    var labels = ctx.enqueue_create_buffer[RankPosition.dtype](max(1, chunks * num_candidates * scratch_width))
    var chunk_counts = ctx.enqueue_create_buffer[ArithmeticDataType](chunks * cells)
    var device_counts = ctx.enqueue_create_buffer[ArithmeticDataType](cells)
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
    chunk_counts.enqueue_fill(0)
    ctx.enqueue_function[tally_ragged_kernel[ArithmeticDataType, ArithmeticMode, Relation]](
        device_inputs.unsafe_ptr(),
        BallotOffset(offset_start),
        BallotOffset(rank_start),
        BallotOffset(weight_start),
        BallotOffset(policy_start),
        BallotOffset(len(ballots.candidates)),
        BallotOffset(len(ballots.ranks)),
        BallotOffset(len(ballots.weights)),
        BallotOffset(len(ballots.policies)),
        ballots.unranked.value,
        chunk_counts.unsafe_ptr(),
        labels.unsafe_ptr(),
        CandidateIndex(scratch_width),
        CandidateIndex(num_candidates),
        BallotOffset(num_ballots),
        BallotOffset(chunks),
        grid_dim=(divide_round_up(chunks * num_candidates, GPU_BLOCK_SIZE), 1, 1),
        block_dim=(GPU_BLOCK_SIZE, 1, 1),
    )
    ctx.enqueue_function[tally_reduce_kernel[ArithmeticDataType, ArithmeticMode]](
        chunk_counts.unsafe_ptr(),
        device_counts.unsafe_ptr(),
        BallotOffset(cells),
        BallotOffset(chunks),
        grid_dim=(divide_round_up(cells, GPU_BLOCK_SIZE), 1, 1),
        block_dim=(GPU_BLOCK_SIZE, 1, 1),
    )
    ctx.enqueue_copy(dst_ptr=counts, src_buf=device_counts)
    ctx.synchronize()


def validate_ragged_ballots(ballots: RaggedBallots) raises -> VoterWeight:
    """Checks the borrowed arrays agree with each other, returning the saturating total weight."""
    var num_candidates = ballots.num_candidates
    if num_candidates < 1:
        raise Error("num_candidates must be positive")
    if len(ballots.offsets) == 0:
        raise Error("Offsets must include their initial zero")
    var num_ballots = len(ballots.offsets) - 1
    if len(ballots.weights) not in (0, num_ballots):
        raise Error("Weights must match the ballot count")
    if len(ballots.policies) not in (0, num_ballots):
        raise Error("Unranked policies must match the ballot count")
    if len(ballots.ranks) not in (0, len(ballots.candidates)):
        raise Error("Ranks must match the entries length")
    if ballots.offsets[0] != 0 or ballots.offsets[num_ballots] != BallotOffset(len(ballots.candidates)):
        raise Error("Offsets must start at zero and end at the entries length")
    for policy in ballots.policies:
        if policy > Unranked.worse.value:
            raise Error("unranked must be unknown or worse")
    var seen = List[Int](length=num_candidates, fill=-1)
    var bound = VoterWeight(0)
    for ballot in range(num_ballots):
        var end = ballots.offsets[ballot + 1]
        if ballots.offsets[ballot] > end or end > BallotOffset(len(ballots.candidates)):
            raise Error("Offsets must be monotone and within the entries length")
        bound = add_counts[Arithmetic.saturated](bound, ballots.weight_at(ballot))
        for position in range(Int(ballots.offsets[ballot]), Int(ballots.offsets[ballot + 1])):
            var candidate = Int(ballots.candidates[position])
            if candidate >= num_candidates or seen[candidate] == ballot:
                raise Error("Candidate IDs must be in range and distinct within each ballot")
            seen[candidate] = ballot
    return bound


def resolve_tally_score_type(bound: VoterWeight, requested: ScoreType) raises -> ScoreType:
    """Resolves or validates tally arithmetic from the saturating total voter weight."""
    if requested == ScoreType.auto:
        if bound == VoterWeight.MAX:
            return ScoreType.saturated64
        return ScoreType.uint64 if bound > VoterWeight(UInt32.MAX) else ScoreType.uint32
    if (
        (requested == ScoreType.uint64 and bound == VoterWeight.MAX)
        or (requested == ScoreType.uint32 and bound > VoterWeight(UInt32.MAX))
        or (requested == ScoreType.uint16 and bound > VoterWeight(UInt16.MAX))
    ):
        raise Error("Tally bound exceeds the selected arithmetic type")
    return requested


def tally_ragged_relations[
    ArithmeticDataType: DType,
    ArithmeticMode: Arithmetic,
    Relation: PairwiseRelation,
](
    ballots: RaggedBallots,
    counts: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    *,
    backend: Backend,
) raises:
    """Validates borrowed partial ballots, then overwrites the relation's caller-owned planes with their tally."""
    var bound = validate_ragged_ballots(ballots)
    comptime if ArithmeticMode == Arithmetic.exact:
        if bound > VoterWeight(SIMD[ArithmeticDataType, 1].MAX) or bound == VoterWeight.MAX:
            raise Error("Tally bound exceeds the selected arithmetic type")
    tally_ragged_typed[ArithmeticDataType, ArithmeticMode, Relation](ballots, backend, counts)


# endregion Ragged Tally
