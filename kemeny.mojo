"""
Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

Schulze answers who wins; Kemeny-Young answers what the whole ordering should be, choosing the
ranking that disagrees with the fewest ballot pairs. That optimum is NP-hard to reach by search,
so this is the Held-Karp subset dynamic program instead: `O(n * 2^n)` time against `O(2^n)`
memory, exact rather than approximate, and practical to roughly two dozen candidates.
"""

from std.bit import count_trailing_zeros
from std.sys import num_logical_cores, size_of

from max.algorithm import parallelize
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from ballots import (
    Arithmetic,
    Backend,
    CandidateIndex,
    GPU_BLOCK_SIZE,
    ScoreType,
    VoteMatrixView,
    VoterWeight,
    add_counts,
    divide_round_up,
    with_score_type,
)


comptime SubsetMask = UInt64
"""A set of candidates, one bit each, which takes 34 bits at the widest field."""

comptime SubsetRank = UInt64
"""A subset's colex rank within the layer of subsets sharing its population count."""

comptime BinomialCount = UInt32
"""One entry of Pascal's triangle, whose width bounds the widest field the tables address."""

comptime KemenyTraceWord = UInt64
"""One word of the packed trace: a candidate, the optimum, the multiplicity, or a winner flag."""


def kemeny_widest_field() -> Int:
    """The most candidates whose widest layer, the middle binomial, a `BinomialCount` still counts."""
    var candidates = 0
    while True:
        var next = candidates + 1
        var half = next // 2
        var middle = SubsetRank(1)
        for step in range(1, half + 1):
            middle = middle * SubsetRank(next - half + step) // SubsetRank(step)
        if middle > SubsetRank(BinomialCount.MAX):
            return candidates
        candidates = next


comptime KEMENY_MAX_CANDIDATES = kemeny_widest_field()
"""The widest field the colex tables address; memory, checked at run time, decides the rest."""


# region Subset Sums


def require_kemeny_width(num_candidates: Int) raises:
    """Refuses a field the exact table cannot address, before anything is allocated for it."""
    if num_candidates < 1 or num_candidates > KEMENY_MAX_CANDIDATES:
        raise Error(t"Kemeny is exact from 1 to {KEMENY_MAX_CANDIDATES} candidates")


struct KemenySumsView[ArithmeticDataType: DType, ArithmeticMode: Arithmetic](
    Copyable, ImplicitlyCopyable, Movable, TrivialRegisterPassable
):
    """The split subset sums as one flat table, every low half first, as either processor addresses it."""

    comptime Count = SIMD[Self.ArithmeticDataType, 1]

    var values: Pointer[Self.Count, ImmUntrackedOrigin]
    var num_candidates: Int
    var low_bits: SubsetMask
    var low_states: Int
    var high_states: Int

    def __init__(out self, values: Pointer[Self.Count, ImmUntrackedOrigin], num_candidates: Int):
        self.values = values
        self.num_candidates = num_candidates
        self.low_bits = SubsetMask(num_candidates // 2)
        self.low_states = 1 << (num_candidates // 2)
        self.high_states = 1 << (num_candidates - num_candidates // 2)

    @always_inline
    def against(self, candidate: Int, subset: SubsetMask) -> Self.Count:
        """Votes that preferred this candidate to every member of the subset."""
        var low_half = Int(subset & SubsetMask(self.low_states - 1))
        var high_half = Int(subset >> self.low_bits)
        var low = self.values[unsafe_offset=candidate * self.low_states + low_half]
        var high = self.values[
            unsafe_offset=self.num_candidates * self.low_states + candidate * self.high_states + high_half
        ]
        return add_counts[Self.ArithmeticMode](low, high)


struct KemenySums[ArithmeticDataType: DType, ArithmeticMode: Arithmetic](Movable):
    """Votes each candidate loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
    """

    comptime Count = SIMD[Self.ArithmeticDataType, 1]

    var values: List[Self.Count]
    var num_candidates: Int

    def __init__[StoredCountDataType: DType](out self, preferences: VoteMatrixView[StoredCountDataType, _]):
        self.num_candidates = preferences.num_candidates
        var low_bits = self.num_candidates // 2
        var low_states = 1 << low_bits
        var high_states = 1 << (self.num_candidates - low_bits)
        var high_start = self.num_candidates * low_states
        self.values = List[Self.Count](length=high_start + self.num_candidates * high_states, fill=0)
        for candidate in range(self.num_candidates):
            self.accumulate(candidate * low_states, low_states, preferences, candidate, 0)
            self.accumulate(high_start + candidate * high_states, high_states, preferences, candidate, low_bits)

    def accumulate[
        StoredCountDataType: DType
    ](
        mut self,
        base: Int,
        states: Int,
        preferences: VoteMatrixView[StoredCountDataType, _],
        candidate: Int,
        first_opponent: Int,
    ):
        """Folds one candidate's votes into every half-subset that contains each opponent."""
        var opponent = first_opponent
        var bit = 1
        while bit < states:
            # Diagonal entries are skipped so they cannot overflow a narrowed subset sum.
            var votes = Self.Count(preferences[candidate, opponent]) if candidate != opponent else Self.Count(0)
            for half in range(bit, states):
                if half & bit:
                    self.values[base + half] = add_counts[Self.ArithmeticMode](self.values[base + (half ^ bit)], votes)
            opponent += 1
            bit <<= 1

    def view(self) -> KemenySumsView[Self.ArithmeticDataType, Self.ArithmeticMode]:
        return KemenySumsView[Self.ArithmeticDataType, Self.ArithmeticMode](
            self.values.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](), self.num_candidates
        )


# endregion Subset Sums


# region Cost Table


def kemeny_binomials(num_candidates: Int) -> List[BinomialCount]:
    """Pascal's triangle, entry `upper * (num_candidates + 1) + lower` counting `C(upper, lower)`."""
    var stride = num_candidates + 1
    var table = List[BinomialCount](length=stride * stride, fill=0)
    for upper in range(stride):
        table[upper * stride] = 1
        for lower in range(1, upper + 1):
            table[upper * stride + lower] = (
                table[(upper - 1) * stride + lower] + table[(upper - 1) * stride + lower - 1]
            )
    return table^


@always_inline
def kemeny_layer_states(binomials: List[BinomialCount], num_candidates: Int, seated: Int) -> SubsetRank:
    """How many subsets seat exactly `seated` of the candidates."""
    return SubsetRank(binomials[num_candidates * (num_candidates + 1) + seated])


@always_inline
def kemeny_unrank_colex(
    binomials: Pointer[BinomialCount, _], num_candidates: Int, seated: Int, rank: SubsetRank
) -> SubsetMask:
    """The subset mask a colex rank names among those seating `seated` of the candidates."""
    var stride = num_candidates + 1
    var subset = SubsetMask(0)
    var remaining = seated
    var position = rank
    var candidate = num_candidates
    while remaining != 0 and candidate != 0:
        candidate -= 1
        var below = SubsetRank(binomials[unsafe_offset=candidate * stride + remaining])
        if position < below:
            continue
        position -= below
        subset |= SubsetMask(1) << SubsetMask(candidate)
        remaining -= 1
    return subset


@always_inline
def kemeny_next_colex(subset: SubsetMask) -> SubsetMask:
    """The next mask of the same population count, which is the next colex rank in its layer."""
    var lowest = subset & (~subset + 1)
    var rippled = subset + lowest
    return rippled | ((rippled ^ subset) >> (SubsetMask(2) + count_trailing_zeros(subset)))


@always_inline
def kemeny_best_cost[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    sums: KemenySumsView[ArithmeticDataType, ArithmeticMode],
    costs: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    subset: SubsetMask,
) -> SIMD[ArithmeticDataType, 1]:
    """The least disagreement seating `subset`, trying each member in the last seat.

    Seating a candidate last within the subset costs the votes that preferred it to the others.
    """
    var best = SIMD[ArithmeticDataType, 1].MAX
    var members = subset
    while members != 0:
        var bit = members & (~members + 1)
        var rest = subset ^ bit
        var candidate = Int(count_trailing_zeros(bit))
        best = min(best, add_counts[ArithmeticMode](costs[unsafe_offset=Int(rest)], sums.against(candidate, rest)))
        members ^= bit
    return best


def kemeny_score_bound[
    StoredCountDataType: DType
](preferences: VoteMatrixView[StoredCountDataType, _]) raises -> VoterWeight:
    """A saturating bound on any ordering's disagreement: the larger side of every pair, summed."""
    require_kemeny_width(preferences.num_candidates)
    var bound = VoterWeight(0)
    for row in range(preferences.num_candidates):
        for column in range(row + 1, preferences.num_candidates):
            bound = add_counts[Arithmetic.saturated](
                bound, VoterWeight(max(preferences[row, column], preferences[column, row]))
            )
    return bound


def resolve_score_type[
    StoredCountDataType: DType
](preferences: VoteMatrixView[StoredCountDataType, _], requested_type: ScoreType = ScoreType.auto) raises -> ScoreType:
    """Resolves or validates arithmetic, reserving its maximum value for unreachable states."""
    var bound = kemeny_score_bound(preferences)
    if requested_type == ScoreType.auto:
        if bound == VoterWeight.MAX:
            return ScoreType.saturated64
        return ScoreType.uint64 if bound >= VoterWeight(UInt32.MAX) else ScoreType.uint32
    if requested_type != ScoreType.saturated64 and bound >= (
        VoterWeight.MAX >> VoterWeight(64 - requested_type.bits())
    ):
        raise Error("Kemeny score bound exceeds the selected arithmetic type")
    return requested_type


def require_kemeny_score_range[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](preferences: VoteMatrixView[StoredCountDataType, _]) raises:
    comptime assert (
        ArithmeticDataType == DType.uint16 or ArithmeticDataType == DType.uint32 or ArithmeticDataType == DType.uint64
    )
    var bound = kemeny_score_bound(preferences)
    comptime if ArithmeticMode == Arithmetic.exact:
        if bound >= VoterWeight(SIMD[ArithmeticDataType, 1].MAX):
            raise Error("Kemeny score bound exceeds the selected arithmetic type")


def compute_kemeny_costs_cpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    sums: KemenySums[ArithmeticDataType, ArithmeticMode],
    costs: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
):
    """
    Fills `costs`, one entry per subset, layer by layer across the CPU cores.

    Entry `subset` is the least disagreement achievable seating those candidates in the leading
    places, counting only the pairs inside it. Clearing a bit drops the population count by
    exactly one, so one layer of subsets depends only on the layer below it.
    """
    var num_candidates = sums.num_candidates
    var binomials = kemeny_binomials(num_candidates)
    var binomials_ptr = binomials.unsafe_ptr()
    var sums_view = sums.view()
    costs[unsafe_offset=0] = 0

    for seated in range(1, num_candidates + 1):
        # Copied because a `parallelize` closure capturing the induction variable faults at -O1.
        var layer_seated = seated
        var layer_states = Int(kemeny_layer_states(binomials, num_candidates, seated))
        # Every subset costs the same, so one equal slice of ranks per core balances the layer.
        var chunks = min(num_logical_cores(), layer_states)

        def fill_layer_chunk(chunk: Int) {imm}:
            var first_rank = layer_states * chunk // chunks
            var last_rank = layer_states * (chunk + 1) // chunks
            var subset = kemeny_unrank_colex(binomials_ptr, num_candidates, layer_seated, SubsetRank(first_rank))
            for _ in range(first_rank, last_rank):
                costs[unsafe_offset=Int(subset)] = kemeny_best_cost(sums_view, costs, subset)
                subset = kemeny_next_colex(subset)

        parallelize(fill_layer_chunk, chunks)
    _ = binomials^


# endregion Cost Table


# region Ranking


@fieldwise_init
struct RankingMultiplicity(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Whether one or several complete orderings achieve the minimum disagreement."""

    var value: UInt8
    """The encoded multiplicity case."""

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def name(self) -> String:
        return String("unique") if self == Self.unique else String("multiple")

    comptime unique = Self(0)
    """Exactly one complete ordering is optimal."""
    comptime multiple = Self(1)
    """Several complete orderings are optimal, possibly with the same winner."""


@fieldwise_init
struct KemenySolution(Movable):
    """An optimal Kemeny-Young ordering, its disagreement score, and its tie metadata."""

    var ranking: List[Int]
    """One optimal ordering, best candidate first."""
    var winners: List[Int]
    """All candidates that can rank first in an optimal ordering."""
    var multiplicity: RankingMultiplicity
    """Whether the complete optimal ordering is unique."""
    var score: KemenyTraceWord
    """Minimum total weight of pairwise preferences contradicted by the ordering."""


@fieldwise_init
struct KemenyTraceError(Copyable, Equatable, ImplicitlyCopyable, Movable, Writable):
    """Why a completed cost table yields no solution."""

    var value: UInt8
    comptime overflow = Self(0)
    """The optimum equals the maximum the selected arithmetic reserves as its sentinel."""
    comptime inconsistent = Self(1)
    """No member of some subset explains its cost, which only a corrupted table produces."""

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def write_to(self, mut writer: Some[Writer]):
        if self == Self.overflow:
            writer.write("Kemeny optimum reaches the overflow sentinel")
        else:
            writer.write("The Kemeny cost table disagrees with its own sums")


@always_inline
def kemeny_trace[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    sums: KemenySumsView[ArithmeticDataType, ArithmeticMode],
    costs: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    trace: Pointer[KemenyTraceWord, MutUntrackedOrigin],
):
    """
    Walks a completed cost table back into `trace`: the ranking best first, then the optimum,
    the multiplicity, and one winner flag per candidate.

    Both processors run this one walk, which is what keeps their tie-breaking identical.
    """
    var num_candidates = sums.num_candidates
    var full = (SubsetMask(1) << SubsetMask(num_candidates)) - 1
    var optimum = costs[unsafe_offset=Int(full)]
    trace[unsafe_offset=num_candidates] = KemenyTraceWord(optimum)
    if optimum == SIMD[ArithmeticDataType, 1].MAX:
        return
    trace[unsafe_offset=num_candidates + 1] = KemenyTraceWord(RankingMultiplicity.unique.value)
    for candidate in range(num_candidates):
        var bit = SubsetMask(1) << SubsetMask(candidate)
        var score = costs[unsafe_offset=Int(full ^ bit)]
        for rival in range(num_candidates):
            if rival != candidate:
                score = add_counts[ArithmeticMode](score, sums.against(rival, bit))
        trace[unsafe_offset=num_candidates + 2 + candidate] = KemenyTraceWord(score == optimum)
    var subset = full
    for place in range(num_candidates - 1, -1, -1):
        var chosen = -1
        for candidate in range(num_candidates):
            var bit = SubsetMask(1) << SubsetMask(candidate)
            if (subset & bit) == 0:
                continue
            var rest = subset ^ bit
            var score = add_counts[ArithmeticMode](costs[unsafe_offset=Int(rest)], sums.against(candidate, rest))
            if score != costs[unsafe_offset=Int(subset)]:
                continue
            if chosen < 0:
                chosen = candidate
            else:
                trace[unsafe_offset=num_candidates + 1] = KemenyTraceWord(RankingMultiplicity.multiple.value)
        trace[unsafe_offset=place] = KemenyTraceWord(chosen)
        if chosen < 0:
            return
        subset ^= SubsetMask(1) << SubsetMask(chosen)


def kemeny_solution_from_trace[
    ArithmeticDataType: DType
](trace: List[KemenyTraceWord]) raises KemenyTraceError -> KemenySolution:
    """Decodes the layout `kemeny_trace` writes, refusing an optimum the arithmetic reserves as its sentinel."""
    var num_candidates = (len(trace) - 2) // 2
    var optimum = trace[num_candidates]
    if optimum == KemenyTraceWord(SIMD[ArithmeticDataType, 1].MAX):
        raise KemenyTraceError.overflow
    var ranking = List[Int]()
    for place in range(num_candidates):
        if trace[place] >= KemenyTraceWord(num_candidates):
            raise KemenyTraceError.inconsistent
        ranking.append(Int(trace[place]))
    var winners = List[Int]()
    for candidate in range(num_candidates):
        if trace[num_candidates + 2 + candidate]:
            winners.append(candidate)
    return KemenySolution(ranking^, winners^, RankingMultiplicity(UInt8(trace[num_candidates + 1])), optimum)


def compute_kemeny_trace_cpu[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](preferences: VoteMatrixView[StoredCountDataType, _]) raises -> List[KemenyTraceWord]:
    """Fills the cost table across the CPU cores and walks it into the packed trace."""
    require_kemeny_score_range[ArithmeticDataType, ArithmeticMode](preferences)
    var num_candidates = preferences.num_candidates
    var sums = KemenySums[ArithmeticDataType, ArithmeticMode](preferences)
    var costs = List[SIMD[ArithmeticDataType, 1]](length=1 << num_candidates, fill=0)
    var costs_ptr = costs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    compute_kemeny_costs_cpu(sums, costs_ptr)
    var trace = List[KemenyTraceWord](length=2 * num_candidates + 2, fill=0)
    kemeny_trace(sums.view(), costs_ptr, trace.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    # The untracked pointers do not keep the tables they address alive.
    _ = sums^
    _ = costs^
    return trace^


def compute_kemeny_ranking_cpu[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](preferences: VoteMatrixView[StoredCountDataType, _]) raises -> KemenySolution:
    """
    Determines the exact Kemeny-Young consensus ranking and its disagreement score.

    The ranking minimises the summed Kendall-tau distance to the ballots, so no ordering
    disagrees with the electorate less. This is the exact optimum rather than an
    approximation, at `O(n * 2^n)` time against `O(2^n)` memory.

    Args:
        preferences: Input preference matrix.

    Returns:
        The consensus ranking and the disagreement it achieves.
    """
    return kemeny_solution_from_trace[ArithmeticDataType](
        compute_kemeny_trace_cpu[ArithmeticDataType, ArithmeticMode](preferences)
    )


# endregion Ranking


# region GPU


def kemeny_layer_kernel[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    sums: Pointer[SIMD[ArithmeticDataType, 1], ImmUntrackedOrigin],
    binomials: Pointer[BinomialCount, ImmUntrackedOrigin],
    costs: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    num_candidates_arg: CandidateIndex,
    seated_arg: CandidateIndex,
    layer_states_arg: SubsetRank,
):
    """Fills every subset that seats `seated_arg` candidates, one thread to a colex rank."""
    var rank = SubsetRank(Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x))
    if rank >= layer_states_arg:
        return
    var num_candidates = Int(num_candidates_arg)
    var subset = kemeny_unrank_colex(binomials, num_candidates, Int(seated_arg), rank)
    costs[unsafe_offset=Int(subset)] = kemeny_best_cost(
        KemenySumsView[ArithmeticDataType, ArithmeticMode](sums, num_candidates), costs, subset
    )


def kemeny_trace_kernel[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](
    sums: Pointer[SIMD[ArithmeticDataType, 1], ImmUntrackedOrigin],
    costs: Pointer[SIMD[ArithmeticDataType, 1], MutUntrackedOrigin],
    trace: Pointer[KemenyTraceWord, MutUntrackedOrigin],
    num_candidates_arg: CandidateIndex,
):
    """Runs the shared trace on one device thread, so the exponential table never leaves the device."""
    kemeny_trace(KemenySumsView[ArithmeticDataType, ArithmeticMode](sums, Int(num_candidates_arg)), costs, trace)


def compute_kemeny_costs_gpu[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](ctx: DeviceContext, sums: KemenySums[ArithmeticDataType, ArithmeticMode]) raises -> Tuple[
    DeviceBuffer[ArithmeticDataType], DeviceBuffer[ArithmeticDataType]
]:
    """Builds the device cost table, returning it beside the device sums needed to trace it."""
    var num_candidates = sums.num_candidates
    var binomials = kemeny_binomials(num_candidates)
    var states = 1 << num_candidates
    var wanted_bytes = (states + len(sums.values)) * size_of[SIMD[ArithmeticDataType, 1]]() + len(binomials) * size_of[
        BinomialCount
    ]()

    # Unified memory oversubscribes rather than failing, so an unaffordable table is refused here.
    var free_bytes = Int(ctx.get_memory_info()[0])
    if wanted_bytes > free_bytes:
        raise Error(
            t"Kemeny over {num_candidates} candidates wants {wanted_bytes >> 20} MiB of device memory, "
            t"of which {free_bytes >> 20} MiB is free"
        )

    var device_sums = ctx.enqueue_create_buffer[ArithmeticDataType](len(sums.values))
    var device_binomials = ctx.enqueue_create_buffer[BinomialCount.dtype](len(binomials))
    var costs = ctx.enqueue_create_buffer[ArithmeticDataType](states)
    ctx.enqueue_copy(dst_buf=device_sums, src_ptr=sums.values.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=device_binomials, src_ptr=binomials.unsafe_ptr())
    costs.enqueue_fill(0)
    for seated in range(1, num_candidates + 1):
        var layer_states = kemeny_layer_states(binomials, num_candidates, seated)
        ctx.enqueue_function[kemeny_layer_kernel[ArithmeticDataType, ArithmeticMode]](
            device_sums.unsafe_ptr(),
            device_binomials.unsafe_ptr(),
            costs.unsafe_ptr(),
            CandidateIndex(num_candidates),
            CandidateIndex(seated),
            layer_states,
            grid_dim=(divide_round_up(Int(layer_states), GPU_BLOCK_SIZE), 1, 1),
            block_dim=(GPU_BLOCK_SIZE, 1, 1),
        )
    ctx.synchronize()
    _ = binomials^
    return (costs^, device_sums^)


def compute_kemeny_trace_gpu[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](preferences: VoteMatrixView[StoredCountDataType, _]) raises -> List[KemenyTraceWord]:
    """Fills the cost table on the GPU, one launch per population count, and traces it there."""
    require_kemeny_score_range[ArithmeticDataType, ArithmeticMode](preferences)
    var num_candidates = preferences.num_candidates
    var sums = KemenySums[ArithmeticDataType, ArithmeticMode](preferences)
    var ctx = DeviceContext()
    var (costs, device_sums) = compute_kemeny_costs_gpu(ctx, sums)
    var device_trace = ctx.enqueue_create_buffer[KemenyTraceWord.dtype](2 * num_candidates + 2)
    ctx.enqueue_function[kemeny_trace_kernel[ArithmeticDataType, ArithmeticMode]](
        device_sums.unsafe_ptr(),
        costs.unsafe_ptr(),
        device_trace.unsafe_ptr(),
        CandidateIndex(num_candidates),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    var trace = List[KemenyTraceWord](length=2 * num_candidates + 2, fill=0)
    ctx.enqueue_copy(dst_ptr=trace.unsafe_ptr(), src_buf=device_trace)
    ctx.synchronize()
    return trace^


def compute_kemeny_ranking_gpu[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](preferences: VoteMatrixView[StoredCountDataType, _]) raises -> KemenySolution:
    """Solves exact consensus on the GPU, one launch per subset population count."""
    return kemeny_solution_from_trace[ArithmeticDataType](
        compute_kemeny_trace_gpu[ArithmeticDataType, ArithmeticMode](preferences)
    )


# endregion GPU


def compute_kemeny_ranking[
    StoredCountDataType: DType
](
    preferences: VoteMatrixView[StoredCountDataType, _],
    *,
    backend: Backend,
    score_type: ScoreType,
) raises -> KemenySolution:
    """Dispatches exact ranking, raising when the optimum is unrepresentable."""

    def solve[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> KemenySolution:
        if backend == Backend.gpu:
            return compute_kemeny_ranking_gpu[ArithmeticDataType, ArithmeticMode](preferences)
        return compute_kemeny_ranking_cpu[ArithmeticDataType, ArithmeticMode](preferences)

    return with_score_type(resolve_score_type(preferences, score_type), solve)
