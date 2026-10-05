"""
Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

Schulze answers who wins; Kemeny-Young answers what the whole ordering should be, choosing the
ranking that disagrees with the fewest ballot pairs. That optimum is NP-hard to reach by search,
so this is the Held-Karp subset dynamic program instead: `O(n * 2^n)` time against `O(2^n)`
memory, exact rather than approximate, and practical to roughly two dozen candidates.
"""

from std.bit import count_trailing_zeros

from max.algorithm import parallelize
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from ballots import Arithmetic, Backend, ScoreType, VoteMatrix, add_counts

# region Kemeny


comptime KEMENY_MAX_CANDIDATES = 33
"""The widest field an exact table can address, bounded by memory rather than by time."""

comptime KEMENY_LAYER_CHUNK = 4096
"""Colex ranks one worker takes at a time, so a layer costs one closure call per chunk."""


def accumulate_subset_sums[
    arithmetic_dtype: DType, arithmetic: Arithmetic = Arithmetic.exact, stored_count_dtype: DType = DType.uint32
](
    mut table: List[SIMD[arithmetic_dtype, 1]],
    base: Int,
    states: Int,
    preferences: VoteMatrix[stored_count_dtype],
    candidate: Int,
    first_opponent: Int,
):
    """Folds one candidate's votes into every subset that contains each opponent."""
    var offset = 0
    var bit = 1
    while bit < states:
        var opponent = first_opponent + offset
        var votes = SIMD[arithmetic_dtype, 1](preferences[candidate, opponent]) if candidate != opponent else SIMD[
            arithmetic_dtype, 1
        ](0)
        for subset in range(bit, states):
            if subset & bit:
                table[base + subset] = add_counts[arithmetic](table[base + (subset ^ bit)], votes)
        offset += 1
        bit <<= 1


struct KemenySums[arithmetic_dtype: DType = DType.uint64, arithmetic: Arithmetic = Arithmetic.exact](Movable):
    """Votes each candidate loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
    """

    comptime arithmetic_t = SIMD[Self.arithmetic_dtype, 1]

    var low: List[Self.arithmetic_t]
    var high: List[Self.arithmetic_t]
    var low_bits: Int
    var low_states: Int
    var high_states: Int

    def __init__[stored_count_dtype: DType](out self, preferences: VoteMatrix[stored_count_dtype]):
        var num_candidates = preferences.num_candidates
        self.low_bits = num_candidates // 2
        self.low_states = 1 << self.low_bits
        self.high_states = 1 << (num_candidates - self.low_bits)
        self.low = List[Self.arithmetic_t]()
        self.low.resize(num_candidates * self.low_states, 0)
        self.high = List[Self.arithmetic_t]()
        self.high.resize(num_candidates * self.high_states, 0)

        for candidate in range(num_candidates):
            accumulate_subset_sums[Self.arithmetic_dtype, Self.arithmetic](
                self.low,
                candidate * self.low_states,
                self.low_states,
                preferences,
                candidate,
                0,
            )
            accumulate_subset_sums[Self.arithmetic_dtype, Self.arithmetic](
                self.high,
                candidate * self.high_states,
                self.high_states,
                preferences,
                candidate,
                self.low_bits,
            )

    def against(self, candidate: Int, subset: Int) -> Self.arithmetic_t:
        """Votes that preferred this candidate to every member of the subset."""
        return add_counts[Self.arithmetic](
            self.low[candidate * self.low_states + (subset & (self.low_states - 1))],
            self.high[candidate * self.high_states + (subset >> self.low_bits)],
        )


@fieldwise_init
struct KemenySolution(Movable):
    """An exact Kemeny-Young consensus ranking and the disagreement it achieves."""

    var ranking: List[Int]
    """The candidates in consensus order, best placed first."""
    var winners: List[Int]
    var unique: Bool
    var score: UInt64
    """Ballot pairs the ranking disagrees with, which no other ordering undercuts."""


def kemeny_binomials(num_candidates: Int) -> List[UInt32]:
    """Pascal's triangle, entry `upper * (num_candidates + 1) + lower` counting `C(upper, lower)`."""
    var stride = num_candidates + 1
    var table = List[UInt32]()
    table.resize(stride * stride, 0)
    # `C(33, 16)` is the widest entry a 32-bit slot has to hold.
    for upper in range(stride):
        table[upper * stride] = 1
        for lower in range(1, upper + 1):
            table[upper * stride + lower] = (
                table[(upper - 1) * stride + lower] + table[(upper - 1) * stride + lower - 1]
            )
    return table^


@always_inline
def kemeny_unrank_colex(binomials: List[UInt32], num_candidates: Int, seated: Int, rank: Int) -> Int:
    """The subset mask a colex rank names among those seating `seated` of the candidates."""
    var stride = num_candidates + 1
    var subset = 0
    var remaining = seated
    var position = rank
    var candidate = num_candidates
    while remaining != 0 and candidate != 0:
        candidate -= 1
        var below = Int(binomials[candidate * stride + remaining])
        if position < below:
            continue
        position -= below
        subset |= 1 << candidate
        remaining -= 1
    return subset


def kemeny_score_bound[stored_count_dtype: DType](preferences: VoteMatrix[stored_count_dtype]) raises -> UInt64:
    if preferences.num_candidates < 1 or preferences.num_candidates > KEMENY_MAX_CANDIDATES:
        raise Error("Kemeny is exact to " + String(KEMENY_MAX_CANDIDATES) + " candidates")
    var bound = UInt64(0)
    for row in range(preferences.num_candidates):
        for column in range(row + 1, preferences.num_candidates):
            bound = add_counts[Arithmetic.saturated](
                bound, UInt64(max(preferences[row, column], preferences[column, row]))
            )
    return bound


def resolve_score_type[
    stored_count_dtype: DType
](preferences: VoteMatrix[stored_count_dtype], requested_type: ScoreType = ScoreType.auto) raises -> ScoreType:
    """Resolves automatic arithmetic, reserving its maximum value for unreachable states."""
    if requested_type != ScoreType.auto:
        return requested_type
    var bound = kemeny_score_bound(preferences)
    if bound < UInt64(UInt32.MAX):
        return ScoreType.uint32
    return ScoreType.uint64 if bound < UInt64.MAX else ScoreType.saturated64


def require_kemeny_score_range[
    arithmetic_dtype: DType, arithmetic: Arithmetic, stored_count_dtype: DType
](preferences: VoteMatrix[stored_count_dtype]) raises:
    comptime assert (
        arithmetic_dtype == DType.uint16 or arithmetic_dtype == DType.uint32 or arithmetic_dtype == DType.uint64
    )
    comptime if arithmetic == Arithmetic.exact:
        if kemeny_score_bound(preferences) >= UInt64(SIMD[arithmetic_dtype, 1].MAX):
            raise Error("Kemeny score bound exceeds the selected arithmetic type")


def compute_kemeny_ranking_cpu[
    arithmetic_dtype: DType = DType.uint64,
    arithmetic: Arithmetic = Arithmetic.exact,
    stored_count_dtype: DType = DType.uint32,
](preferences: VoteMatrix[stored_count_dtype]) raises -> KemenySolution:
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
    var num_candidates = preferences.num_candidates
    if num_candidates < 1 or num_candidates > KEMENY_MAX_CANDIDATES:
        raise Error("Kemeny is exact to " + String(KEMENY_MAX_CANDIDATES) + " candidates")

    require_kemeny_score_range[arithmetic_dtype, arithmetic](preferences)
    var sums = KemenySums[arithmetic_dtype, arithmetic](preferences)
    var binomials = kemeny_binomials(num_candidates)
    var binomials_stride = num_candidates + 1
    var states = 1 << num_candidates

    # Entry `subset` is the least disagreement achievable seating those candidates in the
    # leading places, counting only the pairs inside it. Clearing a bit drops the population
    # count by exactly one, so one layer of subsets depends only on the layer below it.
    var costs = List[SIMD[arithmetic_dtype, 1]]()
    costs.resize(states, 0)
    var costs_data = costs.unsafe_ptr()

    for seated in range(1, num_candidates + 1):
        # Copied because a `parallelize` closure capturing the induction variable faults at -O1.
        var layer_seated = seated
        var layer_states = Int(binomials[num_candidates * binomials_stride + seated])
        var chunks = (layer_states + KEMENY_LAYER_CHUNK - 1) // KEMENY_LAYER_CHUNK

        def fill_layer_chunk(chunk: Int) {imm}:
            var first_rank = chunk * KEMENY_LAYER_CHUNK
            var last_rank = min(first_rank + KEMENY_LAYER_CHUNK, layer_states)
            var subset = kemeny_unrank_colex(binomials, num_candidates, layer_seated, first_rank)
            for _ in range(first_rank, last_rank):
                var best = SIMD[arithmetic_dtype, 1].MAX
                for candidate in range(num_candidates):
                    var bit = 1 << candidate
                    if not subset & bit:
                        continue
                    # Seating this candidate last within the subset costs the votes that preferred
                    # it to each of the others.
                    var rest = subset ^ bit
                    var score = add_counts[arithmetic](costs_data[unsafe_offset=rest], sums.against(candidate, rest))
                    if score < best:
                        best = score
                costs_data[unsafe_offset=subset] = best

                # Colex order over a layer is numeric order, so the next mask is one Gosper step on.
                var lowest = subset & -subset
                var ripple = subset + lowest
                subset = ripple | (((subset ^ ripple) >> 2) // lowest)

        parallelize(fill_layer_chunk, chunks)

    if costs[states - 1] == SIMD[arithmetic_dtype, 1].MAX:
        return KemenySolution(List[Int](), List[Int](), False, UInt64.MAX)
    var ranking = List[Int]()
    var winners = List[Int]()
    var unique = True
    var subset = states - 1
    for candidate in range(num_candidates):
        var score = costs[subset ^ (1 << candidate)]
        for rival in range(num_candidates):
            if rival != candidate:
                score = add_counts[arithmetic](score, SIMD[arithmetic_dtype, 1](preferences[rival, candidate]))
        if score == costs[subset]:
            winners.append(candidate)
    while subset:
        var seated_last = -1
        for candidate in range(num_candidates):
            var bit = 1 << candidate
            if not subset & bit:
                continue
            var rest = subset ^ bit
            if costs[subset] != add_counts[arithmetic](costs[rest], sums.against(candidate, rest)):
                continue
            if seated_last < 0:
                seated_last = candidate
            else:
                unique = False
        if seated_last < 0:
            raise Error("No candidate in the subset explains its cost, so the table is inconsistent")
        ranking.append(seated_last)
        subset ^= 1 << seated_last
    ranking.reverse()
    return KemenySolution(ranking^, winners^, unique, UInt64(costs[states - 1]))


# endregion Kemeny


@always_inline
def kemeny_votes_against_gpu[
    arithmetic_dtype: DType, arithmetic: Arithmetic = Arithmetic.exact
](
    sums: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    low_bits: Int,
    num_candidates: Int,
    candidate: Int,
    subset: Int,
) -> SIMD[arithmetic_dtype, 1]:
    var low_states = 1 << low_bits
    var high_states = 1 << (num_candidates - low_bits)
    return add_counts[arithmetic](
        sums[unsafe_offset=candidate * low_states + (subset & (low_states - 1))],
        sums[unsafe_offset=num_candidates * low_states + candidate * high_states + (subset >> low_bits)],
    )


def kemeny_layer_gpu[
    arithmetic_dtype: DType, arithmetic: Arithmetic = Arithmetic.exact
](
    sums: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    binomials: Pointer[UInt32, MutUntrackedOrigin],
    costs: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    num_candidates_arg: Int32,
    seated_arg: Int32,
    layer_states_arg: Int32,
):
    var num_candidates = Int(num_candidates_arg)
    var seated = Int(seated_arg)
    var layer_states = Int(layer_states_arg)
    var rank = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if rank >= layer_states:
        return
    var subset = 0
    var remaining = seated
    var candidate = num_candidates
    while remaining != 0 and candidate != 0:
        candidate -= 1
        var below = Int(binomials[unsafe_offset=candidate * (num_candidates + 1) + remaining])
        if rank >= below:
            rank -= below
            subset |= 1 << candidate
            remaining -= 1
    var best = SIMD[arithmetic_dtype, 1].MAX
    var members = subset
    while members:
        var bit = members & -members
        var index = Int(count_trailing_zeros(bit))
        var rest = subset ^ bit
        best = min(
            best,
            add_counts[arithmetic](
                costs[unsafe_offset=rest],
                kemeny_votes_against_gpu[arithmetic_dtype, arithmetic](
                    sums, num_candidates // 2, num_candidates, index, rest
                ),
            ),
        )
        members ^= bit
    costs[unsafe_offset=subset] = best


def kemeny_trace_gpu[
    arithmetic_dtype: DType, arithmetic: Arithmetic = Arithmetic.exact
](
    sums: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    costs: Pointer[SIMD[arithmetic_dtype, 1], MutUntrackedOrigin],
    result: Pointer[UInt64, MutUntrackedOrigin],
    num_candidates_arg: Int32,
):
    var n = Int(num_candidates_arg)
    var full = (1 << n) - 1
    var optimum = costs[unsafe_offset=full]
    result[unsafe_offset=n] = UInt64(optimum)
    if optimum == SIMD[arithmetic_dtype, 1].MAX:
        return
    result[unsafe_offset=n + 1] = 1
    for candidate in range(n):
        var score = costs[unsafe_offset=full ^ (1 << candidate)]
        for rival in range(n):
            if rival != candidate:
                score = add_counts[arithmetic](
                    score,
                    kemeny_votes_against_gpu[arithmetic_dtype, arithmetic](sums, n // 2, n, rival, 1 << candidate),
                )
        result[unsafe_offset=n + 2 + candidate] = UInt64(score == optimum)
    var subset = full
    for place in range(n - 1, -1, -1):
        var chosen = -1
        for candidate in range(n):
            var bit = 1 << candidate
            if not subset & bit:
                continue
            var rest = subset ^ bit
            var score = add_counts[arithmetic](
                costs[unsafe_offset=rest],
                kemeny_votes_against_gpu[arithmetic_dtype, arithmetic](sums, n // 2, n, candidate, rest),
            )
            if score == costs[unsafe_offset=subset]:
                if chosen < 0:
                    chosen = candidate
                else:
                    result[unsafe_offset=n + 1] = 0
        result[unsafe_offset=place] = UInt64(chosen)
        if chosen < 0:
            return
        subset ^= 1 << chosen


def compute_kemeny_ranking_gpu[
    arithmetic_dtype: DType = DType.uint64,
    arithmetic: Arithmetic = Arithmetic.exact,
    stored_count_dtype: DType = DType.uint32,
](preferences: VoteMatrix[stored_count_dtype]) raises -> KemenySolution:
    """Solves exact consensus on the GPU, one launch per subset population count."""
    var n = preferences.num_candidates
    if n < 1 or n > KEMENY_MAX_CANDIDATES:
        raise Error("Kemeny supports 1 to " + String(KEMENY_MAX_CANDIDATES) + " candidates")
    require_kemeny_score_range[arithmetic_dtype, arithmetic](preferences)
    var sums = KemenySums[arithmetic_dtype, arithmetic](preferences)
    var binomials = kemeny_binomials(n)
    var ctx = DeviceContext()
    var host_sums = ctx.enqueue_create_host_buffer[arithmetic_dtype](len(sums.low) + len(sums.high))
    var device_sums = ctx.enqueue_create_buffer[arithmetic_dtype](len(sums.low) + len(sums.high))
    var host_binomials = ctx.enqueue_create_host_buffer[DType.uint32](len(binomials))
    var device_binomials = ctx.enqueue_create_buffer[DType.uint32](len(binomials))
    var costs = ctx.enqueue_create_buffer[arithmetic_dtype](1 << n)
    var host_result = ctx.enqueue_create_host_buffer[DType.uint64](2 * n + 2)
    var device_result = ctx.enqueue_create_buffer[DType.uint64](2 * n + 2)
    ctx.synchronize()
    for i in range(len(sums.low)):
        host_sums[i] = sums.low[i]
    for i in range(len(sums.high)):
        host_sums[len(sums.low) + i] = sums.high[i]
    for i in range(len(binomials)):
        host_binomials[i] = binomials[i]
    host_sums.enqueue_copy_to(device_sums)
    host_binomials.enqueue_copy_to(device_binomials)
    costs.enqueue_fill(0)
    for seated in range(1, n + 1):
        var layer_states = Int(binomials[n * (n + 1) + seated])
        ctx.enqueue_function[kemeny_layer_gpu[arithmetic_dtype, arithmetic]](
            device_sums.unsafe_ptr(),
            device_binomials.unsafe_ptr(),
            costs.unsafe_ptr(),
            Int32(n),
            Int32(seated),
            Int32(layer_states),
            grid_dim=((layer_states + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.enqueue_function[kemeny_trace_gpu[arithmetic_dtype, arithmetic]](
        device_sums.unsafe_ptr(),
        costs.unsafe_ptr(),
        device_result.unsafe_ptr(),
        Int32(n),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    device_result.enqueue_copy_to(host_result)
    ctx.synchronize()
    var result = host_result.as_span()
    var optimum = result[n]
    if optimum == UInt64(SIMD[arithmetic_dtype, 1].MAX):
        return KemenySolution(List[Int](), List[Int](), False, UInt64.MAX)
    var ranking = List[Int]()
    var winners = List[Int]()
    for candidate in range(n):
        if result[n + 2 + candidate]:
            winners.append(candidate)
    for i in range(n):
        var candidate = result[i]
        if candidate >= UInt64(n):
            raise Error("No candidate in the subset explains its cost, so the table is inconsistent")
        ranking.append(Int(candidate))
    return KemenySolution(ranking^, winners^, result[n + 1] != 0, optimum)


def compute_kemeny_ranking[
    stored_count_dtype: DType
](
    preferences: VoteMatrix[stored_count_dtype],
    *,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) raises -> KemenySolution:
    """Dispatches exact ranking; a UInt64.MAX score reports an unrepresentable optimum."""
    var resolved = resolve_score_type(preferences, score_type)
    if resolved == ScoreType.uint16:
        return compute_kemeny_ranking_gpu[DType.uint16](
            preferences
        ) if backend == Backend.gpu else compute_kemeny_ranking_cpu[DType.uint16](preferences)
    if resolved == ScoreType.uint32:
        return compute_kemeny_ranking_gpu[DType.uint32](
            preferences
        ) if backend == Backend.gpu else compute_kemeny_ranking_cpu[DType.uint32](preferences)
    if resolved == ScoreType.uint64:
        return compute_kemeny_ranking_gpu[DType.uint64](
            preferences
        ) if backend == Backend.gpu else compute_kemeny_ranking_cpu[DType.uint64](preferences)
    if resolved == ScoreType.saturated64:
        return compute_kemeny_ranking_gpu[DType.uint64, Arithmetic.saturated](
            preferences
        ) if backend == Backend.gpu else compute_kemeny_ranking_cpu[DType.uint64, Arithmetic.saturated](preferences)
    raise Error("Invalid score type")
