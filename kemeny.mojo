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

from ballots import Backend, PreferenceMatrix, ScoreType

# region Kemeny


comptime KEMENY_MAX_CANDIDATES = 33
"""The widest field an exact table can address, bounded by memory rather than by time."""

comptime KEMENY_LAYER_CHUNK = 4096
"""Colex ranks one worker takes at a time, so a layer costs one closure call per chunk."""


def accumulate_subset_sums[
    score_dtype: DType
](
    mut table: List[SIMD[score_dtype, 1]],
    base: Int,
    states: Int,
    preferences: PreferenceMatrix,
    candidate: Int,
    first_opponent: Int,
):
    """Folds one candidate's votes into every subset that contains each opponent."""
    var offset = 0
    var bit = 1
    while bit < states:
        var opponent = first_opponent + offset
        var votes = SIMD[score_dtype, 1](preferences[candidate, opponent]) if candidate != opponent else SIMD[
            score_dtype, 1
        ](0)
        for subset in range(bit, states):
            if subset & bit:
                table[base + subset] = table[base + (subset ^ bit)] + votes
        offset += 1
        bit <<= 1


struct KemenySums[score_dtype: DType = DType.uint64](Movable):
    """Votes each candidate loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
    """

    var low: List[SIMD[Self.score_dtype, 1]]
    var high: List[SIMD[Self.score_dtype, 1]]
    var low_bits: Int
    var low_states: Int
    var high_states: Int

    def __init__(out self, preferences: PreferenceMatrix):
        var num_candidates = preferences.num_candidates
        self.low_bits = num_candidates // 2
        self.low_states = 1 << self.low_bits
        self.high_states = 1 << (num_candidates - self.low_bits)
        self.low = List[SIMD[Self.score_dtype, 1]]()
        self.low.resize(num_candidates * self.low_states, 0)
        self.high = List[SIMD[Self.score_dtype, 1]]()
        self.high.resize(num_candidates * self.high_states, 0)

        for candidate in range(num_candidates):
            accumulate_subset_sums(
                self.low,
                candidate * self.low_states,
                self.low_states,
                preferences,
                candidate,
                0,
            )
            accumulate_subset_sums(
                self.high,
                candidate * self.high_states,
                self.high_states,
                preferences,
                candidate,
                self.low_bits,
            )

    def against(self, candidate: Int, subset: Int) -> SIMD[Self.score_dtype, 1]:
        """Votes that preferred this candidate to every member of the subset."""
        return (
            self.low[candidate * self.low_states + (subset & (self.low_states - 1))]
            + self.high[candidate * self.high_states + (subset >> self.low_bits)]
        )


@fieldwise_init
struct KemenySolution(Movable):
    """An exact Kemeny-Young consensus ranking and the disagreement it achieves."""

    var ranking: List[Int]
    """The candidates in consensus order, best placed first."""
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


def kemeny_score_bound(preferences: PreferenceMatrix) raises -> UInt64:
    if preferences.num_candidates < 1 or preferences.num_candidates > KEMENY_MAX_CANDIDATES:
        raise Error("Kemeny is exact to " + String(KEMENY_MAX_CANDIDATES) + " candidates")
    var bound = UInt64(0)
    for row in range(preferences.num_candidates):
        for column in range(row + 1, preferences.num_candidates):
            bound += UInt64(max(preferences[row, column], preferences[column, row]))
    return bound


def resolve_score_type(preferences: PreferenceMatrix, requested_type: ScoreType = ScoreType.auto) raises -> ScoreType:
    """Resolves automatic arithmetic, reserving its maximum value for unreachable states."""
    if requested_type != ScoreType.auto:
        return requested_type
    return ScoreType.uint32 if kemeny_score_bound(preferences) < UInt64(UInt32.MAX) else ScoreType.uint64


def require_kemeny_score_range[score_dtype: DType](preferences: PreferenceMatrix) raises:
    comptime assert score_dtype == DType.uint16 or score_dtype == DType.uint32 or score_dtype == DType.uint64
    if kemeny_score_bound(preferences) >= UInt64(SIMD[score_dtype, 1].MAX):
        raise Error("Kemeny score bound exceeds the selected arithmetic type")


def compute_kemeny_ranking_cpu[
    score_dtype: DType = DType.uint64
](preferences: PreferenceMatrix) raises -> KemenySolution:
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

    require_kemeny_score_range[score_dtype](preferences)
    var sums = KemenySums[score_dtype](preferences)
    var binomials = kemeny_binomials(num_candidates)
    var binomials_stride = num_candidates + 1
    var states = 1 << num_candidates

    # Entry `subset` is the least disagreement achievable seating those candidates in the
    # leading places, counting only the pairs inside it. Clearing a bit drops the population
    # count by exactly one, so one layer of subsets depends only on the layer below it.
    var costs = List[SIMD[score_dtype, 1]]()
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
                var best = SIMD[score_dtype, 1].MAX
                for candidate in range(num_candidates):
                    var bit = 1 << candidate
                    if not subset & bit:
                        continue
                    # Seating this candidate last within the subset costs the votes that preferred
                    # it to each of the others.
                    var rest = subset ^ bit
                    var score = costs_data[unsafe_offset=rest] + sums.against(candidate, rest)
                    if score < best:
                        best = score
                costs_data[unsafe_offset=subset] = best

                # Colex order over a layer is numeric order, so the next mask is one Gosper step on.
                var lowest = subset & -subset
                var ripple = subset + lowest
                subset = ripple | (((subset ^ ripple) >> 2) // lowest)

        parallelize(fill_layer_chunk, chunks)

    # Walk the choices back out, which recovers the ranking from its last place upwards.
    var ranking = List[Int]()
    var subset = states - 1
    while subset:
        var seated_last = -1
        for candidate in range(num_candidates):
            var bit = 1 << candidate
            if not subset & bit:
                continue
            var rest = subset ^ bit
            if costs[subset] != costs[rest] + sums.against(candidate, rest):
                continue
            seated_last = candidate
            break
        if seated_last < 0:
            raise Error("No candidate in the subset explains its cost, so the table is inconsistent")
        ranking.append(seated_last)
        subset ^= 1 << seated_last

    ranking.reverse()
    return KemenySolution(ranking^, UInt64(costs[states - 1]))


# endregion Kemeny


@always_inline
def kemeny_votes_against_gpu[
    score_dtype: DType
](
    sums: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    low_bits: Int,
    num_candidates: Int,
    candidate: Int,
    subset: Int,
) -> SIMD[score_dtype, 1]:
    var low_states = 1 << low_bits
    var high_states = 1 << (num_candidates - low_bits)
    return (
        sums[unsafe_offset=candidate * low_states + (subset & (low_states - 1))]
        + sums[unsafe_offset=num_candidates * low_states + candidate * high_states + (subset >> low_bits)]
    )


def kemeny_layer_gpu[
    score_dtype: DType
](
    sums: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    binomials: Pointer[UInt32, MutUntrackedOrigin],
    costs: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
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
    var best = SIMD[score_dtype, 1].MAX
    var members = subset
    while members:
        var bit = members & -members
        var index = Int(count_trailing_zeros(bit))
        var rest = subset ^ bit
        best = min(
            best,
            costs[unsafe_offset=rest]
            + kemeny_votes_against_gpu(sums, num_candidates // 2, num_candidates, index, rest),
        )
        members ^= bit
    costs[unsafe_offset=subset] = best


def kemeny_trace_gpu[
    score_dtype: DType
](
    sums: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    costs: Pointer[SIMD[score_dtype, 1], MutUntrackedOrigin],
    result: Pointer[UInt64, MutUntrackedOrigin],
    num_candidates_arg: Int32,
):
    var num_candidates = Int(num_candidates_arg)
    var subset = (1 << num_candidates) - 1
    result[unsafe_offset=num_candidates] = UInt64(costs[unsafe_offset=subset])
    for place in range(num_candidates - 1, -1, -1):
        result[unsafe_offset=place] = UInt64.MAX
        for candidate in range(num_candidates):
            var bit = 1 << candidate
            if not subset & bit:
                continue
            var rest = subset ^ bit
            if costs[unsafe_offset=subset] == costs[unsafe_offset=rest] + kemeny_votes_against_gpu(
                sums, num_candidates // 2, num_candidates, candidate, rest
            ):
                result[unsafe_offset=place] = UInt64(candidate)
                subset = rest
                break


def compute_kemeny_ranking_gpu[
    score_dtype: DType = DType.uint64
](preferences: PreferenceMatrix) raises -> KemenySolution:
    """Solves exact consensus on the GPU, one launch per subset population count."""
    var n = preferences.num_candidates
    if n < 1 or n > KEMENY_MAX_CANDIDATES:
        raise Error("Kemeny supports 1 to " + String(KEMENY_MAX_CANDIDATES) + " candidates")
    require_kemeny_score_range[score_dtype](preferences)
    var sums = KemenySums[score_dtype](preferences)
    var binomials = kemeny_binomials(n)
    var ctx = DeviceContext()
    var host_sums = ctx.enqueue_create_host_buffer[score_dtype](len(sums.low) + len(sums.high))
    var device_sums = ctx.enqueue_create_buffer[score_dtype](len(sums.low) + len(sums.high))
    var host_binomials = ctx.enqueue_create_host_buffer[DType.uint32](len(binomials))
    var device_binomials = ctx.enqueue_create_buffer[DType.uint32](len(binomials))
    var costs = ctx.enqueue_create_buffer[score_dtype](1 << n)
    var host_result = ctx.enqueue_create_host_buffer[DType.uint64](n + 1)
    var device_result = ctx.enqueue_create_buffer[DType.uint64](n + 1)
    ctx.synchronize()
    for i in range(len(sums.low)):
        host_sums.unsafe_ptr()[unsafe_offset=i] = sums.low[i]
    for i in range(len(sums.high)):
        host_sums.unsafe_ptr()[unsafe_offset=len(sums.low) + i] = sums.high[i]
    for i in range(len(binomials)):
        host_binomials.unsafe_ptr()[unsafe_offset=i] = binomials[i]
    host_sums.enqueue_copy_to(device_sums)
    host_binomials.enqueue_copy_to(device_binomials)
    costs.enqueue_fill(0)
    for seated in range(1, n + 1):
        var layer_states = Int(binomials[n * (n + 1) + seated])
        ctx.enqueue_function[kemeny_layer_gpu[score_dtype]](
            device_sums.unsafe_ptr(),
            device_binomials.unsafe_ptr(),
            costs.unsafe_ptr(),
            Int32(n),
            Int32(seated),
            Int32(layer_states),
            grid_dim=((layer_states + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.enqueue_function[kemeny_trace_gpu[score_dtype]](
        device_sums.unsafe_ptr(),
        costs.unsafe_ptr(),
        device_result.unsafe_ptr(),
        Int32(n),
        grid_dim=(1, 1, 1),
        block_dim=(1, 1, 1),
    )
    device_result.enqueue_copy_to(host_result)
    ctx.synchronize()
    var ranking = List[Int]()
    for i in range(n):
        var candidate = host_result.unsafe_ptr()[unsafe_offset=i]
        if candidate >= UInt64(n):
            raise Error("No candidate in the subset explains its cost, so the table is inconsistent")
        ranking.append(Int(candidate))
    return KemenySolution(ranking^, host_result.unsafe_ptr()[unsafe_offset=n])


def compute_kemeny_ranking(
    preferences: PreferenceMatrix, *, backend: Backend = Backend.cpu, score_type: ScoreType = ScoreType.auto
) raises -> KemenySolution:
    """Dispatches exact ranking to the selected device and a safe compiled score width."""
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
    raise Error("Invalid score type")
