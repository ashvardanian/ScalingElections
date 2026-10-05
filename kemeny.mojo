"""
Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

Schulze answers who wins; Kemeny-Young answers what the whole ordering should be, choosing the
ranking that disagrees with the fewest ballot pairs. That optimum is NP-hard to reach by search,
so this is the Held-Karp subset dynamic program instead: `O(n * 2^n)` time against `O(2^n)`
memory, exact rather than approximate, and practical to roughly two dozen candidates.
"""

from max.algorithm import parallelize

from ballots import PreferenceMatrix

# region Kemeny


comptime KEMENY_MAX_CANDIDATES = 33
"""The widest field an exact table can address, bounded by memory rather than by time."""

comptime KemenyScore = UInt64
"""Pairwise disagreements summed, bounded by `voters * n * (n - 1) / 2`, which 64 bits hold."""

comptime KEMENY_LAYER_CHUNK = 4096
"""Colex ranks one worker takes at a time, so a layer costs one closure call per chunk."""


def accumulate_subset_sums(
    mut table: List[KemenyScore],
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
        var votes = KemenyScore(preferences[candidate, first_opponent + offset])
        for subset in range(bit, states):
            if subset & bit:
                table[base + subset] = table[base + (subset ^ bit)] + votes
        offset += 1
        bit <<= 1


struct KemenySums(Movable):
    """Votes each candidate loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.
    """

    var low: List[KemenyScore]
    var high: List[KemenyScore]
    var low_bits: Int
    var low_states: Int
    var high_states: Int

    def __init__(out self, preferences: PreferenceMatrix):
        var num_candidates = preferences.num_candidates
        self.low_bits = num_candidates // 2
        self.low_states = 1 << self.low_bits
        self.high_states = 1 << (num_candidates - self.low_bits)
        self.low = List[KemenyScore]()
        self.low.resize(num_candidates * self.low_states, 0)
        self.high = List[KemenyScore]()
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

    def against(self, candidate: Int, subset: Int) -> KemenyScore:
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
    var score: Int
    """Ballot pairs the ranking disagrees with, which no other ordering undercuts."""


def binomial_table(num_candidates: Int) -> List[UInt32]:
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
def subset_at_colex_rank(binomials: List[UInt32], num_candidates: Int, seated: Int, rank: Int) -> Int:
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


def kemeny_ranking(preferences: PreferenceMatrix) raises -> KemenySolution:
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
        raise Error(
            "Kemeny is exact to " + String(KEMENY_MAX_CANDIDATES) + " candidates, reaching 64 GiB at that width"
        )

    var sums = KemenySums(preferences)
    var binomials = binomial_table(num_candidates)
    var binomials_stride = num_candidates + 1
    var states = 1 << num_candidates

    # Entry `subset` is the least disagreement achievable seating those candidates in the
    # leading places, counting only the pairs inside it. Clearing a bit drops the population
    # count by exactly one, so one layer of subsets depends only on the layer below it.
    var costs = List[KemenyScore]()
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
            var subset = subset_at_colex_rank(binomials, num_candidates, layer_seated, first_rank)
            for _ in range(first_rank, last_rank):
                var best = KemenyScore.MAX
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
    return KemenySolution(ranking^, Int(costs[states - 1]))


# endregion Kemeny
