"""Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

The optimum is found by subset dynamic programming, so the cost is exponential in the
number of candidates rather than approximate.
"""

import os
import sys
from collections.abc import Generator
from enum import StrEnum
from typing import NamedTuple

import numpy as np
from numba import get_num_threads, njit, prange
from numpy.typing import NDArray

from ballots import Arithmetic, ScoreType, add_scores, saturated_sum, saturating_add


class RankingMultiplicity(StrEnum):
    """Whether one or several complete orderings achieve the minimum disagreement."""

    unique = "unique"
    """Exactly one complete ordering is optimal."""
    multiple = "multiple"
    """Several complete orderings are optimal, possibly with the same winner."""


class KemenyResult(NamedTuple):
    """An optimal Kemeny-Young ordering, its disagreement score, and its tie metadata."""

    ranking: list[int]
    """One optimal ordering, best candidate first."""
    score: int
    """Minimum total weight of pairwise preferences contradicted by the ordering."""
    winners: list[int]
    """All candidates that can rank first in an optimal ordering."""
    multiplicity: RankingMultiplicity
    """Whether the complete optimal ordering is unique."""


KEMENY_MAX_CANDIDATES = 33
"""The widest field an exact table can address, bounded by memory rather than by time."""

KEMENY_HOST_HEADROOM = 1 << 30
"""Host memory the tables must leave behind for the rest of the machine."""


# region Subset Sums


@njit(cache=True)
def kemeny_subset_sums(
    preferences: NDArray[np.integer], low_bits: int, arithmetic: Arithmetic = Arithmetic.exact
) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
    """
    Tabulates, for each candidate, the votes it loses to every subset of the others.

    One table would be `n * 2^n` wide. Splitting the subset into a low and a high half makes
    two of `n * 2^(n/2)`, small enough to stay in cache while the score table streams past.

    Space complexity: O(n * 2^(n/2)), where n is the number of candidates.
    Time complexity: O(n * 2^(n/2)), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    high_bits = num_candidates - low_bits
    low = np.zeros((num_candidates, 1 << low_bits), dtype=preferences.dtype)
    high = np.zeros((num_candidates, 1 << high_bits), dtype=preferences.dtype)
    for candidate in range(num_candidates):
        for offset in range(low_bits):
            bit = 1 << offset
            votes = preferences[candidate, offset]
            for subset in range(bit, 1 << low_bits):
                if subset & bit:
                    low[candidate, subset] = add_scores(low[candidate, subset ^ bit], votes, arithmetic)
        for offset in range(high_bits):
            bit = 1 << offset
            votes = preferences[candidate, low_bits + offset]
            for subset in range(bit, 1 << high_bits):
                if subset & bit:
                    high[candidate, subset] = add_scores(high[candidate, subset ^ bit], votes, arithmetic)
    return low, high


@njit(cache=True)
def kemeny_votes_against(
    low: NDArray[np.integer],
    high: NDArray[np.integer],
    low_bits: int,
    candidate: int,
    subset: int,
    arithmetic: Arithmetic = Arithmetic.exact,
) -> np.uint64:
    """Votes that preferred this candidate to every member of the subset."""
    return add_scores(low[candidate, subset & ((1 << low_bits) - 1)], high[candidate, subset >> low_bits], arithmetic)


# endregion Subset Sums


# region Cost Table


@njit(cache=True)
def kemeny_binomials(num_candidates: int) -> NDArray[np.integer]:
    """Pascal's triangle, whose last row counts the subsets seating each number of candidates."""
    binomials = np.zeros((num_candidates + 1, num_candidates + 1), dtype=np.int64)
    for upper in range(num_candidates + 1):
        binomials[upper, 0] = 1
        for lower in range(1, upper + 1):
            binomials[upper, lower] = binomials[upper - 1, lower] + binomials[upper - 1, lower - 1]
    return binomials


@njit(cache=True)
def kemeny_unrank_colex(binomials: NDArray[np.integer], num_candidates: int, seated: int, rank: int) -> int:
    """The subset a colex rank names among those seating `seated` of `num_candidates` candidates."""
    subset = 0
    remaining = seated
    candidate = num_candidates
    while remaining != 0 and candidate != 0:
        candidate -= 1
        below = binomials[candidate, remaining]
        if rank < below:
            continue
        rank -= below
        subset |= 1 << candidate
        remaining -= 1
    return subset


@njit(cache=True)
def kemeny_next_subset(subset: int) -> int:
    """The next mask of the same population count, which is the next subset in colex order."""
    lowest = subset & -subset
    rippled = subset + lowest
    return rippled | (((subset ^ rippled) >> 2) // lowest)


@njit(parallel=True)
def kemeny_costs(
    preferences: NDArray[np.integer],
    low_bits: int,
    low: NDArray[np.integer],
    high: NDArray[np.integer],
    score_type: type[np.unsignedinteger],
    arithmetic: Arithmetic = Arithmetic.exact,
) -> NDArray[np.integer]:
    """
    Computes the least disagreement achievable for every subset of candidates.

    Entry `subset` is the score of the best ordering of those candidates in the leading seats,
    counting only the pairs inside it. Clearing a bit drops the population count by exactly one,
    so one population count depends only on the one below and its subsets all fill at once.

    Space complexity: O(2^n), where n is the number of candidates.
    Time complexity: O(n * 2^n), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    binomials = kemeny_binomials(num_candidates)
    states = 1 << num_candidates

    costs = np.empty(states, dtype=score_type)
    costs[0] = 0
    unreachable = score_type(np.iinfo(score_type).max)
    for seated in range(1, num_candidates + 1):
        layer_states = binomials[num_candidates, seated]
        # Unranking walks the whole field, so a chunk pays it once and steps through the rest.
        chunks = min(get_num_threads() * 8, layer_states)
        for chunk in prange(chunks):
            first = chunk * layer_states // chunks
            last = (chunk + 1) * layer_states // chunks
            subset = kemeny_unrank_colex(binomials, num_candidates, seated, first)
            for _ in range(first, last):
                best = unreachable
                for candidate in range(num_candidates):
                    bit = 1 << candidate
                    if not subset & bit:
                        continue
                    # Seating this candidate last within the subset costs the votes that preferred
                    # it to each of the others.
                    rest = subset ^ bit
                    score = score_type(
                        add_scores(
                            costs[rest],
                            kemeny_votes_against(low, high, low_bits, candidate, rest, arithmetic),
                            arithmetic,
                        )
                    )
                    if score < best:
                        best = score
                costs[subset] = best
                subset = kemeny_next_subset(subset)
    return costs


# endregion Cost Table


# region Ranking


def kemeny_table_bytes(num_candidates: int, score_type: ScoreType = ScoreType.uint64) -> int:
    """Bytes the cost table and both subset-sum tables occupy at this width."""
    low_bits = num_candidates // 2
    sums_states = num_candidates * ((1 << low_bits) + (1 << (num_candidates - low_bits)))
    return ((1 << num_candidates) + sums_states) * np.dtype(f"uint{score_type.bits}").itemsize


def available_host_bytes() -> int:
    """Physical memory the host will still hand out, or every byte it could name when it will not say."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return sys.maxsize


def resolve_score_type(preferences: NDArray[np.integer], score_type: ScoreType = ScoreType.auto) -> ScoreType:
    """Select a width that represents every intermediate score, reserving its maximum as a sentinel."""
    num_candidates = preferences.shape[0]
    if num_candidates < 1 or num_candidates > KEMENY_MAX_CANDIDATES:
        raise ValueError(f"Kemeny is exact to {KEMENY_MAX_CANDIDATES} candidates")

    score_type = ScoreType(score_type)
    bound = int(saturated_sum(np.maximum(preferences, preferences.T)[np.triu_indices(num_candidates, 1)]))
    if score_type is ScoreType.auto:
        if bound < np.iinfo(np.uint32).max:
            score_type = ScoreType.uint32
        else:
            score_type = ScoreType.uint64 if bound < np.iinfo(np.uint64).max else ScoreType.saturated64
    if score_type is not ScoreType.saturated64 and bound >= (1 << score_type.bits) - 1:
        raise OverflowError("Kemeny score bound exceeds the selected arithmetic type")

    return score_type


def _kemeny_tables(
    preferences: NDArray[np.integer], score_type: ScoreType
) -> tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.integer]]:
    """Build arithmetic-width subset sums and costs within the host memory budget."""
    num_candidates = preferences.shape[0]
    score_dtype = np.dtype(f"uint{score_type.bits}").type

    # NumPy reports an unaffordable table as a bare `MemoryError`, so the size is refused by name here.
    wanted_bytes = kemeny_table_bytes(num_candidates, score_type)
    free_bytes = available_host_bytes()
    if wanted_bytes + KEMENY_HOST_HEADROOM > free_bytes:
        raise MemoryError(
            f"Kemeny over {num_candidates} candidates wants {wanted_bytes >> 20} MiB of host memory, "
            f"of which {free_bytes >> 20} MiB is free"
        )

    counts = preferences.astype(score_dtype)
    np.fill_diagonal(counts, 0)
    low_bits = num_candidates // 2
    arithmetic = Arithmetic.saturated if score_type is ScoreType.saturated64 else Arithmetic.exact
    low, high = kemeny_subset_sums(counts, low_bits, arithmetic)
    costs = kemeny_costs(counts, low_bits, low, high, score_dtype, arithmetic)

    return costs, low, high


def compute_kemeny_costs(
    preferences: NDArray[np.integer], *, score_type: ScoreType = ScoreType.auto
) -> NDArray[np.integer]:
    """Retain one cost per subset for lazy enumeration of every optimum."""
    costs, _, _ = _kemeny_tables(preferences, resolve_score_type(preferences, score_type))
    return costs


def compute_kemeny_ranking(preferences: NDArray[np.integer], *, score_type: ScoreType = ScoreType.auto) -> KemenyResult:
    """
    Determines the exact Kemeny-Young consensus ranking and its disagreement score.

    The ranking minimises the summed Kendall-tau distance to the ballots, so no ordering
    disagrees with the electorate less. This is the exact optimum rather than an
    approximation, at O(n * 2^n) time against O(2^n) memory.

    Space complexity: O(2^n), where n is the number of candidates.
    Time complexity: O(n * 2^n), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    score_type = resolve_score_type(preferences, score_type)
    costs, low, high = _kemeny_tables(preferences, score_type)
    arithmetic = Arithmetic.saturated if score_type is ScoreType.saturated64 else Arithmetic.exact
    low_bits = num_candidates // 2

    full = (1 << num_candidates) - 1
    optimum = int(costs[full])
    if optimum == np.iinfo(np.uint64).max:
        raise OverflowError("The optimal Kemeny score exceeds the representable range")
    winners = [
        candidate
        for candidate in range(num_candidates)
        if int(costs[full ^ (1 << candidate)])
        + sum(int(preferences[other, candidate]) for other in range(num_candidates) if other != candidate)
        == optimum
    ]
    ranking = []
    multiplicity = RankingMultiplicity.unique
    subset = full
    while subset:
        choices = []
        for candidate in range(num_candidates):
            bit = 1 << candidate
            if not subset & bit:
                continue
            rest = subset ^ bit
            against = kemeny_votes_against(low, high, low_bits, candidate, rest, arithmetic)
            score = saturating_add(np.uint64(costs[rest]), np.uint64(against))
            if int(costs[subset]) == int(score):
                choices.append(candidate)
        if not choices:
            raise RuntimeError("The Kemeny cost table disagrees with its own sums")
        if len(choices) > 1:
            multiplicity = RankingMultiplicity.multiple
        candidate = choices[0]
        ranking.append(candidate)
        subset ^= 1 << candidate
    ranking.reverse()
    return KemenyResult(ranking, optimum, winners, multiplicity)


def _enumerate_kemeny_rankings(
    preferences: NDArray[np.integer], costs: NDArray[np.integer], subset: int, prefix: tuple[int, ...]
) -> Generator[list[int], None, None]:
    """Traverse optimal transitions in the retained subset cost table."""
    if not subset:
        yield list(prefix)
        return
    candidates = [candidate for candidate in range(len(preferences)) if subset & (1 << candidate)]
    for candidate in candidates:
        remaining = subset ^ (1 << candidate)
        score = int(costs[remaining]) + sum(
            int(preferences[rival, candidate]) for rival in candidates if rival != candidate
        )
        if score == int(costs[subset]):
            yield from _enumerate_kemeny_rankings(preferences, costs, remaining, (*prefix, candidate))


# endregion Ranking
