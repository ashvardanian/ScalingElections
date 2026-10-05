"""Exact Kemeny-Young consensus ranking over a pairwise preference matrix.

The optimum is found by subset dynamic programming, so the cost is exponential in the
number of candidates rather than approximate.
"""

import os
import sys

import numpy as np
from numba import get_num_threads, njit, prange

from ballots import ScoreType

KEMENY_MAX_CANDIDATES = 33
"""The widest field an exact table can address, bounded by memory rather than by time."""

KEMENY_HOST_HEADROOM = 1 << 30
"""Host memory the tables must leave behind for the rest of the machine."""


# region Subset Sums


@njit(cache=True)
def kemeny_subset_sums(preferences: np.ndarray, low_bits: int):
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
                    low[candidate, subset] = low[candidate, subset ^ bit] + votes
        for offset in range(high_bits):
            bit = 1 << offset
            votes = preferences[candidate, low_bits + offset]
            for subset in range(bit, 1 << high_bits):
                if subset & bit:
                    high[candidate, subset] = high[candidate, subset ^ bit] + votes
    return low, high


@njit(cache=True)
def kemeny_votes_against(low, high, low_bits: int, candidate: int, subset: int) -> int:
    """Votes that preferred this candidate to every member of the subset."""
    return low[candidate, subset & ((1 << low_bits) - 1)] + high[candidate, subset >> low_bits]


# endregion Subset Sums


# region Cost Table


@njit(cache=True)
def kemeny_binomials(num_candidates: int) -> np.ndarray:
    """Pascal's triangle, whose last row counts the subsets seating each number of candidates."""
    binomials = np.zeros((num_candidates + 1, num_candidates + 1), dtype=np.int64)
    for upper in range(num_candidates + 1):
        binomials[upper, 0] = 1
        for lower in range(1, upper + 1):
            binomials[upper, lower] = binomials[upper - 1, lower] + binomials[upper - 1, lower - 1]
    return binomials


@njit(cache=True)
def kemeny_unrank_colex(binomials: np.ndarray, num_candidates: int, seated: int, rank: int) -> int:
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
def kemeny_costs(preferences: np.ndarray, low_bits: int, low, high, score_type):
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
                    score = score_type(costs[rest] + kemeny_votes_against(low, high, low_bits, candidate, rest))
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
    return ((1 << num_candidates) + sums_states) * np.dtype(score_type.value).itemsize


def available_host_bytes() -> int:
    """Physical memory the host will still hand out, or every byte it could name when it will not say."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return sys.maxsize


def resolve_score_type(preferences: np.ndarray, score_type: ScoreType = ScoreType.auto) -> ScoreType:
    """Select a width that represents every intermediate score, reserving its maximum as a sentinel."""
    num_candidates = preferences.shape[0]
    if num_candidates < 1 or num_candidates > KEMENY_MAX_CANDIDATES:
        raise ValueError(f"Kemeny is exact to {KEMENY_MAX_CANDIDATES} candidates")

    score_type = ScoreType(score_type)
    bound = sum(
        max(int(preferences[row, column]), int(preferences[column, row]))
        for row in range(num_candidates)
        for column in range(row + 1, num_candidates)
    )
    if score_type is ScoreType.auto:
        score_type = ScoreType.uint32 if bound < np.iinfo(np.uint32).max else ScoreType.uint64
    if bound >= np.iinfo(np.dtype(score_type.value)).max:
        raise OverflowError("Kemeny score bound exceeds the selected arithmetic type")

    return score_type


def compute_kemeny_ranking(preferences: np.ndarray, *, score_type: ScoreType = ScoreType.auto) -> tuple[list[int], int]:
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
    score_dtype = np.dtype(score_type.value).type

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
    low, high = kemeny_subset_sums(counts, low_bits)
    costs = kemeny_costs(counts, low_bits, low, high, score_dtype)

    # Walk the choices back out, which recovers the ranking from its last seat upwards.
    ranking = []
    subset = (1 << num_candidates) - 1
    while subset:
        for candidate in range(num_candidates):
            bit = 1 << candidate
            if not subset & bit:
                continue
            rest = subset ^ bit
            if costs[subset] != costs[rest] + kemeny_votes_against(low, high, low_bits, candidate, rest):
                continue
            ranking.append(candidate)
            subset = rest
            break
        else:
            # Every subset was filled from one of its members, so one of them has to match back.
            raise RuntimeError("The Kemeny cost table disagrees with its own sums")
    ranking.reverse()
    return ranking, int(costs[(1 << num_candidates) - 1])


# endregion Ranking
