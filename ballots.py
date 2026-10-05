"""Pairwise preference matrices, built from voter ballots or drawn at random.

Every downstream method in this repository consumes the same square matrix, where cell
(i, j) counts the voters preferring candidate i to candidate j.
"""

from collections.abc import Iterable, Sequence
from enum import StrEnum

import numpy as np
from numba import njit


class Backend(StrEnum):
    cpu = "cpu"
    gpu = "gpu"


class ScoreType(StrEnum):
    auto = "auto"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    saturated64 = "saturated64"

    @property
    def bits(self) -> int:
        if self is ScoreType.auto:
            raise ValueError("Resolve automatic score type before reading its width")
        return 64 if self is ScoreType.saturated64 else int(self.value.removeprefix("uint"))

    @property
    def max_exact(self) -> int:
        return (1 << self.bits) - 1 - (self.bits == 64)


class Unranked(StrEnum):
    unknown = "unknown"
    worse = "worse"


@njit(inline="always")
def saturating_add(left, right):
    left, right = np.uint64(left), np.uint64(right)
    maximum = np.uint64(0xFFFFFFFFFFFFFFFF)
    return maximum if right > maximum - left else left + right


@njit(inline="always")
def add_scores(left, right, saturated):
    return saturating_add(left, right) if saturated else np.uint64(left) + np.uint64(right)


@njit
def saturated_sum(values):
    total = np.uint64(0)
    for value in values:
        total = saturating_add(total, value)
    return total


def unsigned_array(values, dtype, *, ndim):
    array = np.asarray(values) if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional integer array")
    if array.dtype.kind == "O":
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in array.flat):
            raise TypeError("Entries must be integers")
    elif array.dtype.kind not in "iu" and array.size:
        raise TypeError("Entries must be integers")
    if np.any(array < 0) or np.any(array > np.iinfo(dtype).max):
        raise OverflowError(f"Entries must fit {np.dtype(dtype).name}")
    return np.ascontiguousarray(array, dtype=dtype)


@njit
def validate_ballot_ids(ids, offsets, num_candidates):
    seen = np.full(num_candidates, -1, dtype=np.int64)
    for ballot in range(len(offsets) - 1):
        start, end = int(offsets[ballot]), int(offsets[ballot + 1])
        for entry in range(start, end):
            candidate = ids[entry]
            if seen[candidate] == ballot:
                raise ValueError("Every ballot must list distinct candidates")
            seen[candidate] = ballot


def prepare_ballots(rankings, offsets, num_candidates, ranks, weights):
    if offsets is None:
        dense = unsigned_array(rankings, np.uint32, ndim=2)
        ballots, width = dense.shape
        if num_candidates is None:
            num_candidates = width
        ids = dense.reshape(-1)
        offsets = np.arange(ballots + 1, dtype=np.uint64) * width
        if ranks is not None:
            ranks = unsigned_array(ranks, np.uint32, ndim=2)
            if ranks.shape != dense.shape:
                raise ValueError("Ranks must match the rankings shape")
            ranks = ranks.reshape(-1)
    else:
        if num_candidates is None:
            raise ValueError("CSR ballots require num_candidates")
        ids = unsigned_array(rankings, np.uint32, ndim=1)
        offsets = unsigned_array(offsets, np.uint64, ndim=1)
        if not len(offsets) or offsets[0] != 0 or offsets[-1] != len(ids) or np.any(offsets[1:] < offsets[:-1]):
            raise ValueError("Offsets must start at zero, be monotone, and end at the entries length")
        ballots = len(offsets) - 1
    if (
        isinstance(num_candidates, (bool, np.bool_))
        or not isinstance(num_candidates, (int, np.integer))
        or not 0 < num_candidates <= np.iinfo(np.uint32).max
    ):
        raise ValueError("num_candidates must be a positive UInt32 integer")
    if np.any(ids >= num_candidates):
        raise ValueError("Candidate index exceeds num_candidates")
    validate_ballot_ids(ids, offsets, num_candidates)
    if ranks is not None:
        ranks = unsigned_array(ranks, np.uint32, ndim=1)
        if len(ranks) != len(ids):
            raise ValueError("Ranks must match the entries length")
    weights = np.ones(ballots, dtype=np.uint64) if weights is None else unsigned_array(weights, np.uint64, ndim=1)
    if len(weights) != ballots:
        raise ValueError("Weights must match the ballot count")
    return ids, offsets, int(num_candidates), ranks, weights


def tally_score_type(weights: np.ndarray, score_type: ScoreType = ScoreType.auto) -> ScoreType:
    score_type = ScoreType(score_type)
    bound = int(saturated_sum(weights))
    if score_type is ScoreType.auto:
        if bound <= np.iinfo(np.uint32).max:
            return ScoreType.uint32
        return ScoreType.uint64 if bound < np.iinfo(np.uint64).max else ScoreType.saturated64
    if score_type is not ScoreType.saturated64 and bound > score_type.max_exact:
        raise OverflowError("Tally bound exceeds the selected arithmetic type")
    return score_type


@njit
def tally_ragged(ids, offsets, num_candidates, ranks, weights, worse, score_dtype, saturated):
    preferences = np.zeros((num_candidates, num_candidates), dtype=score_dtype)
    listed = np.empty(num_candidates, dtype=np.bool_)
    for ballot in range(len(weights)):
        start, end = int(offsets[ballot]), int(offsets[ballot + 1])
        weight = weights[ballot]
        if worse:
            listed[:] = False
            for entry in range(start, end):
                listed[ids[entry]] = True
        for entry in range(start, end):
            preferred = ids[entry]
            rank = entry if ranks is None else ranks[entry]
            for other in range(start, end):
                other_rank = other if ranks is None else ranks[other]
                if rank < other_rank:
                    opponent = ids[other]
                    preferences[preferred, opponent] = add_scores(preferences[preferred, opponent], weight, saturated)
            if worse:
                for opponent in range(num_candidates):
                    if not listed[opponent]:
                        preferences[preferred, opponent] = add_scores(
                            preferences[preferred, opponent], weight, saturated
                        )
    return preferences


# region Tally


@njit
def populate_preferences_from_ranking(preferences: np.ndarray, ranking: np.ndarray):
    """
    Populates the preference matrix based on a ranking of candidates.
    The candidate must be represented as monotonic integers starting from 0.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    capacity = np.uint64(np.iinfo(preferences.dtype).max)
    if capacity == np.uint64(0xFFFFFFFFFFFFFFFF):
        capacity -= np.uint64(1)
    if len(ranking) > 1 and np.max(preferences) >= capacity:
        raise OverflowError("Appending a ballot may exceed the count range")
    for position, preferred in enumerate(ranking):
        for opponent in ranking[position + 1 :]:
            preferences[preferred, opponent] += 1


def complete_rankings(voter_rankings: Iterable[Sequence[int] | np.ndarray], num_candidates: int) -> np.ndarray:
    """Pads every ballot to a full ranking, placing the candidates it omits last in index order."""
    rankings = [np.asarray(ranking) for ranking in voter_rankings]
    complete = np.empty((len(rankings), num_candidates), dtype=np.uint32)
    for row, ranking in enumerate(rankings):
        if ranking.ndim != 1 or (ranking.size and ranking.dtype.kind not in "iu"):
            raise ValueError("Every ballot must contain integer candidate indices")
        if np.any(ranking < 0) or np.any(ranking >= num_candidates) or len(np.unique(ranking)) != len(ranking):
            raise ValueError("Every ballot must list distinct candidates within the candidate range")
        complete[row, : len(ranking)] = ranking
        if len(ranking) == num_candidates:
            continue
        unranked = np.ones(num_candidates, dtype=bool)
        unranked[ranking.astype(np.intp)] = False
        complete[row, len(ranking) :] = np.nonzero(unranked)[0]
    return complete


def tally_chunks(
    chunks: Iterable[np.ndarray],
    num_candidates: int,
    backend: Backend = Backend.cpu,
    *,
    implementation: str | None = None,
) -> np.ndarray:
    """
    Sums one pairwise matrix over any number of chunks of complete rankings.

    Taking chunks rather than one array is what keeps a national electorate off the heap: only the
    chunk in hand and the matrix itself are ever resident.
    """
    from scalingelections import _resolve, tally_ballots

    _resolve(implementation, backend)
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint64)
    for chunk in chunks:
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != num_candidates:
            raise ValueError(f"Every chunk must be 2-D and {num_candidates} wide, got {chunk.shape}")
        counted = tally_ballots(chunk, implementation=implementation, backend=backend).astype(np.uint64)
        if np.any(counted >= np.iinfo(np.uint64).max - preferences):
            raise OverflowError("Chunked tally exceeds the representable count range")
        preferences += counted
    return preferences.astype(np.uint32) if np.max(preferences) <= np.iinfo(np.uint32).max else preferences


def build_pairwise_preferences(
    voter_rankings: Iterable[Sequence[int] | np.ndarray],
    num_candidates: int | None = None,
    backend: Backend = Backend.cpu,
    *,
    weights=None,
    unranked: Unranked = Unranked.unknown,
    implementation: str | None = None,
) -> np.ndarray:
    """Count possibly incomplete rankings, preserving the selected interpretation of omitted candidates."""
    from scalingelections import tally_ballots

    rankings = [unsigned_array(ranking, np.uint32, ndim=1) for ranking in voter_rankings]
    if num_candidates is None:
        num_candidates = 1 + max((int(np.max(ranking)) for ranking in rankings if len(ranking)), default=-1)
    offsets = np.zeros(len(rankings) + 1, dtype=np.uint64)
    offsets[1:] = np.cumsum([len(ranking) for ranking in rankings], dtype=np.uint64)
    entries = np.concatenate(rankings) if rankings else np.empty(0, dtype=np.uint32)
    return tally_ballots(
        entries,
        offsets=offsets,
        num_candidates=num_candidates,
        weights=weights,
        unranked=unranked,
        implementation=implementation,
        backend=backend,
    )


def generate_preferences(num_candidates: int, num_voters: int, generator: np.random.Generator) -> np.ndarray:
    """
    Draws a preference matrix for a synthetic election of the requested shape.

    Zero voters asks for the counts themselves to be random, which keeps the largest benchmarks
    from materializing ballots nobody reads.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(m * n^2), where n is the number of candidates and m is the number of voters.
    """
    if num_voters == 0:
        return generator.integers(0, num_candidates, (num_candidates, num_candidates), dtype=np.uint32)
    voter_rankings = [generator.permutation(num_candidates) for _ in range(num_voters)]
    return build_pairwise_preferences(voter_rankings)


# endregion Tally


# region Graphs


def positive_margins(preferences: np.ndarray) -> np.ndarray:
    """
    Rewrites pairwise counts so the strongest-paths kernel closes over margins.

    The kernel keeps `preferences[i, j]` when it exceeds `preferences[j, i]` and zeroes it
    otherwise, so feeding it the clipped margin makes it compute the widest paths of the
    positive-margin graph without any change to the kernel itself.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    return preferences - np.minimum(preferences, preferences.T)


# endregion Graphs
