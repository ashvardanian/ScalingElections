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

    @property
    def bits(self) -> int:
        if self is ScoreType.auto:
            raise ValueError("Resolve automatic score type before reading its width")
        return int(self.value.removeprefix("uint"))


# region Tally


@njit
def populate_preferences_from_ranking(preferences: np.ndarray, ranking: np.ndarray):
    """
    Populates the preference matrix based on a ranking of candidates.
    The candidate must be represented as monotonic integers starting from 0.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
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
    preferences = np.zeros((num_candidates, num_candidates), dtype=np.uint32)
    for chunk in chunks:
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != num_candidates:
            raise ValueError(f"Every chunk must be 2-D and {num_candidates} wide, got {chunk.shape}")
        preferences += tally_ballots(chunk, implementation=implementation, backend=backend)
    return preferences


def build_pairwise_preferences(
    voter_rankings: Iterable[Sequence[int] | np.ndarray],
    num_candidates: int | None = None,
    backend: Backend = Backend.cpu,
    *,
    implementation: str | None = None,
) -> np.ndarray:
    """
    Counts, for every ordered pair, the ballots preferring the first candidate to the second.

    Ballots may omit candidates, in which case every omitted one is ranked last in index order.
    Pass `num_candidates` to skip the pass that would otherwise read every ballot to find it.
    """
    rankings = [np.asarray(ranking) for ranking in voter_rankings]
    if num_candidates is None:
        num_candidates = 1 + max((int(np.max(ranking)) for ranking in rankings if len(ranking)), default=0)
    return tally_chunks(
        [complete_rankings(rankings, num_candidates)], num_candidates, backend, implementation=implementation
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
    signed = preferences.astype(np.int64)
    return np.maximum(signed - signed.T, 0).astype(np.uint32)


# endregion Graphs
