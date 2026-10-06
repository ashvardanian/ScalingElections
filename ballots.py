"""Pairwise preference matrices, built from voter ballots or drawn at random.

Every downstream method in this repository consumes the same square matrix, where cell
(i, j) counts the voters preferring candidate i to candidate j.
"""

from collections.abc import Iterable, Sequence
from enum import IntEnum, StrEnum
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from numba import njit


type CountScalar = np.uint16 | np.uint32 | np.uint64
type UnsignedScalar = np.uint8 | CountScalar


class Arithmetic(IntEnum):
    """Addition semantics used by compiled integer kernels."""

    exact = 0
    saturated = 1


class UnrankedCode(IntEnum):
    """Compact omission policies stored in kernel input buffers."""

    unknown = 0
    worse = 1


class PairwiseRelationCode(IntEnum):
    """Plane indices for weighted pairwise comparison counts."""

    preference = 0
    indifference = 1
    unknown = 2


class Backend(StrEnum):
    """Execution target for the selected implementation."""

    cpu = "cpu"
    gpu = "gpu"


class ScoreType(StrEnum):
    """Requested counter width and overflow handling."""

    auto = "auto"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    saturated64 = "saturated64"

    @property
    def bits(self: "ScoreType") -> int:
        """Storage width after automatic selection has been resolved."""
        if self is ScoreType.auto:
            raise ValueError("Resolve automatic score type before reading its width")
        return 64 if self is ScoreType.saturated64 else int(self.value.removeprefix("uint"))

    @property
    def max_exact(self: "ScoreType") -> int:
        """Largest accepted tally, excluding the UInt64 overflow sentinel."""
        return (1 << self.bits) - 1 - (self.bits == 64)


class Unranked(StrEnum):
    """Meaning of candidates omitted from a voter’s ballot."""

    unknown = "unknown"
    worse = "worse"


class PairwiseRelation(StrEnum):
    """The comparison counted for each ordered candidate pair."""

    preference = "preference"
    """The row candidate is strictly preferred."""
    indifference = "indifference"
    """The candidates share a rank."""
    unknown = "unknown"
    """The ballot leaves the comparison unspecified."""


class PairwiseCounts(NamedTuple):
    """Weighted strict preferences, indifference, and unexpressed comparisons."""

    preferences: NDArray[CountScalar]
    """Weight preferring the row candidate to the column candidate."""
    indifference: NDArray[CountScalar]
    """Weight tying the candidates, including jointly omitted candidates under worse semantics."""
    unknown: NDArray[CountScalar]
    """Weight leaving the comparison unspecified under unknown semantics."""


def prepare_unranked(
    unranked: Unranked | str | Sequence[Unranked | str], num_ballots: int
) -> tuple[UnrankedCode, NDArray[np.uint8] | None]:
    """Keep a scalar omission policy implicit, or encode the supplied per-voter policies."""
    codes = {Unranked.unknown: UnrankedCode.unknown, Unranked.worse: UnrankedCode.worse}
    if isinstance(unranked, str):
        return codes[Unranked(unranked)], None
    values = list(unranked)
    if len(values) != num_ballots:
        raise ValueError("Unranked policies must match the ballot count")
    return UnrankedCode.unknown, np.array([codes[Unranked(value)] for value in values], dtype=np.uint8)


@njit(inline="always")
def saturating_add(left: int | np.integer, right: int | np.integer) -> np.uint64:
    """Clamp an unsigned sum to the UInt64 overflow sentinel."""
    left, right = np.uint64(left), np.uint64(right)
    maximum = np.uint64(0xFFFFFFFFFFFFFFFF)
    return maximum if right > maximum - left else left + right


@njit(inline="always")
def add_scores(left: int | np.integer, right: int | np.integer, arithmetic: Arithmetic) -> np.uint64:
    """Add integer counts with the selected overflow semantics."""
    return saturating_add(left, right) if arithmetic == Arithmetic.saturated else np.uint64(left) + np.uint64(right)


@njit
def saturated_sum(values: NDArray[np.integer]) -> np.uint64:
    """Bound a count total without allowing unsigned wraparound."""
    total = np.uint64(0)
    for value in values:
        total = saturating_add(total, value)
    return total


def integer_source(values: ArrayLike, *, ndim: int) -> tuple[NDArray[np.generic], int]:
    """Validate nonnegative integer values without converting their numeric storage."""
    array = np.asarray(values) if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional integer array")
    if array.dtype.kind == "O":
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in array.flat):
            raise TypeError("Entries must be integers")
    elif array.dtype.kind not in "iu" and array.size:
        raise TypeError("Entries must be integers")
    maximum = int(np.max(array)) if array.size else 0
    if (array.dtype.kind != "u" and array.size and np.min(array) < 0) or maximum > np.iinfo(np.uint64).max:
        raise OverflowError("Entries must fit uint64")
    return array, maximum


def unsigned_array[Scalar: UnsignedScalar](values: ArrayLike, dtype: type[Scalar], *, ndim: int) -> NDArray[Scalar]:
    """Borrow compatible storage or convert validated integers directly to the requested width."""
    array, maximum = integer_source(values, ndim=ndim)
    if maximum > np.iinfo(dtype).max:
        raise OverflowError(f"Entries must fit {np.dtype(dtype).name}")
    return np.ascontiguousarray(array, dtype=dtype)


@njit
def validate_ballot_ids(candidates: NDArray[np.uint32], offsets: NDArray[np.uint64], num_candidates: int) -> None:
    """Reject duplicate candidate identifiers within an already range-checked ballot."""
    seen = np.full(num_candidates, -1, dtype=np.int64)
    for ballot in range(len(offsets) - 1):
        start, end = int(offsets[ballot]), int(offsets[ballot + 1])
        for entry in range(start, end):
            candidate = candidates[entry]
            if seen[candidate] == ballot:
                raise ValueError("Every ballot must list distinct candidates")
            seen[candidate] = ballot


def prepare_ballots(
    candidates: ArrayLike,
    offsets: ArrayLike | None,
    num_candidates: int | None,
    ranks: ArrayLike | None,
    weights: ArrayLike | None,
) -> tuple[NDArray[np.uint32], NDArray[np.uint64], int, NDArray[np.uint32] | None, NDArray[np.uint64] | None]:
    """Validate dense or compressed-row ballots and return canonical input arrays."""
    if ranks is not None:
        ranks = unsigned_array(ranks, np.uint32, ndim=2 if offsets is None else 1)
    if offsets is None:
        dense = unsigned_array(candidates, np.uint32, ndim=2)
        ballots, width = dense.shape
        if num_candidates is None:
            num_candidates = width
        ids = dense.reshape(-1)
        ballot_offsets = np.arange(ballots + 1, dtype=np.uint64) * np.uint64(width)
        if ranks is not None:
            if ranks.shape != dense.shape:
                raise ValueError("Ranks must match the rankings shape")
            ranks = ranks.reshape(-1)
    else:
        if num_candidates is None:
            raise ValueError("CSR ballots require num_candidates")
        ids = unsigned_array(candidates, np.uint32, ndim=1)
        ballot_offsets = unsigned_array(offsets, np.uint64, ndim=1)
        if (
            not len(ballot_offsets)
            or ballot_offsets[0] != 0
            or ballot_offsets[-1] != len(ids)
            or np.any(ballot_offsets[1:] < ballot_offsets[:-1])
        ):
            raise ValueError("Offsets must start at zero, be monotone, and end at the entries length")
        ballots = len(ballot_offsets) - 1
    if isinstance(num_candidates, (bool, np.bool_)) or not isinstance(num_candidates, (int, np.integer)):
        raise TypeError("num_candidates must be an integer")
    if num_candidates < 0 or num_candidates > np.iinfo(np.uint32).max:
        raise OverflowError("num_candidates must fit UInt32")
    if num_candidates == 0:
        raise ValueError("num_candidates must be positive")
    if np.any(ids >= num_candidates):
        raise ValueError("Candidate index exceeds num_candidates")
    validate_ballot_ids(ids, ballot_offsets, num_candidates)
    if ranks is not None:
        if len(ranks) != len(ids):
            raise ValueError("Ranks must match the entries length")
    if weights is not None:
        weights = unsigned_array(weights, np.uint64, ndim=1)
        if len(weights) != ballots:
            raise ValueError("Weights must match the ballot count")
    return ids, ballot_offsets, int(num_candidates), ranks, weights


def resolve_tally_score_type(bound: int | np.uint64, score_type: ScoreType | str = ScoreType.auto) -> ScoreType:
    """Choose or validate tally arithmetic from a saturating total-weight bound."""
    score_type = ScoreType(score_type)
    bound = int(bound)
    if score_type is ScoreType.auto:
        if bound <= np.iinfo(np.uint32).max:
            return ScoreType.uint32
        return ScoreType.uint64 if bound < np.iinfo(np.uint64).max else ScoreType.saturated64
    if score_type is not ScoreType.saturated64 and bound > score_type.max_exact:
        raise OverflowError("Tally bound exceeds the selected arithmetic type")
    return score_type


@njit
def tally_ragged(
    candidates: NDArray[np.uint32],
    offsets: NDArray[np.uint64],
    num_candidates: int,
    ranks: NDArray[np.uint32] | None,
    weights: NDArray[np.uint64] | None,
    policies: NDArray[np.uint8] | None,
    unranked: UnrankedCode,
    relation: PairwiseRelationCode | None,
    score_dtype: type[CountScalar],
    arithmetic: Arithmetic,
) -> NDArray[CountScalar]:
    """Accumulate one relation, or all relations when None, with zero diagonal counts."""
    counts = np.zeros((3 if relation is None else 1, num_candidates, num_candidates), dtype=score_dtype)
    preferences = counts[0]
    labels = np.empty(num_candidates, dtype=np.int64)
    for ballot in range(len(offsets) - 1):
        start, end = int(offsets[ballot]), int(offsets[ballot + 1])
        weight = np.uint64(1) if weights is None else weights[ballot]
        policy = unranked.value if policies is None else policies[ballot]
        if weight == 0:
            continue
        labels[:] = -1
        for entry in range(start, end):
            labels[candidates[entry]] = entry - start if ranks is None else ranks[entry]
        if relation != PairwiseRelationCode.preference:
            for candidate in range(num_candidates):
                for opponent in range(candidate + 1, num_candidates):
                    left, right = labels[candidate], labels[opponent]
                    if (left < 0 or right < 0) and policy == UnrankedCode.unknown:
                        category = PairwiseRelationCode.unknown
                    elif left == right:
                        category = PairwiseRelationCode.indifference
                    else:
                        category = PairwiseRelationCode.preference
                    if relation is not None and category != relation:
                        continue
                    output = counts[category.value if relation is None else 0]
                    if category == PairwiseRelationCode.preference:
                        preferred, worse = candidate, opponent
                        if left < 0 or (right >= 0 and left > right):
                            preferred, worse = opponent, candidate
                        output[preferred, worse] = add_scores(output[preferred, worse], weight, arithmetic)
                    else:
                        count = add_scores(output[candidate, opponent], weight, arithmetic)
                        output[candidate, opponent] = count
                        output[opponent, candidate] = count
            continue
        for entry in range(start, end):
            preferred = candidates[entry]
            for other in range(start, end):
                opponent = candidates[other]
                if labels[preferred] < labels[opponent]:
                    preferences[preferred, opponent] = add_scores(preferences[preferred, opponent], weight, arithmetic)
            if policy == UnrankedCode.worse:
                for opponent in range(num_candidates):
                    if labels[opponent] < 0:
                        preferences[preferred, opponent] = add_scores(
                            preferences[preferred, opponent], weight, arithmetic
                        )
    return counts


# region Tally


@njit
def populate_preferences_from_ranking(preferences: NDArray[np.integer], ranking: NDArray[np.integer]) -> None:
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


def complete_rankings(
    voter_rankings: Iterable[Sequence[int] | NDArray[np.integer]], num_candidates: int
) -> NDArray[np.integer]:
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
    chunks: Iterable[NDArray[np.integer]],
    num_candidates: int,
    backend: Backend = Backend.cpu,
    *,
    implementation: str | None = None,
) -> NDArray[np.integer]:
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
    voter_rankings: Iterable[Sequence[int] | NDArray[np.integer]],
    num_candidates: int | None = None,
    backend: Backend = Backend.cpu,
    *,
    weights: ArrayLike | None = None,
    unranked: Unranked = Unranked.unknown,
    implementation: str | None = None,
) -> NDArray[CountScalar]:
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


def generate_preferences(num_candidates: int, num_voters: int, generator: np.random.Generator) -> NDArray[CountScalar]:
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


def positive_margins(preferences: NDArray[np.integer]) -> NDArray[np.integer]:
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
