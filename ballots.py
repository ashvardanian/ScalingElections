"""Pairwise preference matrices, built from voter ballots or drawn at random.

Every downstream method in this repository consumes the same square matrix, where cell
(i, j) counts the voters preferring candidate i to candidate j.
"""

from collections.abc import Callable, Iterable, Sequence
from enum import IntEnum, StrEnum
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

type CountScalar = np.uint64 | np.uint32 | np.uint16
type UnsignedScalar = CountScalar | np.uint8

# Plain assignments rather than `type` statements, because each one is also passed as a NumPy dtype.
CandidateIndex = np.uint32
"""One candidate ID, as listed on a ballot."""
RankLabel = np.uint32
"""One rank label; equal labels tie and lower labels win."""
BallotOffset = np.uint64
"""One ballot boundary in the flat candidate entries."""
VoterWeight = np.uint64
"""One ballot's integer weight."""
PolicyCode = np.uint8
"""One ballot's omission policy, encoded as an `UnrankedCode`."""
RankPosition = np.int64
"""One candidate's effective rank within a ballot; signed so that -1 can mark an unranked candidate."""


SATURATED_SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)
"""The value a `saturated64` count clamps at, which no exact score type may ever reach."""


class UnrankedCode(IntEnum):
    """Compact omission policies stored in kernel input buffers."""

    unknown = 0
    worse = 1


class PairwiseRelationCode(IntEnum):
    """Plane indices for weighted pairwise comparison counts, or all three planes at once."""

    preference = 0
    indifference = 1
    unknown = 2
    all = 3


class Backend(StrEnum):
    """Execution target for the selected implementation."""

    cpu = "cpu"
    gpu = "gpu"


class ScoreType(StrEnum):
    """Requested counter width and overflow handling."""

    auto = "auto"
    saturated64 = "saturated64"
    uint64 = "uint64"
    uint32 = "uint32"
    uint16 = "uint16"

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

    @property
    def dtype(self: "ScoreType") -> type[np.unsignedinteger]:
        """The NumPy scalar every count of this resolved type is stored and added in."""
        return np.dtype(f"uint{self.bits}").type

    @property
    def addition(self: "ScoreType") -> "Callable[[CountScalar, CountScalar], CountScalar]":
        """The compiled addition kernels run at this resolved type, clamping only for `saturated64`."""
        return add_saturated if self is ScoreType.saturated64 else add_exact


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
) -> tuple[UnrankedCode, NDArray[PolicyCode] | None]:
    """Keep a scalar omission policy implicit, or encode the supplied per-voter policies."""
    codes = {Unranked.unknown: UnrankedCode.unknown, Unranked.worse: UnrankedCode.worse}
    if isinstance(unranked, str):
        return codes[Unranked(unranked)], None
    values = list(unranked)
    if len(values) != num_ballots:
        raise ValueError("Unranked policies must match the ballot count")
    return UnrankedCode.unknown, np.array([codes[Unranked(value)] for value in values], dtype=PolicyCode)


@njit(inline="always")
def add_exact(left: CountScalar, right: CountScalar) -> CountScalar:
    """Adds two counts of one exact score type, whose total bound was validated before the kernel ran."""
    return left + right


@njit(inline="always")
def add_saturated(left: np.uint64, right: np.uint64) -> np.uint64:
    """Adds two `saturated64` counts, clamping at the overflow sentinel."""
    return left + min(right, SATURATED_SENTINEL - left)


@njit
def saturated_sum(values: NDArray[np.unsignedinteger]) -> VoterWeight:
    """Bounds a total weight, clamping at the overflow sentinel rather than wrapping."""
    total = VoterWeight(0)
    for value in values:
        total = add_saturated(total, VoterWeight(value))
    return total


def integer_source(values: ArrayLike, *, ndim: int) -> tuple[NDArray[np.generic], int]:
    """Validate nonnegative integer values without converting their numeric storage."""
    array = np.asarray(values) if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if array.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional integer array")
    if np.issubdtype(array.dtype, np.object_):
        if any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) for value in array.flat):
            raise TypeError("Entries must be integers")
    elif not np.issubdtype(array.dtype, np.integer) and array.size:
        raise TypeError("Entries must be integers")
    if not array.size:
        return array, 0
    maximum = int(np.max(array))
    if not np.issubdtype(array.dtype, np.unsignedinteger) and np.min(array) < 0:
        raise OverflowError("Entries must fit uint64")
    if maximum > np.iinfo(np.uint64).max:
        raise OverflowError("Entries must fit uint64")
    return array, maximum


def unsigned_array[Scalar: UnsignedScalar](values: ArrayLike, dtype: type[Scalar], *, ndim: int) -> NDArray[Scalar]:
    """Borrow compatible storage or convert validated integers directly to the requested width."""
    array, maximum = integer_source(values, ndim=ndim)
    if maximum > np.iinfo(dtype).max:
        raise OverflowError(f"Entries must fit {np.dtype(dtype).name}")
    return np.ascontiguousarray(array, dtype=dtype)


@njit
def validate_ballot_ids(
    candidates: NDArray[CandidateIndex], offsets: NDArray[BallotOffset], num_candidates: int
) -> None:
    """Reject duplicate candidate identifiers within an already range-checked ballot."""
    # A nonempty ballot's end offset is unique and nonzero, so it marks the candidates that ballot listed.
    seen = np.zeros(num_candidates, dtype=BallotOffset)
    for ballot in range(len(offsets) - 1):
        end = offsets[ballot + 1]
        for entry in range(int(offsets[ballot]), int(end)):
            candidate = candidates[entry]
            if seen[candidate] == end:
                raise ValueError("Candidate IDs must be in range and distinct within each ballot")
            seen[candidate] = end


def prepare_ballots(
    candidates: ArrayLike,
    offsets: ArrayLike | None,
    num_candidates: int | None,
    ranks: ArrayLike | None,
    weights: ArrayLike | None,
) -> tuple[NDArray[CandidateIndex], NDArray[BallotOffset], int, NDArray[RankLabel] | None, NDArray[VoterWeight] | None]:
    """Validate dense or compressed-row ballots and return canonical input arrays."""
    if ranks is not None:
        ranks = unsigned_array(ranks, RankLabel, ndim=2 if offsets is None else 1)
    if offsets is None:
        dense = unsigned_array(candidates, CandidateIndex, ndim=2)
        ballots, width = dense.shape
        if num_candidates is None:
            num_candidates = width
        ids = dense.reshape(-1)
        ballot_offsets = np.arange(ballots + 1, dtype=BallotOffset) * BallotOffset(width)
        if ranks is not None:
            if ranks.shape != dense.shape:
                raise ValueError("Ranks must match the rankings shape")
            ranks = ranks.reshape(-1)
    else:
        if num_candidates is None:
            raise ValueError("CSR ballots require num_candidates")
        ids = unsigned_array(candidates, CandidateIndex, ndim=1)
        ballot_offsets = unsigned_array(offsets, BallotOffset, ndim=1)
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
    if num_candidates < 0 or num_candidates > np.iinfo(CandidateIndex).max:
        raise OverflowError("num_candidates must fit UInt32")
    if num_candidates == 0:
        raise ValueError("num_candidates must be positive")
    if np.any(ids >= num_candidates):
        raise ValueError("Candidate IDs must be in range and distinct within each ballot")
    validate_ballot_ids(ids, ballot_offsets, num_candidates)
    if ranks is not None and len(ranks) != len(ids):
        raise ValueError("Ranks must match the entries length")
    if weights is not None:
        weights = unsigned_array(weights, VoterWeight, ndim=1)
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
    candidates: NDArray[CandidateIndex],
    offsets: NDArray[BallotOffset],
    ranks: NDArray[RankLabel] | None,
    weights: NDArray[CountScalar],
    policies: NDArray[PolicyCode] | None,
    unranked: UnrankedCode,
    relation: PairwiseRelationCode,
    counts: NDArray[CountScalar],
    add: Callable[[CountScalar, CountScalar], CountScalar],
) -> None:
    """
    Accumulates one relation, or all three planes, into zeroed `counts`, leaving every diagonal zero.

    `weights` arrive in the count type, so `add` runs at exactly the resolved score type.
    """
    num_candidates = counts.shape[1]
    preferences = counts[0]
    labels = np.empty(num_candidates, dtype=RankPosition)
    for ballot in range(len(offsets) - 1):
        start, end = int(offsets[ballot]), int(offsets[ballot + 1])
        weight = weights[ballot]
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
                    if relation != PairwiseRelationCode.all and category != relation:
                        continue
                    output = counts[category.value if relation == PairwiseRelationCode.all else 0]
                    if category == PairwiseRelationCode.preference:
                        preferred, worse = candidate, opponent
                        if left < 0 or (right >= 0 and left > right):
                            preferred, worse = opponent, candidate
                        output[preferred, worse] = add(output[preferred, worse], weight)
                    else:
                        count = add(output[candidate, opponent], weight)
                        output[candidate, opponent] = count
                        output[opponent, candidate] = count
            continue
        for entry in range(start, end):
            preferred = candidates[entry]
            for other in range(start, end):
                opponent = candidates[other]
                if labels[preferred] < labels[opponent]:
                    preferences[preferred, opponent] = add(preferences[preferred, opponent], weight)
            if policy == UnrankedCode.worse:
                for opponent in range(num_candidates):
                    if labels[opponent] < 0:
                        preferences[preferred, opponent] = add(preferences[preferred, opponent], weight)


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


@njit
def tally_rankings(preferences: NDArray[np.integer], rankings: NDArray[np.integer]) -> None:
    """Adds every complete ranking, one per row, into a matrix wide enough for their count."""
    num_candidates = rankings.shape[1]
    for ranking in rankings:
        for position in range(num_candidates - 1):
            preferred = ranking[position]
            for later in range(position + 1, num_candidates):
                preferences[preferred, ranking[later]] += 1


def complete_rankings(
    voter_rankings: Iterable[Sequence[int] | NDArray[np.integer]], num_candidates: int
) -> NDArray[np.integer]:
    """Pads every ballot to a full ranking, placing the candidates it omits last in index order."""
    rankings = [np.asarray(ranking) for ranking in voter_rankings]
    complete = np.empty((len(rankings), num_candidates), dtype=CandidateIndex)
    for row, ranking in enumerate(rankings):
        if ranking.ndim != 1 or (ranking.size and not np.issubdtype(ranking.dtype, np.integer)):
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
    rankings = np.array([generator.permutation(num_candidates) for _ in range(num_voters)], dtype=CandidateIndex)
    score_type = resolve_tally_score_type(num_voters)
    preferences = np.zeros((num_candidates, num_candidates), dtype=score_type.dtype)
    tally_rankings(preferences, rankings)
    return preferences


# endregion Tally
