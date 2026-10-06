"""Every method in one namespace, so the split into per-method modules stays invisible.

Ballots fold into one `N x N` matrix of pairwise counts, and Schulze, Split Cycle and
Kemeny-Young all read that matrix and nothing else. `bench.py` drives the benchmarks.
Tally operations accumulate ballot weights; compute operations return one result;
enumerate operations lazily yield every optimum from one solve.
"""

import importlib
from collections.abc import Generator, Iterable, Sequence
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray

import kemeny
import schulze
from ballots import (
    SATURATED_SENTINEL,
    Backend,
    BallotOffset,
    CandidateIndex,
    CountScalar,
    PairwiseCounts,
    PairwiseRelation,
    PairwiseRelationCode,
    ScoreType,
    Unranked,
    VoterWeight,
    add_saturated,
    complete_rankings,
    generate_preferences,
    integer_source,
    populate_preferences_from_ranking,
    prepare_ballots,
    prepare_unranked,
    resolve_tally_score_type,
    saturated_sum,
    tally_ragged,
    tally_rankings,
    unsigned_array,
)
from schulze import (
    compute_election_results,
    compute_ranking_tiers,
    compute_strongest_paths_serial,
    compute_strongest_paths_tiled_cpu,
    positive_margins,
    select_split_cycle_winners,
)

KemenyResult = kemeny.KemenyResult
RankingMultiplicity = kemeny.RankingMultiplicity


def _module(implementation: str) -> ModuleType | None:
    """Load a named implementation, with None denoting the Python kernels."""
    if implementation == "python":
        return None
    if implementation not in ("cpp", "mojo"):
        raise ValueError("implementation must be cpp, mojo, or python")
    name = "scalingelections_cuda" if implementation == "cpp" else "scalingelections_mojo"
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"{implementation} implementation is unavailable") from error


def available_implementations() -> list[str]:
    """Installed implementations, in automatic selection order."""
    result = []
    for name in ("cpp", "mojo"):
        try:
            _module(name)
        except RuntimeError:
            continue
        result.append(name)
    return result + ["python"]


def available_backends(*, implementation: str | None = None) -> list[Backend]:
    """Execution targets available for the selected implementation."""
    if implementation is None:
        targets = {target for name in available_implementations() for target in available_backends(implementation=name)}
        return [target for target in Backend if target in targets]
    module = _module(implementation)
    return [Backend(target) for target in module.available_backends()] if module is not None else [Backend.cpu]


def _resolve(implementation: str | None, backend: Backend | str) -> ModuleType | None:
    """Select an installed implementation that explicitly supports the requested target."""
    try:
        backend = Backend(backend)
    except ValueError as error:
        raise ValueError("backend must be cpu or gpu") from error
    names = available_implementations() if implementation is None else [implementation]
    for name in names:
        module = _module(name)
        if backend in (module.available_backends() if module is not None else ["cpu"]):
            return module
    raise RuntimeError(f"{implementation or 'Any installed'} implementation has no {backend} backend")


def _preferences_matrix(values: ArrayLike) -> NDArray[np.uint64 | np.uint32]:
    """Validate a square integer matrix and choose lossless contiguous input storage."""
    values, maximum = integer_source(values, ndim=2)
    if not values.shape[0] or values.shape[0] != values.shape[1]:
        raise ValueError("Preferences must be a nonempty square matrix")
    dtype = np.uint64 if values.dtype == np.uint64 or maximum > np.iinfo(np.uint32).max else np.uint32
    return np.ascontiguousarray(values, dtype=dtype)


def tally_ballots(
    candidates: ArrayLike,
    *,
    offsets: ArrayLike | None = None,
    num_candidates: int | None = None,
    ranks: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    unranked: Unranked | str | Sequence[Unranked | str] = Unranked.unknown,
    relation: PairwiseRelation | str = PairwiseRelation.preference,
    score_type: ScoreType | str = ScoreType.auto,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
) -> NDArray[CountScalar]:
    """Count dense or CSR ballots, using positional ranks and unit weights when omitted.

    Equal rank labels tie; lower labels win, irrespective of entry order or label gaps.
    Omission policies may be scalar or per voter; zero-weight voters contribute nothing.
    """
    return _tally(
        candidates,
        offsets,
        num_candidates,
        ranks,
        weights,
        unranked,
        PairwiseRelationCode[PairwiseRelation(relation).name],
        score_type,
        implementation,
        backend,
    )[0]


def tally_pairwise_relations(
    candidates: ArrayLike,
    *,
    offsets: ArrayLike | None = None,
    num_candidates: int | None = None,
    ranks: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    unranked: Unranked | str | Sequence[Unranked | str] = Unranked.unknown,
    score_type: ScoreType | str = ScoreType.auto,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
) -> PairwiseCounts:
    """Count preferences, effective indifference, and unknown comparisons in one pass.

    Ballot arguments follow tally_ballots; all three output diagonals are zero.
    Indifference includes equal rank labels and jointly omitted candidates under worse semantics.
    Unknown omissions leave every comparison involving an omitted candidate unspecified.
    """
    return PairwiseCounts(
        *_tally(
            candidates,
            offsets,
            num_candidates,
            ranks,
            weights,
            unranked,
            PairwiseRelationCode.all,
            score_type,
            implementation,
            backend,
        )
    )


def _tally(
    candidates: ArrayLike,
    offsets: ArrayLike | None,
    num_candidates: int | None,
    ranks: ArrayLike | None,
    weights: ArrayLike | None,
    unranked: Unranked | str | Sequence[Unranked | str],
    relation: PairwiseRelationCode,
    score_type: ScoreType | str,
    implementation: str | None,
    backend: Backend | str,
) -> tuple[NDArray[CountScalar], ...]:
    """Dispatch one prepared tally and return its requested output matrices."""
    module = _resolve(implementation, backend)
    score_type = ScoreType(score_type)
    if module is not None:
        options = dict(
            offsets=offsets,
            num_candidates=num_candidates,
            ranks=ranks,
            weights=weights,
            unranked=unranked,
            score_type=score_type.value,
            backend=backend,
        )
        if relation is PairwiseRelationCode.all:
            return tuple(module.tally_pairwise_relations(candidates, **options))
        return (module.tally_ballots(candidates, relation=relation.name, **options),)
    if num_candidates is not None:
        if isinstance(num_candidates, (bool, np.bool_)) or not isinstance(num_candidates, (int, np.integer)):
            raise TypeError("num_candidates must be an integer")
        if num_candidates < 0 or num_candidates > np.iinfo(CandidateIndex).max:
            raise OverflowError("num_candidates must fit UInt32")
        if num_candidates == 0:
            raise ValueError("num_candidates must be positive")
    if offsets is None and ranks is None and weights is None and relation is PairwiseRelationCode.preference:
        dense = unsigned_array(candidates, CandidateIndex, ndim=2)
        n = dense.shape[1]
        if isinstance(unranked, str):
            Unranked(unranked)
        else:
            prepare_unranked(unranked, len(dense))
        if n and (num_candidates is None or num_candidates == n):
            if np.any(np.sort(dense, axis=1) != np.arange(n, dtype=CandidateIndex)):
                raise ValueError("Every ballot must rank each candidate exactly once")
            resolved = resolve_tally_score_type(len(dense), score_type)
            preferences = np.zeros((n, n), dtype=resolved.dtype)
            tally_rankings(preferences, dense)
            return (preferences,)
    ids, offsets, n, ranks, weights = prepare_ballots(candidates, offsets, num_candidates, ranks, weights)
    unranked_code, policies = prepare_unranked(unranked, len(offsets) - 1)
    num_ballots = len(offsets) - 1
    bound = VoterWeight(num_ballots) if weights is None else saturated_sum(weights)
    resolved = resolve_tally_score_type(bound, score_type)
    # Every weight fits the resolved type, because the validated total bounds each of them.
    counted_weights = np.ones(num_ballots, dtype=resolved.dtype) if weights is None else weights.astype(resolved.dtype)
    planes = 3 if relation is PairwiseRelationCode.all else 1
    counts = np.zeros((planes, n, n), dtype=resolved.dtype)
    tally_ragged(ids, offsets, ranks, counted_weights, policies, unranked_code, relation, counts, resolved.addition)
    if resolved is ScoreType.saturated64 and np.any(counts == SATURATED_SENTINEL):
        raise OverflowError("Tally reaches the overflow sentinel")
    return tuple(counts)


def tally_chunks(
    chunks: Iterable[NDArray[np.integer]],
    num_candidates: int,
    *,
    score_type: ScoreType | str = ScoreType.auto,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
) -> NDArray[CountScalar]:
    """
    Sums one pairwise matrix over any number of chunks of complete rankings.

    Taking chunks rather than one array is what keeps a national electorate off the heap: only the
    chunk in hand and the matrix itself are ever resident. The score type is resolved against the
    ballots seen so far, so an automatic matrix widens only when the running count demands it.
    """
    _resolve(implementation, backend)
    score_type = ScoreType(score_type)
    num_ballots = VoterWeight(0)
    resolved = resolve_tally_score_type(num_ballots, score_type)
    preferences = np.zeros((num_candidates, num_candidates), dtype=resolved.dtype)
    for chunk in chunks:
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != num_candidates:
            raise ValueError(f"Every chunk must be 2-D and {num_candidates} wide, got {chunk.shape}")
        num_ballots = add_saturated(num_ballots, VoterWeight(len(chunk)))
        resolved = resolve_tally_score_type(num_ballots, score_type)
        preferences = preferences.astype(resolved.dtype, copy=False)
        counted = tally_ballots(chunk, score_type=resolved, implementation=implementation, backend=backend)
        if resolved is ScoreType.saturated64:
            preferences += np.minimum(counted, SATURATED_SENTINEL - preferences)
        else:
            preferences += counted
    if resolved is ScoreType.saturated64 and np.any(preferences == SATURATED_SENTINEL):
        raise OverflowError("Tally reaches the overflow sentinel")
    return preferences


def build_pairwise_preferences(
    voter_rankings: Iterable[Sequence[int] | NDArray[np.integer]],
    num_candidates: int | None = None,
    *,
    weights: ArrayLike | None = None,
    unranked: Unranked | str = Unranked.unknown,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
) -> NDArray[CountScalar]:
    """Count possibly incomplete rankings, preserving the selected interpretation of omitted candidates."""
    rankings = [unsigned_array(ranking, CandidateIndex, ndim=1) for ranking in voter_rankings]
    if num_candidates is None:
        num_candidates = 1 + max((int(np.max(ranking)) for ranking in rankings if len(ranking)), default=-1)
    offsets = np.zeros(len(rankings) + 1, dtype=BallotOffset)
    offsets[1:] = np.cumsum([len(ranking) for ranking in rankings], dtype=BallotOffset)
    entries = np.concatenate(rankings) if rankings else np.empty(0, dtype=CandidateIndex)
    return tally_ballots(
        entries,
        offsets=offsets,
        num_candidates=num_candidates,
        weights=weights,
        unranked=unranked,
        implementation=implementation,
        backend=backend,
    )


def compute_strongest_paths(
    preferences: ArrayLike,
    *,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
    score_type: ScoreType | str = ScoreType.auto,
) -> NDArray[CountScalar]:
    """Compute Schulze paths using the requested implementation and device."""
    module = _resolve(implementation, backend)
    preferences = _preferences_matrix(preferences)
    score_type = ScoreType(score_type)
    if module is not None:
        return module.compute_strongest_paths(preferences, backend=backend, score_type=score_type.value)
    score_type = schulze.resolve_score_type(preferences, score_type)
    return compute_strongest_paths_tiled_cpu(preferences.astype(score_type.dtype, copy=False))


def compute_kemeny_ranking(
    preferences: ArrayLike,
    *,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
    score_type: ScoreType | str = ScoreType.auto,
) -> kemeny.KemenyResult:
    """Compute an exact ranking, choosing arithmetic width before allocating tables."""
    module = _resolve(implementation, backend)
    preferences = _preferences_matrix(preferences)
    score_type = ScoreType(score_type)
    if module is not None:
        ranking, score, winners, multiplicity = module.compute_kemeny_ranking(
            preferences, backend=backend, score_type=score_type.value
        )
        return kemeny.KemenyResult(ranking, score, winners, kemeny.RankingMultiplicity(multiplicity))
    return kemeny.compute_kemeny_ranking(preferences, score_type=score_type)


def enumerate_kemeny_rankings(
    preferences: ArrayLike,
    *,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
    score_type: ScoreType | str = ScoreType.auto,
) -> Generator[list[int], None, None]:
    """Yield every optimum lazily from one owned cost table built on the selected backend."""
    preferences = _preferences_matrix(preferences).copy()
    module = _resolve(implementation, backend)
    backend = Backend(backend)
    score_type = ScoreType(score_type)
    if module is None:
        costs = kemeny.compute_kemeny_costs(preferences, score_type=score_type)
    else:
        costs = module._compute_kemeny_costs(preferences, backend=backend.value, score_type=score_type.value)
    if costs[-1] == np.iinfo(costs.dtype).max:
        raise OverflowError("Kemeny optimum reaches the overflow sentinel")
    return kemeny._enumerate_kemeny_rankings(preferences, costs, (1 << len(preferences)) - 1, ())


def compute_split_cycle_winners(
    preferences: ArrayLike,
    *,
    implementation: str | None = None,
    backend: Backend | str = Backend.cpu,
    score_type: ScoreType | str = ScoreType.auto,
) -> list[int]:
    """Compute the Split Cycle winning set using the requested implementation and device."""
    module = _resolve(implementation, backend)
    preferences = _preferences_matrix(preferences)
    score_type = ScoreType(score_type)
    if module is not None:
        return module.compute_split_cycle_winners(preferences, backend=backend, score_type=score_type.value)
    if score_type is ScoreType.saturated64:
        schulze.resolve_score_type(preferences, score_type)
    score_type = schulze.resolve_score_type(positive_margins(preferences), score_type)
    return select_split_cycle_winners(
        preferences,
        compute_strongest_paths_tiled_cpu(positive_margins(preferences).astype(score_type.dtype, copy=False)),
    )


__all__ = [
    "Backend",
    "ScoreType",
    "Unranked",
    "PairwiseRelation",
    "PairwiseCounts",
    "KemenyResult",
    "RankingMultiplicity",
    "available_backends",
    "available_implementations",
    "tally_ballots",
    "tally_pairwise_relations",
    "compute_strongest_paths",
    "compute_kemeny_ranking",
    "enumerate_kemeny_rankings",
    "compute_split_cycle_winners",
    "build_pairwise_preferences",
    "complete_rankings",
    "generate_preferences",
    "populate_preferences_from_ranking",
    "positive_margins",
    "tally_chunks",
    "compute_strongest_paths_tiled_cpu",
    "compute_strongest_paths_serial",
    "compute_election_results",
    "compute_ranking_tiers",
    "select_split_cycle_winners",
]
