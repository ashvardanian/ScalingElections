"""Every method in one namespace, so the split into per-method modules stays invisible.

Ballots fold into one `N x N` matrix of pairwise counts, and Schulze, Split Cycle and
Kemeny-Young all read that matrix and nothing else. `bench.py` drives the benchmarks.
"""

import importlib

import numpy as np

import kemeny
import schulze
from ballots import (
    Backend,
    ScoreType,
    Unranked,
    build_pairwise_preferences,
    complete_rankings,
    generate_preferences,
    populate_preferences_from_ranking,
    positive_margins,
    prepare_ballots,
    tally_chunks,
    tally_ragged,
    tally_score_type,
    unsigned_array,
)
from schulze import (
    compute_election_results,
    compute_strongest_paths_serial,
    compute_strongest_paths_tiled_cpu,
    select_split_cycle_winners,
)

KemenyResult = kemeny.KemenyResult


def _module(implementation):
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


def _resolve(implementation, backend):
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


def _preferences_matrix(values):
    values = unsigned_array(values, np.uint64, ndim=2)
    if not values.shape[0] or values.shape[0] != values.shape[1]:
        raise ValueError("Preferences must be a nonempty square matrix")
    dtype = np.uint32 if np.max(values) <= np.iinfo(np.uint32).max else np.uint64
    return np.ascontiguousarray(values, dtype=dtype)


def tally_ballots(
    rankings,
    *,
    offsets=None,
    num_candidates=None,
    ranks=None,
    weights=None,
    unranked: Unranked = Unranked.unknown,
    score_type: ScoreType = ScoreType.auto,
    implementation: str | None = None,
    backend: Backend = Backend.cpu,
) -> np.ndarray:
    """Count integer-weighted ranked ballots, preserving explicit ties and omitted-candidate semantics."""
    module = _resolve(implementation, backend)
    unranked = Unranked(unranked)
    if num_candidates is not None and (
        isinstance(num_candidates, (bool, np.bool_))
        or not isinstance(num_candidates, (int, np.integer))
        or not 0 < num_candidates <= np.iinfo(np.uint32).max
    ):
        raise ValueError("num_candidates must be a positive UInt32 integer")
    if offsets is None and ranks is None and weights is None:
        dense = unsigned_array(rankings, np.uint32, ndim=2)
        n = dense.shape[1]
        if n and (num_candidates is None or num_candidates == n):
            if np.any(np.sort(dense, axis=1) != np.arange(n, dtype=np.uint32)):
                raise ValueError("Every ballot must rank each candidate exactly once")
            resolved = tally_score_type(np.array([len(dense)], dtype=np.uint64), score_type)
            if module is not None:
                return module.tally_ballots(dense, backend=backend, score_type=resolved.value)
            preferences = np.zeros((n, n), dtype=f"uint{resolved.bits}")
            for ranking in dense:
                populate_preferences_from_ranking(preferences, ranking)
            return preferences
    ids, offsets, n, ranks, weights = prepare_ballots(rankings, offsets, num_candidates, ranks, weights)
    resolved = tally_score_type(weights, score_type)
    if module is not None:
        return module.tally_ballots(
            ids,
            offsets=offsets,
            num_candidates=n,
            ranks=ranks,
            weights=weights,
            unranked=unranked.value,
            backend=backend,
            score_type=resolved.value,
        )
    result = tally_ragged(
        ids,
        offsets,
        n,
        ranks,
        weights,
        unranked is Unranked.worse,
        np.dtype(f"uint{resolved.bits}").type,
        resolved is ScoreType.saturated64,
    )
    if np.any(result == np.iinfo(np.uint64).max):
        raise OverflowError("Tally exceeds the representable count range")
    return result


def compute_strongest_paths(
    preferences,
    *,
    implementation: str | None = None,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) -> np.ndarray:
    """Compute Schulze paths using the requested implementation and device."""
    module = _resolve(implementation, backend)
    preferences = _preferences_matrix(preferences)
    requested_type = ScoreType(score_type)
    score_type = schulze.resolve_score_type(preferences, requested_type)
    if module is not None:
        return module.compute_strongest_paths(preferences, backend=backend, score_type=score_type.value)
    return compute_strongest_paths_tiled_cpu(preferences.astype(f"uint{score_type.bits}", copy=False))


def compute_kemeny_ranking(
    preferences,
    *,
    implementation: str | None = None,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) -> kemeny.KemenyResult:
    """Compute an exact ranking, choosing arithmetic width before allocating tables."""
    module = _resolve(implementation, backend)
    preferences = _preferences_matrix(preferences)
    score_type = kemeny.resolve_score_type(preferences, score_type)
    if module is not None:
        return kemeny.KemenyResult(
            *module.compute_kemeny_ranking(preferences, backend=backend, score_type=score_type.value)
        )
    return kemeny.compute_kemeny_ranking(preferences, score_type=score_type)


def compute_split_cycle_winners(
    preferences,
    *,
    implementation: str | None = None,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) -> list[int]:
    """Compute the Split Cycle winning set using the requested implementation and device."""
    module = _resolve(implementation, backend)
    preferences = _preferences_matrix(preferences)
    requested_type = ScoreType(score_type)
    if requested_type is ScoreType.saturated64:
        schulze.resolve_score_type(preferences, requested_type)
    score_type = schulze.resolve_score_type(positive_margins(preferences), requested_type)
    if module is not None:
        return module.compute_split_cycle_winners(preferences, backend=backend, score_type=requested_type.value)
    return select_split_cycle_winners(
        preferences,
        compute_strongest_paths_tiled_cpu(positive_margins(preferences).astype(f"uint{score_type.bits}", copy=False)),
    )


__all__ = [
    "Backend",
    "ScoreType",
    "Unranked",
    "KemenyResult",
    "available_backends",
    "available_implementations",
    # Ballots into the pairwise matrix every method reads.
    "build_pairwise_preferences",
    "complete_rankings",
    "generate_preferences",
    "populate_preferences_from_ranking",
    "positive_margins",
    "tally_ballots",
    "tally_chunks",
    # Schulze and Split Cycle, which share one max-min kernel over different graphs.
    "compute_split_cycle_winners",
    "compute_strongest_paths",
    "compute_strongest_paths_tiled_cpu",
    "compute_strongest_paths_serial",
    "compute_election_results",
    "select_split_cycle_winners",
    # Exact Kemeny-Young.
    "compute_kemeny_ranking",
]
