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
    build_pairwise_preferences,
    complete_rankings,
    generate_preferences,
    populate_preferences_from_ranking,
    positive_margins,
    tally_chunks,
)
from schulze import (
    compute_election_results,
    compute_strongest_paths_serial,
    compute_strongest_paths_tiled_cpu,
    select_split_cycle_winners,
)


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


def _uint32_matrix(values, *, square):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] == 0 or (square and values.shape[0] != values.shape[1]):
        raise ValueError(
            "Preferences must be a nonempty square matrix"
            if square
            else "Rankings must be a two-dimensional array with at least one candidate"
        )
    if values.dtype.kind not in "iu":
        raise TypeError("Entries must be integers")
    if np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max):
        raise OverflowError("Entries must fit UInt32")
    return np.ascontiguousarray(values, dtype=np.uint32)


def tally_ballots(rankings, *, implementation: str | None = None, backend: Backend = Backend.cpu) -> np.ndarray:
    """Count complete rankings using the requested implementation and device."""
    module = _resolve(implementation, backend)
    rankings = _uint32_matrix(rankings, square=False)
    n = rankings.shape[1]
    if rankings.shape[0] > np.iinfo(np.uint32).max:
        raise OverflowError("Ballot count must fit UInt32")
    if np.any(np.sort(rankings, axis=1) != np.arange(n, dtype=np.uint32)):
        raise ValueError("Every ballot must rank each candidate exactly once")
    if backend == "gpu" and module is not None and module.__name__ == "scalingelections_mojo" and n > 64:
        raise ValueError("The GPU tally supports at most 64 candidates")
    if module is not None:
        return module.tally_ballots(rankings, backend=backend)
    preferences = np.zeros((n, n), dtype=np.uint32)
    for ranking in rankings:
        populate_preferences_from_ranking(preferences, ranking)
    return preferences


def compute_strongest_paths(
    preferences,
    *,
    implementation: str | None = None,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) -> np.ndarray:
    """Compute Schulze paths using the requested implementation and device."""
    module = _resolve(implementation, backend)
    preferences = _uint32_matrix(preferences, square=True)
    requested_type = ScoreType(score_type)
    score_type = schulze.resolve_score_type(preferences, requested_type)
    if module is not None:
        return module.compute_strongest_paths(preferences, backend=backend, score_type=requested_type.value)
    return compute_strongest_paths_tiled_cpu(preferences.astype(score_type.value, copy=False)).astype(
        np.uint32, copy=False
    )


def compute_kemeny_ranking(
    preferences,
    *,
    implementation: str | None = None,
    backend: Backend = Backend.cpu,
    score_type: ScoreType = ScoreType.auto,
) -> tuple[list[int], int]:
    """Compute an exact ranking, choosing arithmetic width before allocating tables."""
    module = _resolve(implementation, backend)
    preferences = _uint32_matrix(preferences, square=True)
    score_type = kemeny.resolve_score_type(preferences, score_type)
    if module is not None:
        return module.compute_kemeny_ranking(preferences, backend=backend, score_type=score_type.value)
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
    preferences = _uint32_matrix(preferences, square=True)
    requested_type = ScoreType(score_type)
    score_type = schulze.resolve_score_type(positive_margins(preferences), requested_type)
    if module is not None:
        return module.compute_split_cycle_winners(preferences, backend=backend, score_type=requested_type.value)
    return select_split_cycle_winners(
        preferences,
        compute_strongest_paths_tiled_cpu(positive_margins(preferences).astype(score_type.value, copy=False)),
    )


__all__ = [
    "Backend",
    "ScoreType",
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
