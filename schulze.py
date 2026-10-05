"""Widest-path recurrences behind the Schulze method and Split Cycle.

Both methods close the pairwise graph under a max-min recurrence; they differ only in the
edge weights fed to it, winning votes for Schulze and positive margins for Split Cycle.
"""

import warnings

import numpy as np
from numba import njit, prange

from ballots import ScoreType, positive_margins

# Suppress Numba TBB threading layer warnings
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


TILE_SIZE = 32
"""The tile edge every backend is compiled for, matching `tile_size_k` in `types.cuh`."""


def resolve_score_type(preferences: np.ndarray, score_type: ScoreType = ScoreType.auto) -> ScoreType:
    """Select arithmetic that represents every path edge; max-min never increases its maximum."""
    score_type = ScoreType(score_type)
    if score_type is ScoreType.auto:
        score_type = ScoreType.uint32
    if np.max(preferences) > np.iinfo(np.dtype(score_type.value)).max:
        raise OverflowError("Schulze edge exceeds the selected arithmetic type")
    return score_type


# region Serial


@njit
def compute_strongest_paths_serial(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=preferences.dtype)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for source in range(num_candidates):
        for target in range(num_candidates):
            if source != target:
                if preferences[source, target] > preferences[target, source]:
                    strongest_paths[source, target] = preferences[source, target]
                else:
                    strongest_paths[source, target] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm
    for pivot in range(num_candidates):
        for source in range(num_candidates):
            if source != pivot:
                for target in range(num_candidates):
                    if source != target and pivot != target:
                        strongest_paths[source, target] = max(
                            strongest_paths[source, target],
                            min(strongest_paths[source, pivot], strongest_paths[pivot, target]),
                        )

    return strongest_paths


# endregion Serial


# region Tiled Parallel


@njit
def process_tile_cpu(
    output: np.ndarray,
    output_row: int,
    output_column: int,
    left: np.ndarray,
    left_row: int,
    left_column: int,
    right: np.ndarray,
    right_row: int,
    right_column: int,
    tile_size: int = TILE_SIZE,
):
    """
    In-place computation of the widest path path using the Schulze method with tiling for better cache utilization.
    For input of size (n x n), would perform (n) iterations of quadratic complexity each.

    Time complexity: O(n^3), where n is the tile size.
    Space complexity: O(n^2), where n is the tile size.
    """

    # `njit` compiles with `boundscheck=False`, so the tail of a non-divisible matrix has
    # to be clamped here rather than trapped on access.
    num_candidates = output.shape[0]
    pivot_extent = min(tile_size, num_candidates - max(left_column, right_row))
    row_extent = min(tile_size, num_candidates - max(output_row, left_row))
    column_extent = min(tile_size, num_candidates - max(output_column, right_column))

    for pivot in range(pivot_extent):
        for row in range(row_extent):
            for column in range(column_extent):
                if (
                    (output_row + row != output_column + column)
                    and (left_row + row != left_column + pivot)
                    and (right_row + pivot != right_column + column)
                ):
                    replacement = min(
                        left[left_row + row, left_column + pivot], right[right_row + pivot, right_column + column]
                    )
                    if replacement > output[output_row + row, output_column + column]:
                        output[output_row + row, output_column + column] = replacement


@njit(parallel=True)
def compute_strongest_paths_tiled_cpu(
    preferences: np.ndarray,
    tile_size: int = TILE_SIZE,
) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method with tiling for better cache utilization.
    This implementation not only parallelizes the outer loop but also tiles the computation, to maximize
    the utilization of CPU caches.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=preferences.dtype)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for source in range(num_candidates):
        for target in range(num_candidates):
            if source != target:
                if preferences[source, target] > preferences[target, source]:
                    strongest_paths[source, target] = preferences[source, target]
                else:
                    strongest_paths[source, target] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm with tiling
    tiles_count = (num_candidates + tile_size - 1) // tile_size
    for pivot_tile in range(tiles_count):
        # Dependent phase
        pivot_start = pivot_tile * tile_size

        # f(S_kk, S_kk, S_kk)
        process_tile_cpu(
            strongest_paths,
            pivot_start,
            pivot_start,
            strongest_paths,
            pivot_start,
            pivot_start,
            strongest_paths,
            pivot_start,
            pivot_start,
            tile_size,
        )

        # Partially dependent phase (first of two)
        for row_tile in prange(tiles_count):
            if row_tile == pivot_tile:
                continue
            row_start = row_tile * tile_size
            # f(S_ik, S_ik, S_kk)
            process_tile_cpu(
                strongest_paths,
                row_start,
                pivot_start,
                strongest_paths,
                row_start,
                pivot_start,
                strongest_paths,
                pivot_start,
                pivot_start,
                tile_size,
            )

        # Partially dependent phase (second of two)
        for column_tile in prange(tiles_count):
            if column_tile == pivot_tile:
                continue
            column_start = column_tile * tile_size
            # f(S_kj, S_kk, S_kj)
            process_tile_cpu(
                strongest_paths,
                pivot_start,
                column_start,
                strongest_paths,
                pivot_start,
                pivot_start,
                strongest_paths,
                pivot_start,
                column_start,
                tile_size,
            )

        # Independent phase
        for row_tile in prange(tiles_count):
            if row_tile == pivot_tile:
                continue
            row_start = row_tile * tile_size
            for column_tile in range(tiles_count):
                if column_tile == pivot_tile:
                    continue
                column_start = column_tile * tile_size
                # f(S_ij, S_ik, S_kj)
                process_tile_cpu(
                    strongest_paths,
                    row_start,
                    column_start,
                    strongest_paths,
                    row_start,
                    pivot_start,
                    strongest_paths,
                    pivot_start,
                    column_start,
                    tile_size,
                )

    return strongest_paths


# endregion Tiled Parallel


# region Winners


def select_split_cycle_winners(preferences: np.ndarray, margin_paths: np.ndarray) -> list[int]:
    """
    Determines the Split Cycle winners, which are the candidates nobody defeats.

    Holliday and Pacuit's Lemma 3.17: `a` defeats `b` when the margin of `a` over `b` is
    positive and exceeds the strength of the widest path from `b` back to `a`. Unlike
    Schulze, which this repository runs on winning votes, Split Cycle is defined on margins.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    margins = positive_margins(preferences)
    defeats = (margins > 0) & (margins > margin_paths.astype(np.int64).T)
    return [candidate for candidate in range(preferences.shape[0]) if not defeats[:, candidate].any()]


def compute_election_results(
    candidates: list[int],
    strongest_paths: np.ndarray,
) -> tuple[int, list[int]]:
    """
    Determines the winner and the overall ranking of candidates based on the strongest paths matrix.

    Space complexity: O(n), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    num_candidates = len(candidates)
    wins = np.zeros(num_candidates, dtype=int)

    for source in range(num_candidates):
        for target in range(num_candidates):
            if source != target and strongest_paths[source, target] > strongest_paths[target, source]:
                wins[source] += 1

    ranking_indices = sorted(range(num_candidates), key=lambda candidate: wins[candidate], reverse=True)
    winner = candidates[ranking_indices[0]]
    ranked_candidates = [candidates[index] for index in ranking_indices]

    return winner, ranked_candidates


# endregion Winners
