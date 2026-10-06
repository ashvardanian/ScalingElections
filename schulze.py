"""Widest-path recurrences behind the Schulze method and Split Cycle.

Both methods close the pairwise graph under a max-min recurrence; they differ only in the
edge weights fed to it, winning votes for Schulze and positive margins for Split Cycle.
"""

import warnings

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

from ballots import ScoreType

warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


TILE_SIZE = 32
"""The tile edge every backend shares, derived from the 1024-thread GPU block limit: one thread per cell of a 32 by 32 tile."""


def positive_margins(preferences: NDArray[np.integer]) -> NDArray[np.integer]:
    """
    Rewrites pairwise counts so the strongest-paths kernel closes over margins.

    The kernel keeps `preferences[i, j]` when it exceeds `preferences[j, i]` and zeroes it
    otherwise, so feeding it the clipped margin makes it compute the widest paths of the
    positive-margin graph without any change to the kernel itself.
    """
    return preferences - np.minimum(preferences, preferences.T)


def resolve_score_type(preferences: NDArray[np.integer], score_type: ScoreType = ScoreType.auto) -> ScoreType:
    """Select arithmetic that represents every path edge; max-min never increases its maximum."""
    score_type = ScoreType(score_type)
    if score_type is ScoreType.saturated64 and np.any(preferences == np.iinfo(np.uint64).max):
        raise OverflowError("Schulze input reaches the overflow sentinel")
    if score_type is ScoreType.auto:
        score_type = (
            ScoreType.uint32 if np.max(preferences) <= np.iinfo(ScoreType.uint32.dtype).max else ScoreType.uint64
        )
    if np.max(preferences) > np.iinfo(score_type.dtype).max:
        raise OverflowError("Schulze edge exceeds the selected arithmetic type")
    return score_type


# region Serial


@njit
def winning_votes_graph[Scalar: np.integer](preferences: NDArray[Scalar]) -> NDArray[Scalar]:
    """Seeds the strongest paths with each pair's winning side, leaving losses, ties and the diagonal at zero."""
    num_candidates = preferences.shape[0]
    graph = np.zeros((num_candidates, num_candidates), dtype=preferences.dtype)
    for row in range(num_candidates):
        for column in range(num_candidates):
            if row != column and preferences[row, column] > preferences[column, row]:
                graph[row, column] = preferences[row, column]
    return graph


@njit
def compute_strongest_paths_serial[Scalar: np.integer](preferences: NDArray[Scalar]) -> NDArray[Scalar]:
    """
    Computes the widest path strengths using the Schulze method.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    strongest_paths = winning_votes_graph(preferences)
    for pivot in range(num_candidates):
        for source in range(num_candidates):
            if source != pivot:
                for target in range(num_candidates):
                    if source != target and pivot != target:
                        strongest_paths[source, target] = max(
                            strongest_paths[source, target],
                            min(
                                strongest_paths[source, pivot],
                                strongest_paths[pivot, target],
                            ),
                        )

    return strongest_paths


# endregion Serial


# region Tiled Parallel


@njit
def process_tile_cpu(
    paths: NDArray[np.integer],
    tile_row_start: int,
    tile_column_start: int,
    pivot_start: int,
    tile_size: int = TILE_SIZE,
) -> None:
    """
    Relaxes the tile at (`tile_row_start`, `tile_column_start`) of `paths` through every pivot of
    the tile column `pivot_start`, reading the to-pivot and from-pivot tiles from the same matrix.
    """
    # `njit` compiles with `boundscheck=False`, so the tail of a non-divisible matrix has
    # to be clamped here rather than trapped on access.
    num_candidates = paths.shape[0]
    pivot_extent = min(tile_size, num_candidates - pivot_start)
    row_extent = min(tile_size, num_candidates - tile_row_start)
    column_extent = min(tile_size, num_candidates - tile_column_start)

    for pivot in range(pivot_extent):
        pivot_index = pivot_start + pivot
        for row in range(row_extent):
            row_index = tile_row_start + row
            if row_index == pivot_index:
                continue
            to_pivot = paths[row_index, pivot_index]
            for column in range(column_extent):
                column_index = tile_column_start + column
                if row_index == column_index or pivot_index == column_index:
                    continue
                through_pivot = min(to_pivot, paths[pivot_index, column_index])
                if through_pivot > paths[row_index, column_index]:
                    paths[row_index, column_index] = through_pivot


@njit(parallel=True)
def compute_strongest_paths_tiled_cpu[Scalar: np.integer](
    preferences: NDArray[Scalar],
    tile_size: int = TILE_SIZE,
) -> NDArray[Scalar]:
    """
    Computes the widest path strengths in cache-sized tiles, three dependency phases per pivot tile,
    parallelizing the tiles within each phase.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]
    strongest_paths = winning_votes_graph(preferences)
    tiles_count = (num_candidates + tile_size - 1) // tile_size
    for pivot_tile in range(tiles_count):
        pivot_start = pivot_tile * tile_size
        process_tile_cpu(strongest_paths, pivot_start, pivot_start, pivot_start, tile_size)

        for row_tile in prange(tiles_count):
            if row_tile != pivot_tile:
                process_tile_cpu(strongest_paths, row_tile * tile_size, pivot_start, pivot_start, tile_size)

        for column_tile in prange(tiles_count):
            if column_tile != pivot_tile:
                process_tile_cpu(strongest_paths, pivot_start, column_tile * tile_size, pivot_start, tile_size)

        for row_tile in prange(tiles_count):
            if row_tile == pivot_tile:
                continue
            for column_tile in range(tiles_count):
                if column_tile != pivot_tile:
                    process_tile_cpu(
                        strongest_paths, row_tile * tile_size, column_tile * tile_size, pivot_start, tile_size
                    )

    return strongest_paths


# endregion Tiled Parallel


# region Winners


def select_split_cycle_winners(preferences: NDArray[np.integer], margin_paths: NDArray[np.integer]) -> list[int]:
    """
    Determines the Split Cycle winners, which are the candidates nobody defeats.

    Holliday and Pacuit's Lemma 3.17: `a` defeats `b` when the margin of `a` over `b` is
    positive and exceeds the strength of the widest path from `b` back to `a`. Unlike
    Schulze, which this repository runs on winning votes, Split Cycle is defined on margins.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    margins = positive_margins(preferences)
    defeats = (margins > 0) & (margins > margin_paths.T)
    return [candidate for candidate in range(preferences.shape[0]) if not defeats[:, candidate].any()]


def compute_election_results(
    candidates: list[int],
    strongest_paths: NDArray[np.integer],
) -> tuple[list[int], list[int]]:
    """
    Returns every undefeated candidate and a deterministic representative ranking from the strongest paths matrix.

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
    winners = [
        candidates[index]
        for index in range(num_candidates)
        if not np.any(strongest_paths[:, index] > strongest_paths[index, :])
    ]
    ranked_candidates = [candidates[index] for index in ranking_indices]

    return winners, ranked_candidates


def compute_ranking_tiers(candidates: list[int], strongest_paths: NDArray[np.integer]) -> list[list[int]]:
    """Peel undefeated fronts; candidates in one tier need not express pairwise indifference."""
    if strongest_paths.shape != (len(candidates), len(candidates)) or len(set(candidates)) != len(candidates):
        raise ValueError("Distinct candidates must match the square strongest-path matrix")
    defeats = strongest_paths > strongest_paths.T
    remaining = np.ones(len(candidates), dtype=bool)
    incoming = np.sum(defeats, axis=0)
    tiers = []
    while np.any(remaining):
        front = np.flatnonzero(remaining & (incoming == 0))
        if not len(front):
            raise ValueError("Strongest-path defeats must be acyclic")
        tiers.append([candidates[index] for index in front])
        remaining[front] = False
        incoming -= np.sum(defeats[front], axis=0)
    return tiers


# endregion Winners
