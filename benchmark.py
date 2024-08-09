import random
from typing import Sequence, Tuple, List

import numpy as np
from numba import njit, prange, get_num_threads

from scaling_democracy import log_gpus
from scaling_democracy import compute_strongest_paths


@njit
def populate_preferences_from_ranking(preferences: np.ndarray, ranking: np.ndarray):
    """
    Populates the preference matrix based on a ranking of candidates.
    The candidate must be represented as monotonic integers starting from 0.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    for i, preferred in enumerate(ranking):
        for opponent in ranking[i + 1 :]:
            preferences[preferred, opponent] += 1


def build_pairwise_preferences(voter_rankings: Sequence[np.ndarray]) -> np.ndarray:
    """
    For every voter in the population, receives a (potentially incomplete) ranking of candidates,
    and builds a square preference matrix based on the rankings. Every cell (i, j) in the matrix
    contains the number of voters who prefer candidate i to candidate j.
    The candidate must be represented as monotonic integers starting from 0.
    If some candidates aren't included in a specific ranking, to break ties between them, random
    ballots are generated.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(m * n^2), where n is the number of candidates and m is the number of voters.
    """
    # The number of candidates is the maximum candidate index in the rankings plus one.
    count_candidates = 1
    for ranking in voter_rankings:
        count_candidates = max(count_candidates, np.max(ranking) + 1)

    # Initialize the preference matrix
    preferences = np.zeros((count_candidates, count_candidates), dtype=np.uint32)

    # Process each voter's ranking
    for ranking in voter_rankings:

        # We may be dealing with incomplete rankings
        if len(ranking) != count_candidates:
            # Create a mask for integers from 0 to N
            full_mask = np.ones(count_candidates, dtype=bool)
            # Mark the integers present in the incomplete array
            full_mask[ranking] = False
            # Find the missing integers
            missing_integers = np.nonzero(full_mask)[0]
            # Append the missing integers to the incomplete array
            ranking = np.append(ranking, missing_integers)

        # By now the ranking should be complete
        populate_preferences_from_ranking(preferences, ranking)

    return preferences


@njit
def compute_strongest_paths_numba_serial(preferences: np.ndarray) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method.

    Space complexity: O(n^2), where n is the number of candidates.
    Time complexity: O(n^3), where n is the number of candidates.
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    # assert preferences.dtype == np.uint32, f"Wrong type: {preferences.dtype}"
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.uint32)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                if preferences[i, j] > preferences[j, i]:
                    strongest_paths[i, j] = preferences[i, j]
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                for k in range(num_candidates):
                    if i != k and j != k:
                        strongest_paths[j, k] = max(
                            strongest_paths[j, k],
                            min(strongest_paths[j, i], strongest_paths[i, k]),
                        )

    return strongest_paths


@njit
def compute_strongest_paths_tile_numba(
    c: np.ndarray,
    c_row: int,
    c_col: int,
    a: np.ndarray,
    a_row: int,
    a_col: int,
    b: np.ndarray,
    b_row: int,
    b_col: int,
    tile_size: int = 16,
):
    """
    In-place computation of the widest path path using the Schulze method with tiling for better cache utilization.
    For input of size (n x n), would perform (n) iterations of quadratic complexity each.

    Time complexity: O(n^3), where n is the tile size.
    Space complexity: O(n^2), where n is the tile size.
    """

    for k in range(tile_size):
        for i in range(tile_size):
            for j in range(tile_size):
                if (
                    (c_row + i != c_col + j)
                    and (a_row + i != a_col + k)
                    and (b_row + k != b_col + j)
                ):
                    replacement = min(a[a_row + i, a_col + k], b[b_row + k, b_col + j])
                    if replacement > c[c_row + i, c_col + j]:
                        c[c_row + i, c_col + j] = replacement


@njit(parallel=True)
def compute_strongest_paths_numba_parallel(
    preferences: np.ndarray,
    tile_size: int = 16,
) -> np.ndarray:
    """
    Computes the widest path strengths using the Schulze method with tiling for better cache utilization.
    This implementation not only parallelizes the outer loop but also tiles the computation, to maximize
    the utilization of CPU caches.

    Space complexity:
    Time complexity:
    """
    num_candidates = preferences.shape[0]

    # Initialize the strongest paths matrix
    # assert preferences.dtype == np.uint32, f"Wrong type: {preferences.dtype}"
    strongest_paths = np.zeros((num_candidates, num_candidates), dtype=np.uint32)

    # Step 1: Populate the strongest paths matrix based on direct comparisons
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                if preferences[i, j] > preferences[j, i]:
                    strongest_paths[i, j] = preferences[i, j]
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Compute the strongest paths using Floyd-Warshall-like algorithm with tiling
    tiles_count = (num_candidates + tile_size - 1) // tile_size
    for k in range(tiles_count):
        # Dependent phase
        k_start = k * tile_size

        # f(S_kk, S_kk, S_kk)
        compute_strongest_paths_tile_numba(
            strongest_paths,
            k_start,
            k_start,
            strongest_paths,
            k_start,
            k_start,
            strongest_paths,
            k_start,
            k_start,
            tile_size,
        )

        # Partially dependent phase (first of two)
        for i in prange(tiles_count):
            if i == k:
                continue
            i_start = i * tile_size
            # f(S_ik, S_ik, S_kk)
            compute_strongest_paths_tile_numba(
                strongest_paths,
                i_start,
                k_start,
                strongest_paths,
                i_start,
                k_start,
                strongest_paths,
                k_start,
                k_start,
                tile_size,
            )

        # Partially dependent phase (second of two)
        for j in prange(tiles_count):
            if j == k:
                continue
            j_start = j * tile_size
            # f(S_kj, S_kk, S_kj)
            compute_strongest_paths_tile_numba(
                strongest_paths,
                k_start,
                j_start,
                strongest_paths,
                k_start,
                k_start,
                strongest_paths,
                k_start,
                j_start,
                tile_size,
            )

        # Independent phase
        for i in prange(tiles_count):
            if i == k:
                continue
            i_start = i * tile_size
            for j in range(tiles_count):
                if j == k:
                    continue
                j_start = j * tile_size
                # f(S_ij, S_ik, S_kj)
                compute_strongest_paths_tile_numba(
                    strongest_paths,
                    i_start,
                    j_start,
                    strongest_paths,
                    i_start,
                    k_start,
                    strongest_paths,
                    k_start,
                    j_start,
                    tile_size,
                )

    return strongest_paths


def get_winner_and_ranking(
    candidates: list,
    strongest_paths: np.ndarray,
) -> Tuple[int, List[int]]:
    """
    Determines the winner and the overall ranking of candidates based on the strongest paths matrix.

    Space complexity: O(n), where n is the number of candidates.
    Time complexity: O(n^2), where n is the number of candidates.
    """
    num_candidates = len(candidates)
    wins = np.zeros(num_candidates, dtype=int)

    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j and strongest_paths[i, j] > strongest_paths[j, i]:
                wins[i] += 1

    ranking_indices = sorted(range(num_candidates), key=lambda x: wins[x], reverse=True)
    winner = candidates[ranking_indices[0]]
    ranked_candidates = [candidates[i] for i in ranking_indices]

    return winner, ranked_candidates


# Benchmark and comparison code remains the same
if __name__ == "__main__":
    import time
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the Schulze method")
    parser.add_argument(
        "--num-voters",
        type=int,
        default=0,
        help="Number of voters in the population, 0 for random preference matrix",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=256,
        help="Number of candidates in the election",
    )
    parser.add_argument(
        "--run-serial",
        action="store_true",
        help="Run the serial version of the code",
    )
    parser.add_argument(
        "--run-numba",
        action="store_true",
        help="Run the serial version of the code",
    )
    parser.add_argument(
        "--run-openmp",
        action="store_true",
        help="Run the serial version of the code",
    )
    parser.add_argument(
        "--run-cuda",
        action="store_true",
        help="Run the serial version of the code",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Tile size for the tiling optimization",
    )
    args = parser.parse_args()

    tile_size = args.tile_size
    num_voters = args.num_voters
    num_candidates = args.num_candidates

    compute_strongest_paths_cuda = lambda x: compute_strongest_paths(
        x,
        allow_gpu=True,
        allow_tma=False,
        tile_size=tile_size,
    )
    compute_strongest_paths_h100 = lambda x: compute_strongest_paths(
        x,
        allow_gpu=True,
        allow_tma=False,
        tile_size=tile_size,
    )
    compute_strongest_paths_openmp = lambda x: compute_strongest_paths(
        x,
        allow_gpu=False,
        allow_tma=False,
        tile_size=tile_size,
    )
    compute_strongest_paths_numba_tiled = (
        lambda x: compute_strongest_paths_numba_parallel(
            x,
            tile_size=tile_size,
        )
    )

    # Generate random voter rankings
    try:
        log_gpus()
    except Exception as e:
        print(f"✘ Error: Could not log GPUs: {e}")
    print(
        f"Generating {num_voters:,} random voter rankings with {num_candidates:,} candidates"
    )

    # To simplify the benchmark, let's generate a simple square matrix
    if num_voters == 0:
        preferences = np.random.randint(
            0, num_candidates, (num_candidates, num_candidates)
        ).astype(np.uint32)
    else:
        voter_rankings = [
            np.random.permutation(num_candidates) for _ in range(num_voters)
        ]
        preferences = build_pairwise_preferences(voter_rankings)
    print(f"Generated voter rankings, proceeding with {get_num_threads()} threads")

    # To avoid cold-start and aggregating JIT costs, let's run all functions on tiny inputs first
    sub_preferences = preferences[: num_candidates // 8, : num_candidates // 8]
    sub_preferences_baseline = compute_strongest_paths_numba_serial(sub_preferences)

    for name, wanted, callback in [
        ("Numba", args.run_numba, compute_strongest_paths_numba_tiled),
        ("CUDA", args.run_cuda, compute_strongest_paths_cuda),
        ("CUDA with TMA", args.run_cuda, compute_strongest_paths_h100),
        ("OpenMP", args.run_openmp, compute_strongest_paths_openmp),
        ("Serial", args.run_serial, compute_strongest_paths_numba_serial),
    ]:
        if not wanted:
            print(f"↷ Skipping {name}")
            continue

        try:
            start_time = time.time()
            sub_preferences_result = callback(sub_preferences)
            elapsed_time = time.time() - start_time
        except Exception as e:
            print(f"✘ Error: {name} raised an exception: {e}")
            continue

        print(f"{name} warm-up: {elapsed_time:.4f} secs")
        if not np.array_equal(sub_preferences_result, sub_preferences_baseline):
            print(f"✘ Error: {name} returned different results from Numba baseline")
        else:
            print(f"✔ {name} returned correct results")

        # Run the benchmark
        start_time = time.time()
        callback(preferences)
        elapsed_time = time.time() - start_time
        throughput = num_candidates**3 / elapsed_time
        print(f"{name}: {elapsed_time:.4f} secs, {throughput:,.2f} cells^3/sec")

    # Determine the winner and ranking for the final method (they should be the same for all methods)
    candidates = list(range(sub_preferences.shape[0]))
    winner, ranking = get_winner_and_ranking(candidates, sub_preferences)

    # Print the results
    print(f"    Winner is {winner}")
    print(f"    Ranked {len(ranking)} candidates")
