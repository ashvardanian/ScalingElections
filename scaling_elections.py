"""Benchmark suite for comparing Schulze voting algorithm implementations.

This module provides benchmarking tools to compare different implementations of the
Schulze voting method across various backends: serial Numba, parallel Numba with tiling,
OpenMP (CPU), and CUDA (GPU). It includes utilities for building pairwise preference
matrices from voter rankings and computing strongest paths using the Floyd-Warshall
algorithm with block-parallel optimizations.

Usage:
    uv run scaling_elections.py --num-candidates 4096 --num-voters 4096 --run-cpu --run-gpu

See: https://github.com/ashvardanian/ScalingElections
"""
from typing import Sequence, Tuple, List
import warnings

import numpy as np
from numba import njit, prange, get_num_threads

from scaling_elections import log_gpus  # type: ignore
from scaling_elections import compute_strongest_paths  # type: ignore

# Suppress Numba TBB threading layer warnings
warnings.filterwarnings("ignore", message=".*TBB threading layer.*")


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


def format_time(elapsed_sec: float) -> str:
    """Format time with appropriate unit (ms or s)."""
    elapsed_ms = elapsed_sec * 1000
    if elapsed_ms < 1000:
        return f"{int(elapsed_ms)} ms"
    else:
        return f"{elapsed_sec:.2f} s"


def format_throughput(cells_per_sec: float) -> str:
    """Format throughput with appropriate unit (T/G/M cells³/s)."""
    if cells_per_sec >= 1e12:
        return f"{cells_per_sec / 1e12:.1f} Tcells³/s"
    elif cells_per_sec >= 1e9:
        return f"{cells_per_sec / 1e9:.1f} Gcells³/s"
    elif cells_per_sec >= 1e6:
        return f"{cells_per_sec / 1e6:.1f} Mcells³/s"
    else:
        return f"{cells_per_sec / 1e3:.1f} Kcells³/s"


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
        "--run-cpu",
        action="store_true",
        help="Run CPU implementations (Numba, OpenMP)",
    )
    parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="Run GPU implementation (CUDA)",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Skip serial baseline",
    )
    parser.add_argument(
        "--cpu-tile-size",
        type=int,
        default=16,
        help="CPU tile size for tiling optimization",
    )
    parser.add_argument(
        "--gpu-tile-size",
        type=int,
        default=32,
        help="GPU tile size for tiling optimization",
    )
    args = parser.parse_args()

    cpu_tile_size = args.cpu_tile_size
    gpu_tile_size = args.gpu_tile_size
    num_voters = args.num_voters
    num_candidates = args.num_candidates
    run_serial = not args.no_serial

    compute_strongest_paths_cuda = lambda x: compute_strongest_paths(
        x,
        allow_gpu=True,
        allow_tma=False,
        tile_size=gpu_tile_size,
    )
    compute_strongest_paths_hopper = lambda x: compute_strongest_paths(
        x,
        allow_gpu=True,
        allow_tma=True,
        tile_size=gpu_tile_size,
    )
    compute_strongest_paths_openmp = lambda x: compute_strongest_paths(
        x,
        allow_gpu=False,
        allow_tma=False,
        tile_size=cpu_tile_size,
    )
    compute_strongest_paths_numba_tiled = (
        lambda x: compute_strongest_paths_numba_parallel(
            x,
            tile_size=cpu_tile_size,
        )
    )

    # Print header
    print("=== Schulze Voting Algorithm (Python) ===")
    print()

    # Print GPU info if available
    try:
        log_gpus()
    except Exception as e:
        print(f"✗ Could not detect GPU: {e}")

    # Print configuration
    print("Configuration:")
    voters_str = f"{num_voters:,}" if num_voters > 0 else "random"
    print(f"  Problem size: {num_candidates:,} candidates × {voters_str} voters")
    print(f"  CPU tile: {cpu_tile_size} × {cpu_tile_size}")
    if args.run_gpu:
        print(f"  GPU tile: {gpu_tile_size} × {gpu_tile_size}")
    print(f"  CPU threads: {get_num_threads()}")
    print()

    # Generate random voter rankings
    print("Generating preferences...")
    if num_voters == 0:
        preferences = np.random.randint(
            0, num_candidates, (num_candidates, num_candidates)
        ).astype(np.uint32)
    else:
        voter_rankings = [
            np.random.permutation(num_candidates) for _ in range(num_voters)
        ]
        preferences = build_pairwise_preferences(voter_rankings)

    # Warm-up: run all functions on tiny inputs first to avoid JIT costs
    sub_preferences = preferences[: num_candidates // 8, : num_candidates // 8]

    # Benchmarking section
    print()
    print("─── Benchmarking ───────────────────────────────────")
    print()

    # Always run serial first as baseline (unless --no-serial)
    serial_result = None
    if run_serial:
        print("→ Serial (Numba)")
        try:
            start_time = time.time()
            sub_serial_result = compute_strongest_paths_numba_serial(sub_preferences)
            elapsed_time = time.time() - start_time
            print(f"  Warm-up: {format_time(elapsed_time)}")

            start_time = time.time()
            serial_result = compute_strongest_paths_numba_serial(preferences)
            elapsed_time = time.time() - start_time
            throughput = num_candidates**3 / elapsed_time
            print(f"  Run:     {format_time(elapsed_time)} │ {format_throughput(throughput)}")
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")

        print()

    # Run other implementations and validate against serial
    for name, wanted, callback in [
        ("Tiled CPU (Numba)", args.run_cpu, compute_strongest_paths_numba_tiled),
        ("Tiled CPU (C++ with OpenMP)", args.run_cpu, compute_strongest_paths_openmp),
        ("Tiled GPU (CUDA)", args.run_gpu, compute_strongest_paths_cuda),
        ("Tiled GPU (CUDA + TMA)", args.run_gpu, compute_strongest_paths_hopper),
    ]:
        if not wanted:
            continue

        print(f"→ {name}")

        # Warm-up run
        try:
            start_time = time.time()
            sub_result = callback(sub_preferences)
            elapsed_time = time.time() - start_time
            print(f"  Warm-up: {format_time(elapsed_time)}")
        except Exception as e:
            print(f"  ✗ Warm-up failed: {e}")
            print()
            continue

        # Main benchmark run
        try:
            start_time = time.time()
            result = callback(preferences)
            elapsed_time = time.time() - start_time
            throughput = num_candidates**3 / elapsed_time
            print(f"  Run:     {format_time(elapsed_time)} │ {format_throughput(throughput)}")

            # Validate against serial baseline if available
            if serial_result is not None:
                if np.array_equal(result, serial_result):
                    print(f"  ✓ Results validated")
                else:
                    print(f"  ✗ Results don't match baseline!")
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")

        print()

    # Determine the winner and ranking (use serial if available, otherwise first result)
    if serial_result is not None:
        result_for_winner = serial_result
    else:
        # Use a small sample for winner determination
        result_for_winner = compute_strongest_paths_numba_serial(sub_preferences)

    candidates = list(range(result_for_winner.shape[0]))
    winner, ranking = get_winner_and_ranking(candidates, result_for_winner)

    # Print election results
    print("─── Election Results ───────────────────────────────")
    print()
    print(f"  Winner: Candidate #{winner}")
    if len(ranking) >= 5:
        print(f"  Top 5:  #{ranking[0]}, #{ranking[1]}, #{ranking[2]}, #{ranking[3]}, #{ranking[4]}")
    print()
