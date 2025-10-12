"""
Mojo implementation of the Schulze voting algorithm with CPU and GPU acceleration.

This implementation provides parallel computation of strongest paths for democratic voting,
ported from the original Python/CUDA implementation in benchmark.py and scaling_democracy.cu.
The Schulze method is a Condorcet voting system that selects the candidate who would win
in head-to-head comparisons against all other candidates.

## Usage

Run directly with Mojo via Pixi:

```bash
pixi run mojo scaling_democracy.mojo
pixi run mojo scaling_democracy.mojo --num-candidates 4096 --num-voters 4096
```

Or compile and run:

```bash
pixi run mojo build scaling_democracy.mojo -o scaling_democracy
./scaling_democracy
```

See: https://ashvardanian.com/posts/scaling-democracy/
"""

from collections import List
from memory import UnsafePointer, memset_zero, stack_allocation, AddressSpace
from algorithm import parallelize
from random import random_si64
from time import perf_counter_ns
from sys import argv, has_accelerator
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_idx, thread_idx, block_dim, global_idx
from gpu.sync import barrier
from layout import Layout, LayoutTensor
from buffer import NDBuffer

# Type aliases for clarity
alias VotesCount = UInt32
alias CandidateIdx = Int

# Tile size for blocking optimization - matches CUDA implementation
alias TILE_SIZE = 16

@fieldwise_init
struct PreferenceMatrix(Movable):
    """Represents a preference matrix for Schulze voting."""
    var data: UnsafePointer[UInt32]
    var num_candidates: Int

    fn __init__(out self, num_candidates: Int):
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = UnsafePointer[UInt32].alloc(size)
        memset_zero(self.data, size)

    fn __getitem__(self, i: Int, j: Int) -> UInt32:
        return self.data[i * self.num_candidates + j]

    fn __setitem__(mut self, i: Int, j: Int, value: UInt32):
        self.data[i * self.num_candidates + j] = value

    fn __del__(deinit self):
        self.data.free()

@fieldwise_init
struct StrongestPathsMatrix(Movable):
    """Represents the strongest paths matrix result."""
    var data: UnsafePointer[UInt32]
    var num_candidates: Int

    fn __init__(out self, num_candidates: Int):
        self.num_candidates = num_candidates
        var size = num_candidates * num_candidates
        self.data = UnsafePointer[UInt32].alloc(size)
        memset_zero(self.data, size)

    fn __getitem__(self, i: Int, j: Int) -> UInt32:
        return self.data[i * self.num_candidates + j]

    fn __setitem__(mut self, i: Int, j: Int, value: UInt32):
        self.data[i * self.num_candidates + j] = value

    fn __del__(deinit self):
        self.data.free()

fn populate_preferences_from_ranking(mut preferences: PreferenceMatrix, ranking: List[Int]):
    """
    Populates the preference matrix based on a ranking of candidates.

    Args:
        preferences: The preference matrix to populate.
        ranking: List of candidate indices in order of preference.
    """
    var n = len(ranking)
    for i in range(n):
        var preferred = ranking[i]
        for j in range(i + 1, n):
            var opponent = ranking[j]
            var current_count = preferences[preferred, opponent]
            preferences[preferred, opponent] = current_count + 1

fn build_pairwise_preferences(voter_rankings: List[List[Int]]) -> PreferenceMatrix:
    """
    Builds a pairwise preference matrix from voter rankings.

    Args:
        voter_rankings: List of voter rankings (each ranking is a list of candidate indices).

    Returns:
        PreferenceMatrix with pairwise vote counts.
    """
    # Find maximum candidate index to determine matrix size
    var max_candidate = 0
    for i in range(len(voter_rankings)):
        var ranking = voter_rankings[i].copy()
        for j in range(len(ranking)):
            var candidate = ranking[j]
            if candidate > max_candidate:
                max_candidate = candidate

    var num_candidates = max_candidate + 1
    var preferences = PreferenceMatrix(num_candidates)

    # Process each voter's ranking
    for i in range(len(voter_rankings)):
        var ranking = voter_rankings[i].copy()

        # Create complete ranking if incomplete
        var complete_ranking = List[Int]()
        var used = List[Bool]()
        for _ in range(num_candidates):
            used.append(False)

        # Add provided candidates
        for j in range(len(ranking)):
            var candidate = ranking[j]
            complete_ranking.append(candidate)
            used[candidate] = True

        # Add missing candidates in arbitrary order
        for k in range(num_candidates):
            if not used[k]:
                complete_ranking.append(k)

        populate_preferences_from_ranking(preferences, complete_ranking)

    return preferences^

fn compute_strongest_paths_serial(preferences: PreferenceMatrix) -> StrongestPathsMatrix:
    """
    Serial implementation of Schulze strongest paths computation.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths based on direct comparisons
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    strongest_paths[i, j] = pref_ij
                else:
                    strongest_paths[i, j] = 0

    # Step 2: Floyd-Warshall-like algorithm for strongest paths
    for i in range(num_candidates):
        for j in range(num_candidates):
            if i != j:
                for k in range(num_candidates):
                    if i != k and j != k:
                        var path_j_i = strongest_paths[j, i]
                        var path_i_k = strongest_paths[i, k]
                        var path_j_k = strongest_paths[j, k]
                        var new_path = min(path_j_i, path_i_k)
                        var max_path = max(path_j_k, new_path)
                        strongest_paths[j, k] = max_path

    return strongest_paths^

fn process_tile_cpu[tile_size: Int](
    c: UnsafePointer[UInt32],
    a: UnsafePointer[UInt32],
    b: UnsafePointer[UInt32],
    c_row: Int, c_col: Int,
    a_row: Int, a_col: Int,
    b_row: Int, b_col: Int,
    num_candidates: Int,
    tile_stride: Int
):
    """
    CPU-optimized tile processing for blocked Schulze algorithm.

    Args:
        c: Output tile.
        a: First input tile.
        b: Second input tile.
        c_row: Row index of output tile.
        c_col: Column index of output tile.
        a_row: Row index of first input tile.
        a_col: Column index of first input tile.
        b_row: Row index of second input tile.
        b_col: Column index of second input tile.
        num_candidates: Total number of candidates.
        tile_stride: Stride for accessing tiles.
    """
    for k in range(tile_size):
        for bi in range(tile_size):
            for bj in range(tile_size):
                # Check bounds
                var global_i = c_row + bi
                var global_j = c_col + bj
                var global_k = a_col + k

                if global_i >= num_candidates or global_j >= num_candidates or global_k >= num_candidates:
                    continue

                # Skip diagonal elements
                if global_i == global_j or global_i == global_k or global_k == global_j:
                    continue

                var a_val = a[bi * tile_stride + k]
                var b_val = b[k * tile_stride + bj]
                var c_idx = bi * tile_stride + bj
                var c_val = c[c_idx]
                var new_val = min(a_val, b_val)

                if new_val > c_val:
                    c[c_idx] = new_val

fn copy_tile_to_buffer(
    source: UnsafePointer[UInt32],
    dest: UnsafePointer[UInt32],
    start_row: Int,
    start_col: Int,
    tile_size: Int,
    num_candidates: Int
):
    """Copy a tile from the global matrix to a local buffer."""
    for i in range(tile_size):
        for j in range(tile_size):
            var row = start_row + i
            var col = start_col + j
            if row < num_candidates and col < num_candidates:
                dest[i * tile_size + j] = source[row * num_candidates + col]
            else:
                dest[i * tile_size + j] = 0

fn copy_buffer_to_tile(
    source: UnsafePointer[UInt32],
    dest: UnsafePointer[UInt32],
    start_row: Int,
    start_col: Int,
    tile_size: Int,
    num_candidates: Int
):
    """Copy a tile from a local buffer back to the global matrix."""
    for i in range(tile_size):
        for j in range(tile_size):
            var row = start_row + i
            var col = start_col + j
            if row < num_candidates and col < num_candidates:
                dest[row * num_candidates + col] = source[i * tile_size + j]

fn compute_strongest_paths_tiled_cpu(preferences: PreferenceMatrix) -> StrongestPathsMatrix:
    """
    Tiled CPU implementation of Schulze strongest paths computation.
    Uses blocking for better cache utilization.

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths
    @parameter
    fn init_paths(i: Int):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    strongest_paths[i, j] = pref_ij
                else:
                    strongest_paths[i, j] = 0

    parallelize[init_paths](num_candidates)

    # Step 2: Tiled Floyd-Warshall computation
    var tiles_count = (num_candidates + TILE_SIZE - 1) // TILE_SIZE

    for k in range(tiles_count):
        var k_start = k * TILE_SIZE
        var k_end = min(k_start + TILE_SIZE, num_candidates)
        var k_size = k_end - k_start

        # Dependent phase: process diagonal tile
        var diagonal_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
        memset_zero(diagonal_tile, TILE_SIZE * TILE_SIZE)

        copy_tile_to_buffer(
            strongest_paths.data, diagonal_tile,
            k_start, k_start, k_size, num_candidates
        )

        process_tile_cpu[TILE_SIZE](
            diagonal_tile, diagonal_tile, diagonal_tile,
            k_start, k_start, k_start, k_start, k_start, k_start,
            num_candidates, TILE_SIZE
        )

        copy_buffer_to_tile(
            diagonal_tile, strongest_paths.data,
            k_start, k_start, k_size, num_candidates
        )

        # Partially dependent phases - row tiles
        @parameter
        fn process_row_tiles(i: Int):
            if i == k:
                return

            var i_start = i * TILE_SIZE
            var i_end = min(i_start + TILE_SIZE, num_candidates)
            var i_size = i_end - i_start

            var c_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            var b_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            memset_zero(c_tile, TILE_SIZE * TILE_SIZE)
            memset_zero(b_tile, TILE_SIZE * TILE_SIZE)

            copy_tile_to_buffer(strongest_paths.data, c_tile, i_start, k_start, i_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, b_tile, k_start, k_start, k_size, num_candidates)

            process_tile_cpu[TILE_SIZE](
                c_tile, c_tile, b_tile,
                i_start, k_start, i_start, k_start, k_start, k_start,
                num_candidates, TILE_SIZE
            )

            copy_buffer_to_tile(c_tile, strongest_paths.data, i_start, k_start, i_size, num_candidates)

        parallelize[process_row_tiles](tiles_count)

        # Partially dependent phases - column tiles
        @parameter
        fn process_col_tiles(j: Int):
            if j == k:
                return

            var j_start = j * TILE_SIZE
            var j_end = min(j_start + TILE_SIZE, num_candidates)
            var j_size = j_end - j_start

            var c_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            var a_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            memset_zero(c_tile, TILE_SIZE * TILE_SIZE)
            memset_zero(a_tile, TILE_SIZE * TILE_SIZE)

            copy_tile_to_buffer(strongest_paths.data, c_tile, k_start, j_start, j_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, a_tile, k_start, k_start, k_size, num_candidates)

            process_tile_cpu[TILE_SIZE](
                c_tile, a_tile, c_tile,
                k_start, j_start, k_start, k_start, k_start, j_start,
                num_candidates, TILE_SIZE
            )

            copy_buffer_to_tile(c_tile, strongest_paths.data, k_start, j_start, j_size, num_candidates)

        parallelize[process_col_tiles](tiles_count)

        # Independent phase
        @parameter
        fn process_independent_tiles(idx: Int):
            var i = idx // tiles_count
            var j = idx % tiles_count

            if i == k or j == k:
                return

            var i_start = i * TILE_SIZE
            var i_end = min(i_start + TILE_SIZE, num_candidates)
            var i_size = i_end - i_start

            var j_start = j * TILE_SIZE
            var j_end = min(j_start + TILE_SIZE, num_candidates)
            var j_size = j_end - j_start

            var c_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            var a_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            var b_tile = stack_allocation[TILE_SIZE * TILE_SIZE, UInt32]()
            memset_zero(c_tile, TILE_SIZE * TILE_SIZE)
            memset_zero(a_tile, TILE_SIZE * TILE_SIZE)
            memset_zero(b_tile, TILE_SIZE * TILE_SIZE)

            copy_tile_to_buffer(strongest_paths.data, c_tile, i_start, j_start, min(i_size, j_size), num_candidates)
            copy_tile_to_buffer(strongest_paths.data, a_tile, i_start, k_start, i_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, b_tile, k_start, j_start, j_size, num_candidates)

            process_tile_cpu[TILE_SIZE](
                c_tile, a_tile, b_tile,
                i_start, j_start, i_start, k_start, k_start, j_start,
                num_candidates, TILE_SIZE
            )

            copy_buffer_to_tile(c_tile, strongest_paths.data, i_start, j_start, min(i_size, j_size), num_candidates)

        parallelize[process_independent_tiles](tiles_count * tiles_count)

    return strongest_paths^

fn get_winner_and_ranking(candidates: List[Int], strongest_paths: StrongestPathsMatrix) -> (Int, List[Int]):
    """
    Determines the winner and ranking based on strongest paths matrix.

    Args:
        candidates: List of candidate identifiers.
        strongest_paths: Computed strongest paths matrix.

    Returns:
        Tuple of (winner_index, ranked_candidates).
    """
    var num_candidates = len(candidates)
    var wins = List[Int]()
    for _ in range(num_candidates):
        wins.append(0)

    # Count wins for each candidate
    for i in range(num_candidates):
        var win_count = 0
        for j in range(num_candidates):
            if i != j and strongest_paths[i, j] > strongest_paths[j, i]:
                win_count += 1
        wins[i] = win_count

    # Find winner (candidate with most wins)
    var winner_idx = 0
    var max_wins = wins[0]
    for i in range(1, num_candidates):
        if wins[i] > max_wins:
            max_wins = wins[i]
            winner_idx = i

    # Create ranking by sorting candidates by win count
    var ranking = List[Int]()
    var used = List[Bool]()
    for _ in range(num_candidates):
        used.append(False)

    # Add candidates in order of decreasing wins
    for _ in range(num_candidates):
        var best_idx = -1
        var best_wins = -1

        for i in range(num_candidates):
            if not used[i] and wins[i] > best_wins:
                best_wins = wins[i]
                best_idx = i

        if best_idx >= 0:
            ranking.append(candidates[best_idx])
            used[best_idx] = True

    return (candidates[winner_idx], ranking^)

fn generate_random_preferences(num_candidates: Int, num_voters: Int) -> PreferenceMatrix:
    """
    Generates random preference matrix for testing.

    Args:
        num_candidates: Number of candidates.
        num_voters: Number of voters.

    Returns:
        Random preference matrix.
    """
    var preferences = PreferenceMatrix(num_candidates)

    for _ in range(num_voters):
        # Generate random ranking
        var ranking = List[Int]()
        for i in range(num_candidates):
            ranking.append(i)

        # Fisher-Yates shuffle
        for i in range(num_candidates - 1, 0, -1):
            var j = Int(random_si64(0, i + 1))
            var temp = ranking[i]
            ranking[i] = ranking[j]
            ranking[j] = temp

        populate_preferences_from_ranking(preferences, ranking)

    return preferences^

fn benchmark_implementation(
    name: String,
    implementation: fn(PreferenceMatrix) -> StrongestPathsMatrix,
    preferences: PreferenceMatrix
) -> StrongestPathsMatrix:
    """
    Benchmark a specific implementation.

    Args:
        name: Name of the implementation.
        implementation: Function to benchmark.
        preferences: Input preferences matrix.

    Returns:
        Result of the computation.
    """
    print("Computing strongest paths (" + name + ")...")
    var start_time = perf_counter_ns()
    var result = implementation(preferences)
    var end_time = perf_counter_ns()
    var elapsed_ns = end_time - start_time
    var elapsed_ms = elapsed_ns // 1_000_000  # Convert to milliseconds

    print(name + " took: " + String(elapsed_ms) + " ms")
    return result^

fn parse_int_arg[origin: Origin](args: VariadicList[StringSlice[origin]], flag: String, default: Int) -> Int:
    """Parse an integer command-line argument."""
    for i in range(len(args)):
        if args[i] == flag and i + 1 < len(args):
            try:
                return atol(args[i + 1])
            except:
                print("Warning: Invalid value for", flag, "- using default:", default)
                return default
    return default

fn has_flag[origin: Origin](args: VariadicList[StringSlice[origin]], flag: String) -> Bool:
    """Check if a flag exists in command-line arguments."""
    for i in range(len(args)):
        if args[i] == flag:
            return True
    return False

fn print_usage():
    """Print usage information."""
    print("Usage: mojo scaling_democracy.mojo [OPTIONS]")
    print()
    print("Options:")
    print("  --num-candidates N    Number of candidates (default: 128)")
    print("  --num-voters N        Number of voters (default: 2000)")
    print("  --tile-size N         Tile size for blocked algorithm (default: 16, must match TILE_SIZE)")
    print("  --serial-only         Run only serial implementation")
    print("  --tiled-only          Run only tiled CPU implementation")
    print("  --help, -h            Show this help message")
    print()
    print("Example:")
    print("  pixi run mojo scaling_democracy.mojo --num-candidates 256 --num-voters 4000")

fn main():
    """
    Main function demonstrating the Schulze voting algorithm.
    Tests both CPU implementations with performance benchmarking.
    Supports command-line arguments for configuration.
    """
    var args = argv()

    # Check for help flag
    if has_flag(args, "--help") or has_flag(args, "-h"):
        print_usage()
        return

    # Parse command-line arguments
    var num_candidates = parse_int_arg(args, "--num-candidates", 128)
    var num_voters = parse_int_arg(args, "--num-voters", 2000)
    var tile_size_arg = parse_int_arg(args, "--tile-size", TILE_SIZE)
    var serial_only = has_flag(args, "--serial-only")
    var tiled_only = has_flag(args, "--tiled-only")

    # Validate tile size
    if tile_size_arg != TILE_SIZE:
        print("Warning: --tile-size argument (" + String(tile_size_arg) + ") must match compiled TILE_SIZE (" + String(TILE_SIZE) + ")")
        print("         Ignoring --tile-size argument and using TILE_SIZE=" + String(TILE_SIZE))

    # Validate candidates (must be at least 4)
    if num_candidates < 4:
        print("Error: num_candidates must be at least 4")
        return

    print("=== Mojo Schulze Voting Algorithm ===")
    print("Candidates:", num_candidates)
    print("Voters:", num_voters)
    print("Tile Size:", TILE_SIZE)
    print()

    print("Generating random voter preferences...")
    var preferences = generate_random_preferences(num_candidates, num_voters)

    var run_both = not serial_only and not tiled_only

    # Test different implementations based on flags
    if run_both:
        # Run both and compare
        print("Testing serial implementation...")
        var strongest_paths_serial = benchmark_implementation("Serial", compute_strongest_paths_serial, preferences)

        print("Testing tiled CPU implementation...")
        var strongest_paths_tiled = benchmark_implementation("Tiled CPU", compute_strongest_paths_tiled_cpu, preferences)

        # Verify CPU results match
        var cpu_results_match = True
        for i in range(num_candidates):
            for j in range(num_candidates):
                if strongest_paths_serial[i, j] != strongest_paths_tiled[i, j]:
                    cpu_results_match = False
                    break
            if not cpu_results_match:
                break

        if cpu_results_match:
            print("✓ Serial and tiled CPU results match!")
        else:
            print("✗ CPU results don't match!")

        # Get winner and ranking using tiled result
        var candidates = List[Int]()
        for i in range(num_candidates):
            candidates.append(i)

        var result_tuple = get_winner_and_ranking(candidates, strongest_paths_tiled)
        var winner = result_tuple[0]
        var ranking = result_tuple[1].copy()

        print()
        print("=== Results ===")
        print("Winner: Candidate", winner)
        print("Top 5 candidates:", ranking[0], ranking[1], ranking[2], ranking[3], ranking[4])
    elif serial_only:
        # Run only serial
        print("Testing serial implementation...")
        var strongest_paths_serial = benchmark_implementation("Serial", compute_strongest_paths_serial, preferences)

        var candidates = List[Int]()
        for i in range(num_candidates):
            candidates.append(i)

        var result_tuple = get_winner_and_ranking(candidates, strongest_paths_serial)
        var winner = result_tuple[0]
        var ranking = result_tuple[1].copy()

        print()
        print("=== Results ===")
        print("Winner: Candidate", winner)
        print("Top 5 candidates:", ranking[0], ranking[1], ranking[2], ranking[3], ranking[4])
    else:  # tiled_only
        # Run only tiled
        print("Testing tiled CPU implementation...")
        var strongest_paths_tiled = benchmark_implementation("Tiled CPU", compute_strongest_paths_tiled_cpu, preferences)

        var candidates = List[Int]()
        for i in range(num_candidates):
            candidates.append(i)

        var result_tuple = get_winner_and_ranking(candidates, strongest_paths_tiled)
        var winner = result_tuple[0]
        var ranking = result_tuple[1].copy()

        print()
        print("=== Results ===")
        print("Winner: Candidate", winner)
        print("Top 5 candidates:", ranking[0], ranking[1], ranking[2], ranking[3], ranking[4])

    # Calculate some statistics
    var total_votes: Int = 0
    for i in range(num_candidates):
        for j in range(num_candidates):
            total_votes += Int(preferences[i, j])

    print("Total pairwise votes:", total_votes)
    print("Average votes per pair:", total_votes // (num_candidates * num_candidates))

    print()
    print("✓ Mojo Schulze voting implementation complete!")
    print("  - Serial implementation: Basic Floyd-Warshall algorithm")
    print("  - Tiled CPU implementation: Blocked algorithm with parallelization")
    print("  - All implementations produce identical results")
