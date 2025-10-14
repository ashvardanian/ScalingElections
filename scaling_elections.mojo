"""
Mojo implementation of the Schulze voting algorithm with CPU and GPU acceleration.

This implementation provides parallel computation of strongest paths for democratic voting,
ported from the original Python/CUDA implementation in scaling_elections.py and scaling_elections.cu.
The Schulze method is a Condorcet voting system that selects the candidate who would win
in head-to-head comparisons against all other candidates.

## Usage

Run directly with Mojo via Pixi:

```bash
pixi run mojo scaling_elections.mojo
pixi run mojo scaling_elections.mojo --num-candidates 4096 --num-voters 4096 --run-cpu --run-gpu
```

For proper benchmarking with large random-generated preference matrices:

```bash
pixi run mojo scaling_elections.mojo --num-candidates 2048 --num-voters 0 --run-gpu --no-serial --warmup 1 --repeat 20
pixi run mojo scaling_elections.mojo --num-candidates 4096 --num-voters 0 --run-gpu --no-serial --warmup 1 --repeat 10
pixi run mojo scaling_elections.mojo --num-candidates 8192 --num-voters 0 --run-gpu --no-serial --warmup 1 --repeat 5
pixi run mojo scaling_elections.mojo --num-candidates 16384 --num-voters 0 --run-gpu --no-serial --warmup 1 --repeat 3
pixi run mojo scaling_elections.mojo --num-candidates 32768 --num-voters 0 --run-gpu --no-serial --warmup 1 --repeat 1
```

Or compile and run:

```bash
pixi run mojo build scaling_elections.mojo -o scaling_elections
./scaling_elections
```

See: https://ashvardanian.com/posts/scaling-democracy/
"""

# Core data structures and memory management
from collections import List
from memory import UnsafePointer, memset_zero, stack_allocation, AddressSpace

# Algorithms and utilities
from algorithm import parallelize
from builtin.sort import sort
from random import random_si64

# System and runtime
from sys import argv, has_accelerator, simdwidthof
from time import perf_counter_ns

# GPU acceleration
from buffer import NDBuffer
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from gpu.id import block_idx, thread_idx, block_dim, global_idx
from gpu.sync import barrier
from layout import Layout, LayoutTensor

# Type aliases for clarity
alias VotesCount = UInt32
alias CandidateIdx = Int

# ! Tile sizes control cache utilization and parallelism efficiency.
# ! CPU: 16x16 fits L1 cache (typical 32KB), balances reuse vs overhead.
# ! GPU: 32x32 matches NVIDIA warp size (32 threads), maximizes occupancy.
# ! Pre-compiled variants allow runtime selection without JIT overhead.
alias DEFAULT_CPU_TILE_SIZE = 16
alias DEFAULT_GPU_TILE_SIZE = 32
alias ALLOWED_CPU_TILE_SIZES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
alias ALLOWED_GPU_TILE_SIZES = (4, 8, 12, 16, 24, 32, 48, 64)

@fieldwise_init
struct IndexedScore(Copyable, Movable, Comparable):
    """Pairs a candidate index with their win count for sorting."""
    var idx: Int
    var score: Int

    fn __lt__(self, other: Self) -> Bool:
        # Sort in descending order by score (higher scores first)
        return self.score > other.score

    fn __le__(self, other: Self) -> Bool:
        return self.score >= other.score

    fn __eq__(self, other: Self) -> Bool:
        return self.score == other.score

    fn __ne__(self, other: Self) -> Bool:
        return self.score != other.score

    fn __gt__(self, other: Self) -> Bool:
        return self.score < other.score

    fn __ge__(self, other: Self) -> Bool:
        return self.score <= other.score

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
        used.resize(num_candidates, False)

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

fn compute_strongest_paths_serial(preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
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
    a_col: Int,
    b_col: Int,
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
        a_col: Column index of first input tile.
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

# =============================================================================
# SIMD-VECTORIZED CPU TILE PROCESSORS FOR SCHULZE VOTING ALGORITHM
# =============================================================================
#
# Three-phase specialized tile processors using SIMD for vectorization
# Mirrors GPU's phase-specific design but uses SIMD vectors instead of threads
# =============================================================================

fn process_tile_cpu_simd_independent[tile_size: Int, simd_width: Int](
    c: UnsafePointer[UInt32],
    a: UnsafePointer[UInt32],
    b: UnsafePointer[UInt32],
    tile_stride: Int
):
    """
    SIMD-vectorized tile processor for independent tiles (no diagonal checking needed).
    Processes multiple elements along the j dimension using SIMD vectors.

    Args:
        c: Output tile.
        a: First input tile.
        b: Second input tile.
        tile_stride: Stride for accessing tiles.
    """
    # Compile-time assertion: tile_size must be divisible by simd_width
    constrained[tile_size % simd_width == 0, "tile_size must be divisible by simd_width"]()

    # Process k loop over intermediate values
    for k in range(tile_size):
        # Process each row
        for bi in range(tile_size):
            var a_val = a[bi * tile_stride + k]

            # Vectorized processing of columns (j dimension)
            alias num_simd_chunks = tile_size // simd_width

            # Process all elements with SIMD
            for chunk in range(num_simd_chunks):
                var bj = chunk * simd_width
                var c_base_idx = bi * tile_stride + bj
                var b_base_idx = k * tile_stride + bj

                # Load SIMD vectors
                var c_vec = c.load[width=simd_width](c_base_idx)
                var b_vec = b.load[width=simd_width](b_base_idx)

                # Broadcast a_val to SIMD vector
                var a_vec = SIMD[DType.uint32, simd_width](a_val)

                # Compute min(a, b) and max(c, min_val)
                var min_val = min(a_vec, b_vec)
                var new_c = max(c_vec, min_val)

                # Store result
                c.store[width=simd_width](c_base_idx, new_c)

fn process_tile_cpu_simd_diagonal[tile_size: Int, simd_width: Int](
    c: UnsafePointer[UInt32],
    a: UnsafePointer[UInt32],
    b: UnsafePointer[UInt32],
    c_row: Int,
    c_col: Int,
    a_col: Int,
    num_candidates: Int,
    tile_stride: Int
):
    """
    SIMD-vectorized tile processor for diagonal tiles (requires diagonal avoidance).
    Uses masking and select operations to avoid branches.

    Args:
        c: Output tile.
        a: First input tile.
        b: Second input tile.
        c_row: Global row index of output tile.
        c_col: Global column index of output tile.
        a_col: Global column index for intermediate dimension.
        num_candidates: Total number of candidates.
        tile_stride: Stride for accessing tiles.
    """
    # Compile-time assertion: tile_size must be divisible by simd_width
    constrained[tile_size % simd_width == 0, "tile_size must be divisible by simd_width"]()

    # Process k loop
    for k in range(tile_size):
        var global_k = a_col + k

        # Process each row
        for bi in range(tile_size):
            var global_i = c_row + bi
            var a_val = a[bi * tile_stride + k]

            # Vectorized processing with diagonal masking
            alias num_simd_chunks = tile_size // simd_width

            # Process all elements with SIMD
            for chunk in range(num_simd_chunks):
                var bj = chunk * simd_width
                var c_base_idx = bi * tile_stride + bj
                var b_base_idx = k * tile_stride + bj

                # Load SIMD vectors
                var c_vec = c.load[width=simd_width](c_base_idx)
                var b_vec = b.load[width=simd_width](b_base_idx)
                var a_vec = SIMD[DType.uint32, simd_width](a_val)

                # Compute candidate new values
                var min_val = min(a_vec, b_vec)

                # Build diagonal mask (branchless)
                var mask = SIMD[DType.bool, simd_width]()
                @parameter
                fn compute_mask[width: Int]():
                    for lane in range(width):
                        var global_j = c_col + bj + lane
                        # Allow update if: not on diagonal AND new_val > old_val
                        var not_diag_c = global_i != global_j
                        var not_diag_a = global_i != global_k
                        var not_diag_b = global_k != global_j
                        var is_bigger = min_val[lane] > c_vec[lane]
                        mask[lane] = not_diag_c and not_diag_a and not_diag_b and is_bigger

                compute_mask[simd_width]()

                # Conditional update using select (mask ? min_val : c_vec)
                var new_c = mask.select(min_val, c_vec)
                c.store[width=simd_width](c_base_idx, new_c)

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

@always_inline
fn calculate_tile_bounds(tile_idx: Int, tile_size: Int, total_size: Int) -> (Int, Int, Int):
    """
    Calculate tile boundaries for blocking algorithms.

    Args:
        tile_idx: Index of the tile.
        tile_size: Size of each tile.
        total_size: Total problem size.

    Returns:
        Tuple of (start_index, end_index, actual_size).
    """
    var start = tile_idx * tile_size
    var end = min(start + tile_size, total_size)
    var size = end - start
    return (start, end, size)

fn compute_strongest_paths_tiled_cpu[tile_size: Int = DEFAULT_CPU_TILE_SIZE](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    Tiled CPU implementation of Schulze strongest paths computation.
    Uses blocking for better cache utilization.

    Parameters:
        tile_size: Compile-time tile size for CPU processing (default: 16).

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
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for k in range(num_tiles):
        var k_bounds = calculate_tile_bounds(k, tile_size, num_candidates)
        var k_start = k_bounds[0]
        var k_size = k_bounds[2]

        # Dependent phase: process diagonal tile
        var diagonal_tile = stack_allocation[tile_size * tile_size, UInt32]()
        memset_zero(diagonal_tile, tile_size * tile_size)

        copy_tile_to_buffer(
            strongest_paths.data, diagonal_tile,
            k_start, k_start, tile_size, num_candidates
        )

        process_tile_cpu[tile_size](
            diagonal_tile, diagonal_tile, diagonal_tile,
            k_start, k_start, k_start, k_start,
            num_candidates, tile_size
        )

        copy_buffer_to_tile(
            diagonal_tile, strongest_paths.data,
            k_start, k_start, tile_size, num_candidates
        )

        # Partially dependent phases - row tiles
        @parameter
        fn process_row_tiles(i: Int):
            if i == k:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]
            var i_size = i_bounds[2]

            var c_tile = stack_allocation[tile_size * tile_size, UInt32]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32]()
            memset_zero(c_tile, tile_size * tile_size)
            memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(strongest_paths.data, c_tile, i_start, k_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, b_tile, k_start, k_start, tile_size, num_candidates)

            process_tile_cpu[tile_size](
                c_tile, c_tile, b_tile,
                i_start, k_start, k_start, k_start,
                num_candidates, tile_size
            )

            copy_buffer_to_tile(c_tile, strongest_paths.data, i_start, k_start, tile_size, num_candidates)

        parallelize[process_row_tiles](num_tiles)

        # Partially dependent phases - column tiles
        @parameter
        fn process_col_tiles(j: Int):
            if j == k:
                return

            var j_bounds = calculate_tile_bounds(j, tile_size, num_candidates)
            var j_start = j_bounds[0]
            var j_size = j_bounds[2]

            var c_tile = stack_allocation[tile_size * tile_size, UInt32]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32]()
            memset_zero(c_tile, tile_size * tile_size)
            memset_zero(a_tile, tile_size * tile_size)

            copy_tile_to_buffer(strongest_paths.data, c_tile, k_start, j_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, a_tile, k_start, k_start, tile_size, num_candidates)

            process_tile_cpu[tile_size](
                c_tile, a_tile, c_tile,
                k_start, j_start, k_start, j_start,
                num_candidates, tile_size
            )

            copy_buffer_to_tile(c_tile, strongest_paths.data, k_start, j_start, tile_size, num_candidates)

        parallelize[process_col_tiles](num_tiles)

        # Independent phase
        @parameter
        fn process_independent_tiles(idx: Int):
            var i = idx // num_tiles
            var j = idx % num_tiles

            if i == k or j == k:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]
            var i_size = i_bounds[2]

            var j_bounds = calculate_tile_bounds(j, tile_size, num_candidates)
            var j_start = j_bounds[0]
            var j_size = j_bounds[2]

            var c_tile = stack_allocation[tile_size * tile_size, UInt32]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32]()
            memset_zero(c_tile, tile_size * tile_size)
            memset_zero(a_tile, tile_size * tile_size)
            memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(strongest_paths.data, c_tile, i_start, j_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, a_tile, i_start, k_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, b_tile, k_start, j_start, tile_size, num_candidates)

            process_tile_cpu[tile_size](
                c_tile, a_tile, b_tile,
                i_start, j_start, k_start, j_start,
                num_candidates, tile_size
            )

            copy_buffer_to_tile(c_tile, strongest_paths.data, i_start, j_start, tile_size, num_candidates)

        parallelize[process_independent_tiles](num_tiles * num_tiles)

    return strongest_paths^

fn compute_strongest_paths_tiled_cpu_dispatch[*allowed_sizes: Int](preferences: PreferenceMatrix, tile_size: Int) raises -> StrongestPathsMatrix:
    """
    Runtime dispatcher for CPU tiled implementation.
    Pre-compiles variants for allowed tile sizes and dispatches based on runtime value.

    Parameters:
        allowed_sizes: Variadic list of compile-time tile sizes to support.

    Args:
        preferences: Input preference matrix.
        tile_size: Runtime tile size selection.

    Returns:
        Computed strongest paths matrix.
    """
    alias sizes = VariadicList(allowed_sizes)

    @parameter
    for size in sizes:
        if tile_size == size:
            return compute_strongest_paths_tiled_cpu[size](preferences)

    # Fallback for unsupported sizes
    print("Warning: Unsupported CPU tile size", tile_size, "- falling back to default", DEFAULT_CPU_TILE_SIZE)
    return compute_strongest_paths_tiled_cpu[DEFAULT_CPU_TILE_SIZE](preferences)

fn compute_strongest_paths_tiled_cpu_simd[tile_size: Int = DEFAULT_CPU_TILE_SIZE](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    SIMD-vectorized tiled CPU implementation of Schulze strongest paths computation.
    Uses phase-specific SIMD tile processors for optimal vectorization and minimal branching.

    Parameters:
        tile_size: Compile-time tile size for CPU processing (default: 16).

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    # Use SIMD width of 8 for uint32 (works on AVX2/AVX-512)
    alias simd_width = 8
    var num_candidates = preferences.num_candidates
    var strongest_paths = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize strongest paths (same as regular CPU version)
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

    # Step 2: SIMD-vectorized tiled Floyd-Warshall computation
    var num_tiles = (num_candidates + tile_size - 1) // tile_size

    for k in range(num_tiles):
        var k_bounds = calculate_tile_bounds(k, tile_size, num_candidates)
        var k_start = k_bounds[0]

        # Diagonal phase: uses diagonal-aware SIMD processor
        var diagonal_tile = stack_allocation[tile_size * tile_size, UInt32]()
        memset_zero(diagonal_tile, tile_size * tile_size)

        copy_tile_to_buffer(
            strongest_paths.data, diagonal_tile,
            k_start, k_start, tile_size, num_candidates
        )

        process_tile_cpu_simd_diagonal[tile_size, simd_width](
            diagonal_tile, diagonal_tile, diagonal_tile,
            k_start, k_start, k_start,
            num_candidates, tile_size
        )

        copy_buffer_to_tile(
            diagonal_tile, strongest_paths.data,
            k_start, k_start, tile_size, num_candidates
        )

        # Partially dependent phases - row and column tiles
        @parameter
        fn process_row_col_tiles(i: Int):
            if i == k:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]

            # Row tile (i, k)
            var c_tile_row = stack_allocation[tile_size * tile_size, UInt32]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32]()
            memset_zero(c_tile_row, tile_size * tile_size)
            memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(strongest_paths.data, c_tile_row, i_start, k_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, b_tile, k_start, k_start, tile_size, num_candidates)

            process_tile_cpu_simd_diagonal[tile_size, simd_width](
                c_tile_row, c_tile_row, b_tile,
                i_start, k_start, k_start,
                num_candidates, tile_size
            )

            copy_buffer_to_tile(c_tile_row, strongest_paths.data, i_start, k_start, tile_size, num_candidates)

            # Column tile (k, i)
            var c_tile_col = stack_allocation[tile_size * tile_size, UInt32]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32]()
            memset_zero(c_tile_col, tile_size * tile_size)
            memset_zero(a_tile, tile_size * tile_size)

            copy_tile_to_buffer(strongest_paths.data, c_tile_col, k_start, i_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, a_tile, k_start, k_start, tile_size, num_candidates)

            process_tile_cpu_simd_diagonal[tile_size, simd_width](
                c_tile_col, a_tile, c_tile_col,
                k_start, i_start, k_start,
                num_candidates, tile_size
            )

            copy_buffer_to_tile(c_tile_col, strongest_paths.data, k_start, i_start, tile_size, num_candidates)

        parallelize[process_row_col_tiles](num_tiles)

        # Independent phase: uses fast SIMD processor (no diagonal checks)
        @parameter
        fn process_independent_tiles(idx: Int):
            var i = idx // num_tiles
            var j = idx % num_tiles

            if i == k or j == k:
                return

            var i_bounds = calculate_tile_bounds(i, tile_size, num_candidates)
            var i_start = i_bounds[0]

            var j_bounds = calculate_tile_bounds(j, tile_size, num_candidates)
            var j_start = j_bounds[0]

            var c_tile = stack_allocation[tile_size * tile_size, UInt32]()
            var a_tile = stack_allocation[tile_size * tile_size, UInt32]()
            var b_tile = stack_allocation[tile_size * tile_size, UInt32]()
            memset_zero(c_tile, tile_size * tile_size)
            memset_zero(a_tile, tile_size * tile_size)
            memset_zero(b_tile, tile_size * tile_size)

            copy_tile_to_buffer(strongest_paths.data, c_tile, i_start, j_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, a_tile, i_start, k_start, tile_size, num_candidates)
            copy_tile_to_buffer(strongest_paths.data, b_tile, k_start, j_start, tile_size, num_candidates)

            # Use independent processor if not on diagonal, otherwise use diagonal processor
            if i == j:
                process_tile_cpu_simd_diagonal[tile_size, simd_width](
                    c_tile, a_tile, b_tile,
                    i_start, j_start, k_start,
                    num_candidates, tile_size
                )
            else:
                process_tile_cpu_simd_independent[tile_size, simd_width](
                    c_tile, a_tile, b_tile,
                    tile_size
                )

            copy_buffer_to_tile(c_tile, strongest_paths.data, i_start, j_start, tile_size, num_candidates)

        parallelize[process_independent_tiles](num_tiles * num_tiles)

    return strongest_paths^

fn get_winner_and_ranking(strongest_paths: StrongestPathsMatrix) -> (Int, List[Int]):
    """
    Determines the winner and ranking based on strongest paths matrix.

    Args:
        strongest_paths: Computed strongest paths matrix.

    Returns:
        Tuple of (winner_candidate_id, ranked_candidate_ids).
    """
    var num_candidates = strongest_paths.num_candidates
    var wins = List[Int]()
    wins.resize(num_candidates, 0)

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

    # Create ranking by sorting candidates by win count (O(n log n) using built-in sort)
    var scored_candidates = List[IndexedScore]()
    for i in range(num_candidates):
        scored_candidates.append(IndexedScore(i, wins[i]))

    # Sort by score (IndexedScore's __lt__ sorts in descending order)
    sort(scored_candidates)

    # Extract just the candidate indices
    var ranking = List[Int]()
    for i in range(len(scored_candidates)):
        ranking.append(scored_candidates[i].idx)

    return (winner_idx, ranking^)

fn generate_random_preferences(num_candidates: Int, num_voters: Int) -> PreferenceMatrix:
    """
    Generates random preference matrix for testing.

    Args:
        num_candidates: Number of candidates.
        num_voters: Number of voters. If 0, generates random preference matrix directly.

    Returns:
        Random preference matrix.
    """
    var preferences = PreferenceMatrix(num_candidates)

    # Fast path: directly generate random preference matrix (parallelized)
    if num_voters == 0:
        @parameter
        fn fill_row(i: Int):
            for j in range(num_candidates):
                preferences[i, j] = UInt32(random_si64(0, num_candidates))

        parallelize[fill_row](num_candidates)
        return preferences^

    # Slow path: generate from voter rankings
    # Allocate ranking once and reuse for all voters
    var ranking = List[Int]()
    ranking.resize(num_candidates, 0)

    for _ in range(num_voters):
        # Re-initialize ranking in-place
        for i in range(num_candidates):
            ranking[i] = i

        # Fisher-Yates shuffle
        for i in range(num_candidates - 1, 0, -1):
            var j = Int(random_si64(0, i + 1))
            var temp = ranking[i]
            ranking[i] = ranking[j]
            ranking[j] = temp

        populate_preferences_from_ranking(preferences, ranking)

    return preferences^

fn run_warmup(
    implementation: fn(PreferenceMatrix) raises -> StrongestPathsMatrix,
    preferences: PreferenceMatrix,
    warmup: Int
) raises:
    """Run warmup iterations and print timing."""
    for i in range(warmup):
        var start_time = perf_counter_ns()
        _ = implementation(preferences)
        var elapsed_ns = perf_counter_ns() - start_time
        if warmup > 1:
            print("  Warm-up " + String(i+1) + "/" + String(warmup) + ": " + format_time(elapsed_ns))
        else:
            print("  Warm-up: " + format_time(elapsed_ns))

fn run_and_average(
    implementation: fn(PreferenceMatrix) raises -> StrongestPathsMatrix,
    preferences: PreferenceMatrix,
    repeat: Int,
    mut result: StrongestPathsMatrix
) raises -> Int:
    """Run benchmark iterations and return avg_time_ns, storing result in mutable parameter."""
    var times = List[Int]()
    for _ in range(repeat):
        var start_time = perf_counter_ns()
        result = implementation(preferences)
        var elapsed_ns = perf_counter_ns() - start_time
        times.append(elapsed_ns)

    # Calculate average time
    var total_time: Int = 0
    for i in range(len(times)):
        total_time += times[i]
    var avg_time = total_time // repeat
    return avg_time

fn format_time(elapsed_ns: Int) -> String:
    """Format time with appropriate unit (ms or s)."""
    var elapsed_ms = elapsed_ns // 1_000_000
    if elapsed_ms < 1000:
        return String(elapsed_ms) + " ms"
    else:
        var elapsed_sec = Float64(elapsed_ns) / 1_000_000_000.0
        var sec_x100 = Int(elapsed_sec * 100.0)
        return String(sec_x100 // 100) + "." + String((sec_x100 % 100) // 10) + String(sec_x100 % 10) + " s"

fn format_throughput(cells_per_sec: Float64) -> String:
    """Format throughput with appropriate unit (T/G/M cells³/s)."""
    if cells_per_sec >= 1e12:
        var t_x10 = Int(cells_per_sec / 1e11)
        return String(t_x10 // 10) + "." + String(t_x10 % 10) + " Tcells³/s"
    elif cells_per_sec >= 1e9:
        var g_x10 = Int(cells_per_sec / 1e8)
        return String(g_x10 // 10) + "." + String(g_x10 % 10) + " Gcells³/s"
    elif cells_per_sec >= 1e6:
        var m_x10 = Int(cells_per_sec / 1e5)
        return String(m_x10 // 10) + "." + String(m_x10 % 10) + " Mcells³/s"
    else:
        var k_x10 = Int(cells_per_sec / 1e2)
        return String(k_x10 // 10) + "." + String(k_x10 % 10) + " Kcells³/s"

fn format_throughput_from_ns(elapsed_ns: Int, num_candidates: Int) -> String:
    """Calculate and format throughput from timing."""
    if elapsed_ns <= 0:
        return "N/A"
    var elapsed_sec = Float64(elapsed_ns) / 1_000_000_000.0
    if elapsed_sec <= 0.0:
        return "N/A"
    var candidates_f = Float64(num_candidates)
    var total_cells = candidates_f * candidates_f * candidates_f
    var cells_per_sec = total_cells / elapsed_sec
    return format_throughput(cells_per_sec)

# =============================================================================
# PURE MOJO GPU KERNELS FOR SCHULZE VOTING ALGORITHM
# =============================================================================
#
# Implementation of three-phase tiled Floyd-Warshall algorithm on GPU
# Matching the CUDA reference in scaling_elections.cu
# =============================================================================

alias SharedUInt32Ptr = UnsafePointer[UInt32, address_space=AddressSpace(3)]

@always_inline
fn process_tile_gpu_device[
    tile_size: Int, may_be_diagonal: Bool, synchronize: Bool
](
    c_shared: SharedUInt32Ptr,
    a_shared: SharedUInt32Ptr,
    b_shared: SharedUInt32Ptr,
    c_row: Int, c_col: Int,
    a_row: Int, a_col: Int,
    b_row: Int, b_col: Int
):
    """
    Core tile processing logic for GPU - runs on each thread.
    Processes one cell (bi, bj) of the tile through all k values.

    This matches the CUDA process_tile_cuda_ template function.
    """
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)
    var c_idx = bi * tile_size + bj

    # Each thread processes one cell of the output tile
    var c_val = c_shared[c_idx]

    # Floyd-Warshall inner loop over k
    for k in range(tile_size):
        var global_k = a_col + k

        var a_idx = bi * tile_size + k
        var b_idx = k * tile_size + bj

        var a_val = a_shared[a_idx]
        var b_val = b_shared[b_idx]
        var smallest = min(a_val, b_val)

        @parameter
        if may_be_diagonal:
            # Compute global indices
            var global_i = c_row + bi
            var global_j = c_col + bj

            # Diagonal avoidance using branchless bit operations
            var is_not_diagonal_c = UInt32(1) if global_i != global_j else UInt32(0)
            var is_not_diagonal_a = UInt32(1) if global_i != global_k else UInt32(0)
            var is_not_diagonal_b = UInt32(1) if global_k != global_j else UInt32(0)
            var is_bigger = UInt32(1) if smallest > c_val else UInt32(0)
            var will_replace = is_not_diagonal_c & is_not_diagonal_a & is_not_diagonal_b & is_bigger

            if will_replace == 1:
                c_val = smallest
        else:
            # Non-diagonal case - simple max
            c_val = max(c_val, smallest)

        # Write back IMMEDIATELY after update - critical for correctness!
        # When a_shared/b_shared/c_shared point to the same buffer (diagonal phase),
        # threads must see updated values from previous k iterations.
        c_shared[c_idx] = c_val

        # Synchronize after each k iteration when needed (for diagonal/shared tiles)
        @parameter
        if synchronize:
            barrier()

fn gpu_diagonal_kernel[tile_size: Int](
    graph: UnsafePointer[UInt32],
    n: Int,
    k: Int
):
    """
    GPU kernel for diagonal phase - processes tile (k, k).
    Matches cuda_diagonal_ from CUDA implementation.
    """
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)

    # Allocate shared memory for one tile
    var c_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()

    # Load tile from global memory
    var global_idx = k * tile_size * n + k * tile_size + bi * n + bj
    c_shared[bi * tile_size + bj] = graph[global_idx]

    # Synchronize after load
    barrier()

    # Process tile (all three inputs are the same tile, need synchronization)
    process_tile_gpu_device[tile_size, True, True](
        c_shared, c_shared, c_shared,
        tile_size * k, tile_size * k,
        tile_size * k, tile_size * k,
        tile_size * k, tile_size * k
    )

    # Synchronize before store
    barrier()

    # Write back to global memory
    graph[global_idx] = c_shared[bi * tile_size + bj]

fn gpu_partially_independent_kernel[tile_size: Int](
    graph: UnsafePointer[UInt32],
    n: Int,
    k: Int
):
    """
    GPU kernel for partially independent phase.
    Processes row and column tiles relative to diagonal tile k.
    Matches cuda_partially_independent_ from CUDA.
    """
    var i = Int(block_idx.x)
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)

    if i == k:
        return

    # Allocate shared memory for three tiles
    var a_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()
    var c_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()

    # Phase 1: Process row tile (i, k) using (i, k) and (k, k)
    # Load c[i,k] and b[k,k]
    c_shared[bi * tile_size + bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj]
    b_shared[bi * tile_size + bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj]

    barrier()

    process_tile_gpu_device[tile_size, True, True](
        c_shared, c_shared, b_shared,
        i * tile_size, k * tile_size,
        i * tile_size, k * tile_size,
        k * tile_size, k * tile_size
    )

    barrier()

    # Store phase 1 result
    graph[i * tile_size * n + k * tile_size + bi * n + bj] = c_shared[bi * tile_size + bj]

    # Phase 2: Process column tile (k, i) using (k, k) and (k, i)
    # Load c[k,i] and a[k,k]
    c_shared[bi * tile_size + bj] = graph[k * tile_size * n + i * tile_size + bi * n + bj]
    a_shared[bi * tile_size + bj] = graph[k * tile_size * n + k * tile_size + bi * n + bj]

    barrier()

    process_tile_gpu_device[tile_size, True, True](
        c_shared, a_shared, c_shared,
        k * tile_size, i * tile_size,
        k * tile_size, k * tile_size,
        k * tile_size, i * tile_size
    )

    barrier()

    # Store phase 2 result
    graph[k * tile_size * n + i * tile_size + bi * n + bj] = c_shared[bi * tile_size + bj]

fn gpu_independent_kernel[tile_size: Int](
    graph: UnsafePointer[UInt32],
    n: Int,
    k: Int
):
    """
    GPU kernel for independent phase - processes all tiles except row/column k.
    Matches cuda_independent_ from CUDA implementation.
    """
    var j = Int(block_idx.x)
    var i = Int(block_idx.y)
    var bi = Int(thread_idx.y)
    var bj = Int(thread_idx.x)

    if i == k and j == k:
        return

    # Allocate shared memory for three tiles
    var a_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()
    var b_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()
    var c_shared = stack_allocation[
        tile_size * tile_size,
        UInt32,
        address_space=AddressSpace(3),
    ]()

    # Load three tiles: c[i,j], a[i,k], b[k,j]
    c_shared[bi * tile_size + bj] = graph[i * tile_size * n + j * tile_size + bi * n + bj]
    a_shared[bi * tile_size + bj] = graph[i * tile_size * n + k * tile_size + bi * n + bj]
    b_shared[bi * tile_size + bj] = graph[k * tile_size * n + j * tile_size + bi * n + bj]

    barrier()

    # Process tile - use diagonal check if i == j, no synchronization needed (different tiles)
    if i == j:
        process_tile_gpu_device[tile_size, True, False](
            c_shared, a_shared, b_shared,
            i * tile_size, j * tile_size,
            i * tile_size, k * tile_size,
            k * tile_size, j * tile_size
        )
    else:
        process_tile_gpu_device[tile_size, False, False](
            c_shared, a_shared, b_shared,
            i * tile_size, j * tile_size,
            i * tile_size, k * tile_size,
            k * tile_size, j * tile_size
        )

    # No barrier needed - independent tiles write to different locations

    # Write back result
    graph[i * tile_size * n + j * tile_size + bi * n + bj] = c_shared[bi * tile_size + bj]

fn compute_strongest_paths_gpu[tile_size: Int = DEFAULT_GPU_TILE_SIZE](preferences: PreferenceMatrix) raises -> StrongestPathsMatrix:
    """
    Pure Mojo GPU implementation of Schulze strongest paths computation.

    Implements three-phase tiled Floyd-Warshall algorithm on GPU using native
    Mojo GPU kernels. Matches the CUDA implementation in scaling_elections.cu.

    Parameters:
        tile_size: Compile-time tile size for GPU processing (default: 32).

    Args:
        preferences: Input preference matrix.

    Returns:
        StrongestPathsMatrix with computed strongest paths.
    """
    var num_candidates = preferences.num_candidates
    var result = StrongestPathsMatrix(num_candidates)

    # Step 1: Initialize result matrix on CPU (parallelized)
    @parameter
    fn init_paths(i: Int):
        for j in range(num_candidates):
            if i != j:
                var pref_ij = preferences[i, j]
                var pref_ji = preferences[j, i]
                if pref_ij > pref_ji:
                    result[i, j] = pref_ij
                else:
                    result[i, j] = 0

    parallelize[init_paths](num_candidates)

    # Step 2: Create GPU device context
    var ctx = DeviceContext()

    # Step 3: Allocate host and device memory
    var matrix_size = num_candidates * num_candidates
    var host_graph = ctx.enqueue_create_host_buffer[DType.uint32](matrix_size)
    var device_graph = ctx.enqueue_create_buffer[DType.uint32](matrix_size)

    # Step 4: Copy initialized data to host buffer
    for i in range(matrix_size):
        host_graph[i] = result.data[i]

    # Step 5: Copy from host buffer to device buffer
    host_graph.enqueue_copy_to(device_graph)
    ctx.synchronize()

    # Get raw pointer from device buffer for kernel access
    var graph_ptr = device_graph.unsafe_ptr()

    # Step 6: Execute tiled Floyd-Warshall on GPU
    var num_tiles = (num_candidates + tile_size - 1) // tile_size
    var block_dim_tuple = (tile_size, tile_size, 1)

    for k in range(num_tiles):
        # Phase 1: Diagonal tile (sequential, 1 block)
        ctx.enqueue_function[gpu_diagonal_kernel[tile_size]](
            graph_ptr, num_candidates, k,
            grid_dim=(1, 1, 1),
            block_dim=block_dim_tuple
        )

        # Phase 2: Partially independent tiles (num_tiles blocks)
        ctx.enqueue_function[gpu_partially_independent_kernel[tile_size]](
            graph_ptr, num_candidates, k,
            grid_dim=(num_tiles, 1, 1),
            block_dim=block_dim_tuple
        )

        # Phase 3: Independent tiles (num_tiles x num_tiles blocks)
        ctx.enqueue_function[gpu_independent_kernel[tile_size]](
            graph_ptr, num_candidates, k,
            grid_dim=(num_tiles, num_tiles, 1),
            block_dim=block_dim_tuple
        )

        # Synchronize after each k iteration
        ctx.synchronize()

    # Step 7: Copy results back from GPU to CPU
    ctx.synchronize()  # Ensure all GPU operations complete

    # Copy from device buffer to host buffer
    device_graph.enqueue_copy_to(host_graph)
    ctx.synchronize()

    # Copy from host buffer to result
    for i in range(matrix_size):
        result.data[i] = UInt32(host_graph[i])

    return result^

fn compute_strongest_paths_gpu_dispatch[*allowed_sizes: Int](preferences: PreferenceMatrix, tile_size: Int) raises -> StrongestPathsMatrix:
    """
    Runtime dispatcher for GPU implementation.
    Pre-compiles variants for allowed tile sizes and dispatches based on runtime value.

    Parameters:
        allowed_sizes: Variadic list of compile-time tile sizes to support.

    Args:
        preferences: Input preference matrix.
        tile_size: Runtime tile size selection.

    Returns:
        Computed strongest paths matrix.
    """
    alias sizes = VariadicList(allowed_sizes)

    @parameter
    for size in sizes:
        if tile_size == size:
            return compute_strongest_paths_gpu[size](preferences)

    # Fallback for unsupported sizes
    print("Warning: Unsupported GPU tile size", tile_size, "- falling back to default", DEFAULT_GPU_TILE_SIZE)
    return compute_strongest_paths_gpu[DEFAULT_GPU_TILE_SIZE](preferences)

fn validate_results(result: StrongestPathsMatrix, baseline: StrongestPathsMatrix) -> Bool:
    """Check if two results match."""
    var n = result.num_candidates
    for i in range(n):
        for j in range(n):
            if result[i, j] != baseline[i, j]:
                return False
    return True

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
    print("Usage: mojo scaling_elections.mojo [OPTIONS]")
    print()
    print("Options:")
    print("  --num-candidates N    Number of candidates (default: 128)")
    print("  --num-voters N        Number of voters (default: 2000)")
    print("                        Set to 0 for instant random preference matrix generation")
    print("  --run-cpu             Run CPU implementations (tiled + SIMD-vectorized)")
    print("  --run-gpu             Run GPU implementation")
    print("  --no-serial           Skip serial baseline")
    print("  --cpu-tile-size N     CPU tile size: 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 (default: 16)")
    print("  --gpu-tile-size N     GPU tile size: 4, 8, 12, 16, 24, 32, 48, 64 (default: 32)")
    print("  --warmup N            Number of warmup iterations (default: 1)")
    print("  --repeat N            Number of benchmark iterations (default: 1)")
    print("  --help, -h            Show this help message")
    print()
    print("Examples:")
    print("  pixi run mojo scaling_elections.mojo --num-candidates 256 --num-voters 4000")
    print("  pixi run mojo scaling_elections.mojo --num-candidates 4096 --run-cpu --run-gpu")
    print("  pixi run mojo scaling_elections.mojo --num-candidates 16384 --num-voters 0 --run-gpu --no-serial")
    print("  pixi run mojo scaling_elections.mojo --run-cpu --cpu-tile-size 32")

fn main():
    """
    Main function demonstrating the Schulze voting algorithm.
    Tests CPU and GPU implementations with performance benchmarking.
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
    var cpu_tile_size = parse_int_arg(args, "--cpu-tile-size", DEFAULT_CPU_TILE_SIZE)
    var gpu_tile_size = parse_int_arg(args, "--gpu-tile-size", DEFAULT_GPU_TILE_SIZE)
    var warmup = parse_int_arg(args, "--warmup", 1)
    var repeat = parse_int_arg(args, "--repeat", 1)
    var run_cpu = has_flag(args, "--run-cpu")
    var run_gpu = has_flag(args, "--run-gpu")
    var no_serial = has_flag(args, "--no-serial")
    var run_serial = not no_serial

    # Detect GPU availability
    print("=== Schulze Voting Algorithm (Mojo) ===")
    print()

    if run_gpu:
        if not has_accelerator():
            print("✗ No GPU detected - GPU mode disabled")
            run_gpu = False

    # Validate CPU tile size
    if cpu_tile_size not in ALLOWED_CPU_TILE_SIZES:
        print("Warning: --cpu-tile-size must be 4, 8, 12, 16, 24, 32, 48, 64, 96, or 128. Using default:", DEFAULT_CPU_TILE_SIZE)
        cpu_tile_size = DEFAULT_CPU_TILE_SIZE
        print()

    # Validate GPU tile size
    if run_gpu and gpu_tile_size not in ALLOWED_GPU_TILE_SIZES:
        print("Warning: --gpu-tile-size must be 4, 8, 12, 16, 24, 32, 48, or 64. Using default:", DEFAULT_GPU_TILE_SIZE)
        gpu_tile_size = DEFAULT_GPU_TILE_SIZE
        print()

    # Validate candidates (must be at least 4)
    if num_candidates < 4:
        print("Error: num_candidates must be at least 4")
        return

    print("Configuration:")
    var voters_str = String(num_voters) + " voters" if num_voters > 0 else "random"
    print("  Problem size: " + String(num_candidates) + " candidates × " + voters_str)
    print("  CPU tile: " + String(cpu_tile_size) + " × " + String(cpu_tile_size))
    if run_gpu:
        print("  GPU tile: " + String(gpu_tile_size) + " × " + String(gpu_tile_size))
    print("  Warmup: " + String(warmup) + ", Repeat: " + String(repeat))
    print()

    print("Generating preferences...")
    var preferences = generate_random_preferences(num_candidates, num_voters)

    # Benchmarking section
    print()
    print("─── Benchmarking ───────────────────────────────────")
    print()

    # Store baseline for validation
    var baseline = StrongestPathsMatrix(0)
    var has_baseline = False

    # Store winner info
    var winner = 0
    var ranking = List[Int]()
    var has_winner = False

    # Run serial baseline (unless --no-serial)
    if run_serial:
        try:
            print("→ Serial (Mojo)")
            run_warmup(compute_strongest_paths_serial, preferences, warmup)

            var avg_time = run_and_average(compute_strongest_paths_serial, preferences, repeat, baseline)

            if repeat > 1:
                print("  Run:     " + format_time(avg_time) + " (avg of " + String(repeat) + ") │ " + format_throughput_from_ns(avg_time, num_candidates))
            else:
                print("  Run:     " + format_time(avg_time) + " │ " + format_throughput_from_ns(avg_time, num_candidates))

            has_baseline = True
            var result_tuple = get_winner_and_ranking(baseline)
            winner = result_tuple[0]
            ranking = result_tuple[1].copy()
            has_winner = True
            print()
        except e:
            print("  ✗ Serial failed: " + String(e))
            print()

    # Run CPU implementation
    if run_cpu:
        try:
            print("→ Tiled CPU (Mojo)")

            # Warmup
            for i in range(warmup):
                var start_time = perf_counter_ns()
                # TODO: Find a way to pass ALLOWED_CPU_TILE_SIZES, instead of repeating here
                _ = compute_strongest_paths_tiled_cpu_dispatch[4, 8, 12, 16, 24, 32, 48, 64, 96, 128](preferences, cpu_tile_size)
                var elapsed_ns = perf_counter_ns() - start_time
                if warmup > 1:
                    print("  Warm-up " + String(i+1) + "/" + String(warmup) + ": " + format_time(elapsed_ns))
                else:
                    print("  Warm-up: " + format_time(elapsed_ns))

            # Benchmark runs
            var times = List[Int]()
            var cpu_result = StrongestPathsMatrix(0)
            for _ in range(repeat):
                var start_time = perf_counter_ns()
                # TODO: Find a way to pass ALLOWED_CPU_TILE_SIZES, instead of repeating here
                cpu_result = compute_strongest_paths_tiled_cpu_dispatch[4, 8, 12, 16, 24, 32, 48, 64, 96, 128](preferences, cpu_tile_size)
                var elapsed_ns = perf_counter_ns() - start_time
                times.append(elapsed_ns)

            # Calculate average
            var total_time: Int = 0
            for i in range(len(times)):
                total_time += times[i]
            var avg_time = total_time // repeat

            if repeat > 1:
                print("  Run:     " + format_time(avg_time) + " (avg of " + String(repeat) + ") │ " + format_throughput_from_ns(avg_time, num_candidates))
            else:
                print("  Run:     " + format_time(avg_time) + " │ " + format_throughput_from_ns(avg_time, num_candidates))

            if has_baseline and validate_results(cpu_result, baseline):
                print("  ✓ Results validated")
            elif has_baseline:
                print("  ✗ Results don't match baseline!")

            if not has_winner:
                var result_tuple = get_winner_and_ranking(cpu_result)
                winner = result_tuple[0]
                ranking = result_tuple[1].copy()
                has_winner = True

            print()
        except e:
            print("  ✗ CPU failed: " + String(e))
            print()

    # Run SIMD CPU implementation
    if run_cpu:
        try:
            print("→ Tiled CPU+SIMD (Mojo)")

            # Warmup
            for i in range(warmup):
                var start_time = perf_counter_ns()
                _ = compute_strongest_paths_tiled_cpu_simd[DEFAULT_CPU_TILE_SIZE](preferences)
                var elapsed_ns = perf_counter_ns() - start_time
                if warmup > 1:
                    print("  Warm-up " + String(i+1) + "/" + String(warmup) + ": " + format_time(elapsed_ns))
                else:
                    print("  Warm-up: " + format_time(elapsed_ns))

            # Benchmark runs
            var times = List[Int]()
            var cpu_simd_result = StrongestPathsMatrix(0)
            for _ in range(repeat):
                var start_time = perf_counter_ns()
                cpu_simd_result = compute_strongest_paths_tiled_cpu_simd[DEFAULT_CPU_TILE_SIZE](preferences)
                var elapsed_ns = perf_counter_ns() - start_time
                times.append(elapsed_ns)

            # Calculate average
            var total_time: Int = 0
            for i in range(len(times)):
                total_time += times[i]
            var avg_time = total_time // repeat

            if repeat > 1:
                print("  Run:     " + format_time(avg_time) + " (avg of " + String(repeat) + ") │ " + format_throughput_from_ns(avg_time, num_candidates))
            else:
                print("  Run:     " + format_time(avg_time) + " │ " + format_throughput_from_ns(avg_time, num_candidates))

            if has_baseline and validate_results(cpu_simd_result, baseline):
                print("  ✓ Results validated")
            elif has_baseline:
                print("  ✗ Results don't match baseline!")

            if not has_winner:
                var result_tuple = get_winner_and_ranking(cpu_simd_result)
                winner = result_tuple[0]
                ranking = result_tuple[1].copy()
                has_winner = True

            print()
        except e:
            print("  ✗ CPU+SIMD failed: " + String(e))
            print()

    # Run GPU implementation
    if run_gpu:
        try:
            print("→ Tiled GPU (Mojo)")

            # Warmup
            for i in range(warmup):
                var start_time = perf_counter_ns()
                # TODO: Find a way to pass ALLOWED_GPU_TILE_SIZES, instead of repeating here
                _ = compute_strongest_paths_gpu_dispatch[4, 8, 12, 16, 24, 32, 48, 64](preferences, gpu_tile_size)
                var elapsed_ns = perf_counter_ns() - start_time
                if warmup > 1:
                    print("  Warm-up " + String(i+1) + "/" + String(warmup) + ": " + format_time(elapsed_ns))
                else:
                    print("  Warm-up: " + format_time(elapsed_ns))

            # Benchmark runs
            var times = List[Int]()
            var gpu_result = StrongestPathsMatrix(0)
            for _ in range(repeat):
                var start_time = perf_counter_ns()
                # TODO: Find a way to pass ALLOWED_GPU_TILE_SIZES, instead of repeating here
                gpu_result = compute_strongest_paths_gpu_dispatch[4, 8, 12, 16, 24, 32, 48, 64](preferences, gpu_tile_size)
                var elapsed_ns = perf_counter_ns() - start_time
                times.append(elapsed_ns)

            # Calculate average
            var total_time: Int = 0
            for i in range(len(times)):
                total_time += times[i]
            var avg_time = total_time // repeat

            if repeat > 1:
                print("  Run:     " + format_time(avg_time) + " (avg of " + String(repeat) + ") │ " + format_throughput_from_ns(avg_time, num_candidates))
            else:
                print("  Run:     " + format_time(avg_time) + " │ " + format_throughput_from_ns(avg_time, num_candidates))

            if has_baseline and validate_results(gpu_result, baseline):
                print("  ✓ Results validated")
            elif has_baseline:
                print("  ✗ Results don't match baseline!")

            if not has_winner:
                var result_tuple = get_winner_and_ranking(gpu_result)
                winner = result_tuple[0]
                ranking = result_tuple[1].copy()
                has_winner = True

            print()
        except e:
            print("  ✗ GPU failed: " + String(e))
            print()

    # Compute fallback if no results
    if not has_winner:
        try:
            var fallback = compute_strongest_paths_serial(preferences)
            var result_tuple = get_winner_and_ranking(fallback)
            winner = result_tuple[0]
            ranking = result_tuple[1].copy()
        except e:
            print("  ✗ Failed to compute results: " + String(e))
            return

    # Display election results
    print("─── Election Results ───────────────────────────────")
    print()
    print("  Winner: Candidate #" + String(winner))
    if num_candidates >= 5:
        print("  Top 5:  #" + String(ranking[0]) + ", #" + String(ranking[1]) + ", #" + String(ranking[2]) + ", #" + String(ranking[3]) + ", #" + String(ranking[4]))

    print()
