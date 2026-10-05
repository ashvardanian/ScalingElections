"""Benchmark dispatched ballot, Schulze, and Kemeny implementations.

Input generation is excluded; timings include allocations, transfers, and reconstruction.
Backend filters are comma-separated, case-insensitive substrings, as in `cli.mojo`.
"""

import argparse
import time
from collections.abc import Callable
from functools import partial

import numpy as np

import kemeny
import scalingelections
import schulze
from ballots import generate_preferences


def benchmark_implementation[Result](
    callback: Callable[[np.ndarray], Result],
    inputs: np.ndarray,
    warmup: int,
    repeat: int,
) -> tuple[int, Result]:
    """Times complete calls and retains the last result for validation."""
    if repeat < 1:
        raise ValueError("At least one timed iteration is required")
    for _ in range(warmup):
        callback(inputs)
    total = 0
    remaining = repeat
    while True:
        start = time.perf_counter_ns()
        result = callback(inputs)
        elapsed = time.perf_counter_ns() - start
        print("  sample_ns", elapsed)
        total += elapsed
        remaining -= 1
        if remaining == 0:
            break
    average = total // repeat
    print("  mean_ns", average, "│", f"{average / 1e6:.3f} ms")
    return average, result


def selected_by(pattern: str, name: str) -> bool:
    return pattern == "." or any(part and part in name.lower() for part in pattern.lower().split(","))


def main():
    parser = argparse.ArgumentParser(description="Benchmark ballots, Schulze, and Kemeny-Young")
    parser.add_argument("--method", choices=("ballots", "schulze", "kemeny"), default="schulze")
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument(
        "--num-voters",
        type=int,
        default=2000,
        help="0 draws matrix counts in [0, 350M]",
    )
    parser.add_argument(
        "-k",
        "--filter",
        default=".",
        help="Comma-separated backend substrings, case-insensitive",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-bits", choices=("auto", "16", "32", "64"))
    args = parser.parse_args()
    n, voters = args.num_candidates, args.num_voters
    if n < 1 or voters < 0 or args.warmup < 0 or args.repeat < 1:
        parser.error("Candidates and repeat must be positive; voters and warmup cannot be negative")
    if args.method == "ballots" and args.score_bits is not None:
        parser.error("--score-bits applies only to Schulze and Kemeny")
    if args.method == "ballots" and voters == 0:
        parser.error("--num-voters must be positive for ballot benchmarks")
    if args.method == "kemeny" and n > 33:
        parser.error("Kemeny supports at most 33 candidates")

    targets = []
    for implementation in scalingelections.available_implementations():
        for backend in scalingelections.available_backends(implementation=implementation):
            backend = scalingelections.Backend(backend)
            name = f"{implementation.upper()} {backend.value.upper()}"
            if selected_by(args.filter, name):
                targets.append((name, implementation, backend))
    if not targets:
        parser.error("No selected backend is available")

    print("Method:", args.method, "Candidates:", n, "Voters:", voters, "Seed:", args.seed)
    print("Warmup:", args.warmup, "Repeat:", args.repeat)
    generator = np.random.default_rng(args.seed)
    if args.method == "ballots":
        inputs = np.empty((voters, n), dtype=np.uint32)
        for ballot in inputs:
            ballot[:] = generator.permutation(n)
    else:
        inputs = (
            generate_preferences(n, voters, generator)
            if voters
            else generator.integers(0, 350_000_001, (n, n), dtype=np.uint32)
        )
        np.fill_diagonal(inputs, 0)
    operation = {
        "ballots": scalingelections.tally_ballots,
        "schulze": scalingelections.compute_strongest_paths,
        "kemeny": scalingelections.compute_kemeny_ranking,
    }[args.method]
    options = {}
    if args.method in ("schulze", "kemeny"):
        options["score_type"] = scalingelections.ScoreType(
            "auto" if args.score_bits in (None, "auto") else f"uint{args.score_bits}"
        )
        resolve_score_type = {
            "schulze": schulze.resolve_score_type,
            "kemeny": kemeny.resolve_score_type,
        }[args.method]
        print("Score bits:", resolve_score_type(inputs, options["score_type"]).bits)

    baseline = None
    for name, implementation, backend in targets:
        print("→", name)
        callback = partial(operation, implementation=implementation, backend=backend, **options)
        average, result = benchmark_implementation(callback, inputs, args.warmup, args.repeat)
        if args.method == "ballots":
            if np.any(np.diag(result)) or np.any(
                (result.astype(np.uint64) + result.T)[np.triu_indices(n, 1)] != voters
            ):
                raise RuntimeError("Tally did not count every ballot")
            print("  rate", voters * 1e9 / average, "ballots/s")
        elif args.method == "schulze":
            print("  rate", n**3 * 1e9 / average, "cells/s")
        else:
            ranking, score = result
            if sorted(ranking) != list(range(n)) or score != sum(
                int(inputs[ranking[later], ranking[earlier]]) for earlier in range(n) for later in range(earlier + 1, n)
            ):
                raise RuntimeError("Kemeny ranking disagrees with its score")
        if baseline is not None:
            matches = (
                result[1] == baseline[1] and np.array_equal(result[0], baseline[0])
                if args.method == "kemeny"
                else np.array_equal(result, baseline)
            )
            if not matches:
                raise RuntimeError("Backend results disagree")
            print("  ✓ Results match")
        else:
            baseline = result

    assert baseline is not None
    if args.method == "schulze":
        winner, ranking = schulze.compute_election_results(list(range(n)), baseline)
        print("Winner:", winner, "Top candidates:", ranking[:5])
    elif args.method == "kemeny":
        print("Score:", baseline[1], "Ranking:", baseline[0])


if __name__ == "__main__":
    main()
