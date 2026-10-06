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
from ballots import generate_preferences, resolve_tally_score_type

NATIONAL_ELECTORATE = 350_000_000
"""Voters in a national election, the largest count a random benchmark matrix cell is drawn up to."""


def benchmark_implementation[Input, Result](
    callback: Callable[[Input], Result],
    inputs: Input,
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
        print(f"  sample_ns {elapsed:d}")
        total += elapsed
        remaining -= 1
        if remaining == 0:
            break
    average = total // repeat
    print(f"  mean_ns {average:d} │ {average / 1e6:.3f} ms")
    return average, result


def selected_by(pattern: str, name: str) -> bool:
    """Match a backend name against the command-line filter."""
    return pattern == "." or any(part and part in name.lower() for part in pattern.lower().split(","))


def main() -> None:
    """Benchmark selected backends and verify that their results agree."""
    parser = argparse.ArgumentParser(description="Benchmark ballots, Schulze, and Kemeny-Young")
    parser.add_argument("--method", choices=("ballots", "schulze", "kemeny"), default="schulze")
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument(
        "--num-voters",
        type=int,
        default=2000,
        help=f"0 draws matrix counts in [0, {NATIONAL_ELECTORATE:,}]",
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
    parser.add_argument("--score-type", choices=tuple(scalingelections.ScoreType), default="auto")
    args = parser.parse_args()
    n, voters = args.num_candidates, args.num_voters
    if n < 1 or voters < 0 or args.warmup < 0 or args.repeat < 1:
        parser.error("Candidates and repeat must be positive; voters and warmup cannot be negative")
    if args.method == "ballots" and voters == 0:
        parser.error("--num-voters must be positive for ballot benchmarks")
    if args.method == "kemeny" and n > kemeny.KEMENY_MAX_CANDIDATES:
        parser.error(f"Kemeny is exact from 1 to {kemeny.KEMENY_MAX_CANDIDATES} candidates")

    targets = []
    for implementation in scalingelections.available_implementations():
        for backend in scalingelections.available_backends(implementation=implementation):
            backend = scalingelections.Backend(backend)
            name = f"{implementation.upper()} {backend.value.upper()}"
            if selected_by(args.filter, name):
                targets.append((name, implementation, backend))
    if not targets:
        parser.error("No selected backend is available")

    print(f"Method: {args.method} Candidates: {n:d} Voters: {voters:d} Seed: {args.seed:d}")
    print(f"Warmup: {args.warmup:d} Repeat: {args.repeat:d}")
    generator = np.random.default_rng(args.seed)
    if args.method == "ballots":
        inputs = np.empty((voters, n), dtype=np.uint32)
        for ballot in inputs:
            ballot[:] = generator.permutation(n)
    else:
        inputs = (
            generate_preferences(n, voters, generator)
            if voters
            else generator.integers(0, NATIONAL_ELECTORATE + 1, (n, n), dtype=np.uint32)
        )
        np.fill_diagonal(inputs, 0)
    operation = {
        "ballots": scalingelections.tally_ballots,
        "schulze": scalingelections.compute_strongest_paths,
        "kemeny": scalingelections.compute_kemeny_ranking,
    }[args.method]
    score_type = scalingelections.ScoreType(args.score_type)
    if args.method == "ballots":
        resolved_score_type = resolve_tally_score_type(voters, score_type)
    else:
        resolve_score_type = {
            "schulze": schulze.resolve_score_type,
            "kemeny": kemeny.resolve_score_type,
        }[args.method]
        resolved_score_type = resolve_score_type(inputs, score_type)
    print(f"Score bits: {resolved_score_type.bits:d}")

    baseline = None
    for name, implementation, backend in targets:
        print(f"→ {name}")
        callback = partial(operation, implementation=implementation, backend=backend, score_type=score_type)
        average, result = benchmark_implementation(callback, inputs, args.warmup, args.repeat)
        if args.method == "ballots":
            if np.any(np.diag(result)) or np.any((result + result.T)[np.triu_indices(n, 1)] != voters):
                raise RuntimeError("Tally did not count every ballot")
            print(f"  rate {voters * 1e9 / average:.3f} ballots/s")
        elif args.method == "schulze":
            print(f"  rate {n**3 * 1e9 / average:.3f} cells/s")
        else:
            ranking, score, winners, multiplicity = result
            if sorted(ranking) != list(range(n)) or score != sum(
                int(inputs[ranking[later], ranking[earlier]]) for earlier in range(n) for later in range(earlier + 1, n)
            ):
                raise RuntimeError("Kemeny ranking disagrees with its score")
        if baseline is not None:
            matches = result == baseline if args.method == "kemeny" else np.array_equal(result, baseline)
            if not matches:
                raise RuntimeError("Backend results disagree")
            print("  ✓ Results match")
        else:
            baseline = result

    assert baseline is not None
    if args.method == "schulze":
        winners, ranking = schulze.compute_election_results(list(range(n)), baseline)
        print(f"Winners: {winners} Top candidates: {ranking[:5]}")
    elif args.method == "kemeny":
        print(f"Score: {baseline.score:d} Ranking: {baseline.ranking}")
        print(f"Winners: {baseline.winners} Multiplicity: {baseline.multiplicity.value}")


if __name__ == "__main__":
    main()
