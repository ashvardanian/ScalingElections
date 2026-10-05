"""Native benchmarks for ballot tallying, Schulze, and Kemeny-Young.

Input generation is excluded; timings include allocations, transfers, and reconstruction.
Run `pixi run bench --help` for method, backend, and repetition selectors.
"""

from std.random.philox import Random
from std.sys import argv, has_accelerator, num_logical_cores
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from ballots import (
    PreferenceMatrix,
    SeedGraph,
    TALLY_MAX_CANDIDATES,
    generate_random_preferences,
    tally_ballots_cpu,
    tally_ballots_gpu,
)
from kemeny import KemenySolution, kemeny_ranking, kemeny_ranking_gpu
from schulze import (
    TILE_SIZE,
    compute_election_results,
    compute_strongest_paths_gpu,
    compute_strongest_paths_serial,
    compute_strongest_paths_tiled_cpu,
    compute_strongest_paths_tiled_cpu_simd,
)


def profile[
    Result: Movable & Deinitable, Func: def() raises -> Result
](label: String, function: Func, warmup: Int, repeat: Int, work: Float64, unit: String) raises -> Result:
    """Times complete calls and retains the last result for validation."""
    print("→", label)
    for _ in range(warmup):
        _ = function()
    var start = perf_counter_ns()
    var result = function()
    var elapsed = perf_counter_ns() - start
    print("  sample_ns", elapsed)
    var total = elapsed
    for _ in range(1, repeat):
        start = perf_counter_ns()
        result = function()
        elapsed = perf_counter_ns() - start
        print("  sample_ns", elapsed)
        total += elapsed
    var average = total // repeat
    print("  mean_ns", average, "│", format_time(average))
    if work > 0 and average > 0:
        print("  rate", work * 1e9 / Float64(average), unit)
    return result^


def check_matrix(actual: PreferenceMatrix, expected: PreferenceMatrix) raises:
    for row in range(expected.num_candidates):
        for column in range(expected.num_candidates):
            if actual[row, column] != expected[row, column]:
                raise Error("Backend matrices disagree")
    print("  ✓ Matrices match")


def run_ballots(n: Int, voters: Int, seed: Int, selector: String, warmup: Int, repeat: Int) raises:
    var rankings = List[UInt32]()
    rankings.resize(n * voters, 0)
    var generator = Random(seed=UInt64(seed))
    for ballot in range(voters):
        var base = ballot * n
        for candidate in range(n):
            rankings[base + candidate] = UInt32(candidate)
        for upper in range(n - 1, 0, -1):
            var chosen = Int(generator.step()[0] % UInt32(upper + 1))
            var held = rankings[base + upper]
            rankings[base + upper] = rankings[base + chosen]
            rankings[base + chosen] = held

    var baseline = PreferenceMatrix(0)
    var backends: List[String] = ["CPU", "GPU"]
    for backend in backends:
        if not selected_by(selector, backend):
            continue
        if backend == "GPU" and n > TALLY_MAX_CANDIDATES:
            print("GPU tally supports at most", TALLY_MAX_CANDIDATES, "candidates")
            continue
        if backend == "GPU" and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> PreferenceMatrix:
            return tally_ballots_gpu(rankings, voters, n) if backend == "GPU" else tally_ballots_cpu(
                rankings, voters, n
            )

        var result = profile[PreferenceMatrix](backend, calculate, warmup, repeat, Float64(voters), "ballots/s")
        for row in range(n):
            if result[row, row] != 0:
                raise Error("Nonzero tally diagonal")
            for column in range(row + 1, n):
                if UInt64(result[row, column]) + UInt64(result[column, row]) != UInt64(voters):
                    raise Error("Tally did not count every ballot")
        if baseline.num_candidates:
            check_matrix(result, baseline)
        else:
            baseline = result^
    if baseline.num_candidates == 0:
        raise Error("No selected backend could run")


def run_schulze(preferences: PreferenceMatrix, selector: String, warmup: Int, repeat: Int) raises:
    var baseline = PreferenceMatrix(0)
    var backends: List[String] = ["Serial", "Tiled CPU", "Tiled CPU+SIMD", "Tiled GPU"]
    var n = preferences.num_candidates
    for backend in backends:
        if not selected_by(selector, backend):
            continue
        if "GPU" in backend and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> PreferenceMatrix:
            if backend == "Serial":
                return compute_strongest_paths_serial[SeedGraph.winning_votes](preferences)
            if backend == "Tiled CPU":
                return compute_strongest_paths_tiled_cpu[TILE_SIZE](preferences)
            if backend == "Tiled CPU+SIMD":
                return compute_strongest_paths_tiled_cpu_simd[TILE_SIZE](preferences)
            return compute_strongest_paths_gpu[TILE_SIZE](preferences)

        var result = profile[PreferenceMatrix](backend, calculate, warmup, repeat, Float64(n) ** 3, "cells/s")
        if baseline.num_candidates:
            check_matrix(result, baseline)
        else:
            baseline = result^
    if baseline.num_candidates == 0:
        raise Error("No selected backend could run")
    var outcome = compute_election_results(baseline)
    var top = List[Int]()
    for place in range(min(5, len(outcome.ranking))):
        top.append(outcome.ranking[place])
    print("Winner:", outcome.winner, "Top candidates:", top)


def run_kemeny(preferences: PreferenceMatrix, selector: String, warmup: Int, repeat: Int) raises:
    var baseline = KemenySolution(List[Int](), -1)
    var backends: List[String] = ["CPU", "GPU"]
    var n = preferences.num_candidates
    for backend in backends:
        if not selected_by(selector, backend):
            continue
        if backend == "GPU" and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> KemenySolution:
            return kemeny_ranking_gpu(preferences) if backend == "GPU" else kemeny_ranking(preferences)

        var result = profile[KemenySolution](backend, calculate, warmup, repeat, 0, "")
        var score = UInt64(0)
        var seen = UInt64(0)
        for earlier in range(n):
            var candidate = result.ranking[earlier]
            if candidate < 0 or candidate >= n or seen & (UInt64(1) << UInt64(candidate)):
                raise Error("Kemeny ranking is not a permutation")
            seen |= UInt64(1) << UInt64(candidate)
            for later in range(earlier + 1, n):
                score += UInt64(preferences[result.ranking[later], candidate])
        if score != UInt64(result.score):
            raise Error("Kemeny ranking disagrees with its score")
        if baseline.score >= 0:
            if result.score != baseline.score or result.ranking != baseline.ranking:
                raise Error("Backend Kemeny results disagree")
            print("  ✓ Rankings and scores match")
        else:
            baseline = result^
    if baseline.score < 0:
        raise Error("No selected backend could run")
    print("Score:", baseline.score, "Ranking:", baseline.ranking)


def format_time(elapsed_ns: Int) -> String:
    """Formats a duration in milliseconds, switching to seconds past one thousand."""
    var elapsed_milliseconds = elapsed_ns // 1_000_000
    if elapsed_milliseconds < 1000:
        return "{} ms".format(Float64(elapsed_ns) / 1_000_000.0)
    else:
        var elapsed_seconds = Float64(elapsed_ns) / 1_000_000_000.0
        var hundredths_of_second = Int(elapsed_seconds * 100.0)
        return "{}.{}{} s".format(
            hundredths_of_second // 100, (hundredths_of_second % 100) // 10, hundredths_of_second % 10
        )


def parse_int_arg(args: Span[StaticString, ImmStaticOrigin], flag: String, default: Int) raises -> Int:
    """Parse an integer command-line argument, raising if it is malformed."""
    for index in range(len(args)):
        if String(args[index]) != flag:
            continue
        if index + 1 >= len(args):
            raise Error(String(flag, " needs a value"))
        try:
            return Int(String(args[index + 1]))
        except:
            raise Error(String("Invalid value for ", flag, ": ", args[index + 1]))
    return default


def parse_text_arg(args: Span[StaticString, ImmStaticOrigin], flag: String, default: String) raises -> String:
    """Parse a string argument, raising when its value is missing."""
    for index in range(len(args)):
        if String(args[index]) != flag:
            continue
        if index + 1 >= len(args):
            raise Error(String(flag, " needs a value"))
        return String(args[index + 1])
    return default


def selected_by(pattern: String, name: String) -> Bool:
    """Whether a backend name matches the selector, case-insensitively."""
    if pattern == ".":
        return True
    for part in pattern.lower().split(","):
        if part and part in name.lower():
            return True
    return False


def has_flag(args: Span[StaticString, ImmStaticOrigin], flag: String) -> Bool:
    """Check if a flag exists in command-line arguments."""
    for index in range(len(args)):
        if String(args[index]) == flag:
            return True
    return False


def reject_unknown_flags(
    args: Span[StaticString, ImmStaticOrigin],
) raises:
    """Rejects unrecognized flags, so a typo cannot silently use defaults."""
    var valued: List[String] = [
        "--method",
        "--num-candidates",
        "--num-voters",
        "--warmup",
        "--repeat",
        "--seed",
        "--filter",
        "-k",
    ]
    var bare: List[String] = ["--help", "-h"]
    var index = 1
    while index < len(args):
        var argument = String(args[index])
        var matched = False

        for slot in range(len(valued)):
            if not matched and argument == valued[slot]:
                matched = True
                index += 1

        for slot in range(len(bare)):
            if argument == bare[slot]:
                matched = True

        if not matched:
            raise Error(String("Unknown option: ", argument))
        index += 1


comptime USAGE = """Usage: scalingelections [OPTIONS]

  --method NAME        ballots, schulze (default), or kemeny
  --num-candidates N   Number of candidates (default: 128)
  --num-voters N       Number of voters (default: 2000); 0 draws matrix counts in [0, 350M]
  -k, --filter TEXT    Comma-separated backend substrings, case-insensitive (default: all)
                      Ballots/Kemeny: CPU, GPU; Schulze: Serial, Tiled CPU, Tiled CPU+SIMD, Tiled GPU
  --warmup N           Warmup iterations (default: 1)
  --repeat N           Measured iterations (default: 1)
  --seed N             Reproducible input seed (default: 42)
  --help, -h           Show help
"""


def main() raises:
    var args = argv()
    if has_flag(args, "--help") or has_flag(args, "-h"):
        print(USAGE)
        return
    reject_unknown_flags(args)
    var method = parse_text_arg(args, "--method", "schulze")
    var n = parse_int_arg(args, "--num-candidates", 128)
    var voters = parse_int_arg(args, "--num-voters", 2000)
    var warmup = parse_int_arg(args, "--warmup", 1)
    var repeat = parse_int_arg(args, "--repeat", 1)
    var seed = parse_int_arg(args, "--seed", 42)
    var selector = parse_text_arg(args, "--filter", ".")
    selector = parse_text_arg(args, "-k", selector)
    if n < 1 or voters < 0 or warmup < 0 or repeat < 1:
        raise Error("Candidates and repeat must be positive; voters and warmup cannot be negative")
    if method != "ballots" and method != "schulze" and method != "kemeny":
        raise Error("--method must be ballots, schulze, or kemeny")
    if method == "ballots" and voters == 0:
        raise Error("--num-voters must be positive for ballot benchmarks")
    if method == "kemeny" and n > 33:
        raise Error("Kemeny supports at most 33 candidates")
    print("Method:", method, "Candidates:", n, "Voters:", voters, "Seed:", seed)
    print("Warmup:", warmup, "Repeat:", repeat, "CPU threads:", num_logical_cores())
    if has_accelerator():
        var ctx = DeviceContext()
        print("GPU:", ctx.name())
    if method == "ballots":
        run_ballots(n, voters, seed, selector, warmup, repeat)
    else:
        var preferences = generate_random_preferences(n, voters, seed)
        if method == "schulze":
            run_schulze(preferences, selector, warmup, repeat)
        else:
            run_kemeny(preferences, selector, warmup, repeat)
