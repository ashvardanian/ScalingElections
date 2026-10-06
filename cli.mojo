"""Native benchmarks for ballot tallying, Schulze, and Kemeny-Young.

Input generation is excluded; timings include allocations, transfers, and reconstruction.
Run `pixi run bench --help` for method, backend, and repetition selectors.
"""

from std.collections import StringDict
from std.random.philox import Random
from std.sys import argv, has_accelerator, num_logical_cores
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

import ballots
import kemeny
import schulze
from ballots import Arithmetic, Backend, UInt32VoteMatrix, VoteMatrix, ScoreType
from kemeny import KemenySolution


def profile[
    ResultType: Movable & Deinitable, Function: def() raises -> ResultType
](label: String, function: Function, warmup: Int, repeat: Int, work: Float64, unit: String) raises -> ResultType:
    """Times complete calls and retains the last result for validation."""
    print("→ {}".format(label))
    for _ in range(warmup):
        _ = function()
    var start = perf_counter_ns()
    var result = function()
    var elapsed = perf_counter_ns() - start
    print("  sample_ns {}".format(elapsed))
    var total = elapsed
    for _ in range(1, repeat):
        start = perf_counter_ns()
        result = function()
        elapsed = perf_counter_ns() - start
        print("  sample_ns {}".format(elapsed))
        total += elapsed
    var average = total // repeat
    print("  mean_ns {} │ {}".format(average, format_time(average)))
    if work > 0 and average > 0:
        print("  rate {} {}".format(work * 1e9 / Float64(average), unit))
    return result^


def check_matrix[CountDataType: DType](actual: VoteMatrix[CountDataType], expected: VoteMatrix[CountDataType]) raises:
    for row in range(expected.num_candidates):
        for column in range(expected.num_candidates):
            if actual[row, column] != expected[row, column]:
                raise Error("Backend matrices disagree")
    print("  ✓ Matrices match")


def run_ballots(
    n: Int, voters: Int, seed: Int, selector: String, warmup: Int, repeat: Int, requested: ScoreType
) raises:
    var score_type = ballots.resolve_tally_score_type(UInt64(voters), requested)
    print("Score bits: {}".format(score_type.width()))
    if score_type == ScoreType.uint16:
        run_ballots_typed[DType.uint16, Arithmetic.exact](n, voters, seed, selector, warmup, repeat)
    elif score_type == ScoreType.uint32:
        run_ballots_typed[DType.uint32, Arithmetic.exact](n, voters, seed, selector, warmup, repeat)
    elif score_type == ScoreType.uint64:
        run_ballots_typed[DType.uint64, Arithmetic.exact](n, voters, seed, selector, warmup, repeat)
    else:
        run_ballots_typed[DType.uint64, Arithmetic.saturated](n, voters, seed, selector, warmup, repeat)


def run_ballots_typed[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](n: Int, voters: Int, seed: Int, selector: String, warmup: Int, repeat: Int) raises:
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

    var offsets = List[ballots.BallotOffset]()
    comptime if ArithmeticDataType != DType.uint32:
        for ballot in range(voters + 1):
            offsets.append(UInt64(ballot) * UInt64(n))
    var baseline = VoteMatrix[ArithmeticDataType](0)
    var backends: List[Backend] = [Backend.cpu, Backend.gpu]
    for backend in backends:
        if not selected_by(selector, backend.name()):
            continue
        if backend == Backend.gpu and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> VoteMatrix[ArithmeticDataType]:
            comptime if ArithmeticDataType == DType.uint32:
                return rebind_var[VoteMatrix[ArithmeticDataType]](
                    ballots.tally_ballots(ballots.ballot_span(rankings), voters, n, backend=backend)
                )
            else:
                var prepared = ballots.RaggedBallots(
                    ballots.ballot_span(rankings),
                    ballots.ballot_span(offsets),
                    Span[ballots.RankLabel, ImmUntrackedOrigin](),
                    Span[ballots.VoterWeight, ImmUntrackedOrigin](),
                    Span[ballots.PolicyCode, ImmUntrackedOrigin](),
                    n,
                    ballots.Unranked.unknown,
                )
                return ballots.tally_ragged_ballots[ArithmeticDataType, ArithmeticMode](prepared, backend=backend)

        var result = profile[VoteMatrix[ArithmeticDataType]](
            backend.name(), calculate, warmup, repeat, Float64(voters), "ballots/s"
        )
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
    _ = rankings^
    _ = offsets^
    if baseline.num_candidates == 0:
        raise Error("No selected backend could run")


def run_schulze(
    preferences: UInt32VoteMatrix, selector: String, warmup: Int, repeat: Int, score_type: ScoreType
) raises:
    var resolved = schulze.resolve_score_type(preferences.view(), score_type)
    print("Score bits: {}".format(resolved.width()))
    if resolved == ScoreType.uint16:
        run_schulze_typed[DType.uint16](preferences, selector, warmup, repeat)
    elif resolved == ScoreType.uint32:
        run_schulze_typed[DType.uint32](preferences, selector, warmup, repeat)
    else:
        run_schulze_typed[DType.uint64](preferences, selector, warmup, repeat)


def run_schulze_typed[
    ArithmeticDataType: DType
](preferences: UInt32VoteMatrix, selector: String, warmup: Int, repeat: Int) raises:
    var baseline = VoteMatrix[ArithmeticDataType](0)
    var backends: List[Backend] = [Backend.cpu, Backend.gpu]
    var n = preferences.num_candidates
    for backend in backends:
        if not selected_by(selector, backend.name()):
            continue
        if backend == Backend.gpu and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> VoteMatrix[ArithmeticDataType]:
            return schulze.strongest_paths_typed[ArithmeticDataType, ballots.SeedGraph.winning_votes](
                preferences.view(), backend
            )

        var result = profile[VoteMatrix[ArithmeticDataType]](
            backend.name(), calculate, warmup, repeat, Float64(n) ** 3, "cells/s"
        )
        if baseline.num_candidates:
            check_matrix(result, baseline)
        else:
            baseline = result^
    if baseline.num_candidates == 0:
        raise Error("No selected backend could run")
    var outcome = schulze.compute_election_results(baseline.view())
    var top = List[Int]()
    for place in range(min(5, len(outcome.ranking))):
        top.append(outcome.ranking[place])
    print("Winners: {} Top candidates: {}".format(outcome.winners, top))


def run_kemeny(preferences: UInt32VoteMatrix, selector: String, warmup: Int, repeat: Int, score_type: ScoreType) raises:
    print("Score bits: {}".format(kemeny.resolve_score_type(preferences.view(), score_type).width()))
    var baseline = KemenySolution(List[Int](), List[Int](), kemeny.RankingMultiplicity.unique, 0)
    var backends: List[Backend] = [Backend.cpu, Backend.gpu]
    var n = preferences.num_candidates
    for backend in backends:
        if not selected_by(selector, backend.name()):
            continue
        if backend == Backend.gpu and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> KemenySolution:
            var result = kemeny.compute_kemeny_ranking(preferences.view(), backend=backend, score_type=score_type)
            if result.score == UInt64.MAX:
                raise Error("Kemeny optimum reached the saturation sentinel")
            return result^

        var result = profile[KemenySolution](backend.name(), calculate, warmup, repeat, 0, "")
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
        if len(baseline.ranking):
            if (
                result.score != baseline.score
                or result.ranking != baseline.ranking
                or result.winners != baseline.winners
                or result.multiplicity != baseline.multiplicity
            ):
                raise Error("Backend Kemeny results disagree")
            print("  ✓ Rankings and scores match")
        else:
            baseline = result^
    if not len(baseline.ranking):
        raise Error("No selected backend could run")
    print("Score: {} Ranking: {}".format(baseline.score, baseline.ranking))
    print("Winners: {} Multiplicity: {}".format(baseline.winners, baseline.multiplicity.name()))


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
        "--score-bits",
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
                      Backends: CPU, GPU
  --warmup N           Warmup iterations (default: 1)
  --repeat N           Measured iterations (default: 1)
  --seed N             Reproducible input seed (default: 42)
  --score-bits TYPE    Tally/solver arithmetic: auto, uint16, uint32, uint64, or saturated64 (default: auto)
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
    var requested_type = parse_text_arg(args, "--score-bits", "auto")
    var score_type = ScoreType.parse(requested_type)
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
    print("Method: {} Candidates: {} Voters: {} Seed: {}".format(method, n, voters, seed))
    print("Warmup: {} Repeat: {} CPU threads: {}".format(warmup, repeat, num_logical_cores()))
    if selected_by(selector, "GPU") and has_accelerator():
        var ctx = DeviceContext()
        print("GPU: {}".format(ctx.name()))
    if method == "ballots":
        run_ballots(n, voters, seed, selector, warmup, repeat, score_type)
        return
    var preferences = ballots.generate_random_preferences(n, voters, seed)
    var solvers = StringDict[def(UInt32VoteMatrix, String, Int, Int, ScoreType) raises thin -> None]()
    solvers["schulze"] = run_schulze
    solvers["kemeny"] = run_kemeny
    solvers[method](preferences, selector, warmup, repeat, score_type)
