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
from ballots import Arithmetic, Backend, ScoreType, VoteMatrix, with_score_type
from kemeny import KemenySolution

comptime BENCHMARK_COUNT_TYPE = DType.uint32
"""The benchmark's preference element, wide enough for every synthetic electorate it draws."""


def profile[
    ResultType: Movable & Deinitable, Function: def() raises -> ResultType
](label: String, function: Function, warmup: Int, repeat: Int, work: Float64, unit: String) raises -> ResultType:
    """Times complete calls and retains the last result for validation."""
    print(t"→ {label}")
    for _ in range(warmup):
        _ = function()
    var start = perf_counter_ns()
    var result = function()
    var elapsed = perf_counter_ns() - start
    print(t"  sample_ns {elapsed}")
    var total = elapsed
    for _ in range(1, repeat):
        start = perf_counter_ns()
        result = function()
        elapsed = perf_counter_ns() - start
        print(t"  sample_ns {elapsed}")
        total += elapsed
    var average = total // repeat
    print(t"  mean_ns {average} │ {format_time(average)}")
    if work > 0 and average > 0:
        print(t"  rate {work * 1e9 / Float64(average)} {unit}")
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
    var score_type = ballots.resolve_tally_score_type(ballots.VoterWeight(voters), requested)
    print(t"Score bits: {score_type.bits()}")

    def run_with[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> None:
        run_ballots_typed[ArithmeticDataType, ArithmeticMode](n, voters, seed, selector, warmup, repeat)

    with_score_type(score_type, run_with)


def run_ballots_typed[
    ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](n: Int, voters: Int, seed: Int, selector: String, warmup: Int, repeat: Int) raises:
    var rankings = List[ballots.CandidateIndex](length=n * voters, fill=0)
    var generator = Random(seed=UInt64(seed))
    for ballot in range(voters):
        var base = ballot * n
        for candidate in range(n):
            rankings[base + candidate] = ballots.CandidateIndex(candidate)
        for upper in range(n - 1, 0, -1):
            var chosen = Int(generator.step()[0] % UInt32(upper + 1))
            var held = rankings[base + upper]
            rankings[base + upper] = rankings[base + chosen]
            rankings[base + chosen] = held

    var offsets = List[ballots.BallotOffset]()
    for ballot in range(voters + 1):
        offsets.append(ballots.BallotOffset(ballot) * ballots.BallotOffset(n))
    var baseline = VoteMatrix[ArithmeticDataType](0)
    var backends: List[Backend] = [Backend.cpu, Backend.gpu]
    for backend in backends:
        if not selected_by(selector, backend.name()):
            continue
        if backend == Backend.gpu and not has_accelerator():
            print("GPU unavailable")
            continue

        def calculate() raises {imm} -> VoteMatrix[ArithmeticDataType]:
            var result = VoteMatrix[ArithmeticDataType](n)
            if backend == Backend.cpu or ballots.tally_dense_serves_gpu[ArithmeticDataType, ArithmeticMode](
                DeviceContext(), n
            ):
                ballots.tally_ballots[ArithmeticDataType, ArithmeticMode](
                    ballots.ballot_span(rankings), voters, n, result.data, backend=backend
                )
                return result^
            ballots.tally_ragged_relations[ArithmeticDataType, ArithmeticMode, ballots.PairwiseRelation.preference](
                ballots.RaggedBallots(
                    ballots.ballot_span(rankings),
                    ballots.ballot_span(offsets),
                    Span[ballots.RankLabel, ImmUntrackedOrigin](),
                    Span[ballots.VoterWeight, ImmUntrackedOrigin](),
                    Span[ballots.PolicyCode, ImmUntrackedOrigin](),
                    n,
                    ballots.Unranked.unknown,
                ),
                result.data,
                backend=backend,
            )
            return result^

        var result = profile[VoteMatrix[ArithmeticDataType]](
            backend.name(), calculate, warmup, repeat, Float64(voters), "ballots/s"
        )
        for row in range(n):
            if result[row, row] != 0:
                raise Error("Nonzero tally diagonal")
            for column in range(row + 1, n):
                var total = ballots.VoterWeight(result[row, column]) + ballots.VoterWeight(result[column, row])
                if total != ballots.VoterWeight(voters):
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
    preferences: VoteMatrix[BENCHMARK_COUNT_TYPE], selector: String, warmup: Int, repeat: Int, score_type: ScoreType
) raises:
    var resolved = schulze.resolve_score_type[schulze.SeedGraph.winning_votes](preferences.view(), score_type)
    print(t"Score bits: {resolved.bits()}")

    def run_with[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> None:
        run_schulze_typed[ArithmeticDataType](preferences, selector, warmup, repeat)

    with_score_type(resolved, run_with)


def run_schulze_typed[
    ArithmeticDataType: DType
](preferences: VoteMatrix[BENCHMARK_COUNT_TYPE], selector: String, warmup: Int, repeat: Int) raises:
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
            var paths = VoteMatrix[ArithmeticDataType](n)
            schulze.strongest_paths_typed[ArithmeticDataType, schulze.SeedGraph.winning_votes](
                preferences.view(), backend, paths.data
            )
            return paths^

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
    print(t"Winners: {outcome.winners} Top candidates: {top}")


def run_kemeny(
    preferences: VoteMatrix[BENCHMARK_COUNT_TYPE], selector: String, warmup: Int, repeat: Int, score_type: ScoreType
) raises:
    print(t"Score bits: {kemeny.resolve_score_type(preferences.view(), score_type).bits()}")
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
            return kemeny.compute_kemeny_ranking(preferences.view(), backend=backend, score_type=score_type)

        var result = profile[KemenySolution](backend.name(), calculate, warmup, repeat, 0, "")
        var score = ballots.VoterWeight(0)
        var seen = kemeny.SubsetMask(0)
        for earlier in range(n):
            var candidate = result.ranking[earlier]
            var bit = kemeny.SubsetMask(1) << kemeny.SubsetMask(candidate)
            if candidate < 0 or candidate >= n or (seen & bit) != 0:
                raise Error("Kemeny ranking is not a permutation")
            seen |= bit
            for later in range(earlier + 1, n):
                score += ballots.VoterWeight(preferences[result.ranking[later], candidate])
        if score != ballots.VoterWeight(result.score):
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
    print(t"Score: {baseline.score} Ranking: {baseline.ranking}")
    print(t"Winners: {baseline.winners} Multiplicity: {baseline.multiplicity.name()}")


def format_time(elapsed_ns: Int) -> String:
    """Formats a duration in milliseconds, switching to seconds past one thousand."""
    var elapsed_milliseconds = elapsed_ns // 1_000_000
    if elapsed_milliseconds < 1000:
        return String(t"{Float64(elapsed_ns) / 1_000_000.0} ms")
    var hundredths_of_second = Int(Float64(elapsed_ns) / 1_000_000_000.0 * 100.0)
    return String(t"{hundredths_of_second // 100}.{(hundredths_of_second % 100) // 10}{hundredths_of_second % 10} s")


def parse_int_arg(args: Span[StaticString, ImmStaticOrigin], flag: String, default: Int) raises -> Int:
    """Parse an integer command-line argument, raising if it is malformed."""
    for index in range(len(args)):
        if String(args[index]) != flag:
            continue
        if index + 1 >= len(args):
            raise Error(t"{flag} needs a value")
        try:
            return Int(String(args[index + 1]))
        except:
            raise Error(t"Invalid value for {flag}: {args[index + 1]}")
    return default


def parse_text_arg(args: Span[StaticString, ImmStaticOrigin], flag: String, default: String) raises -> String:
    """Parse a string argument, raising when its value is missing."""
    for index in range(len(args)):
        if String(args[index]) != flag:
            continue
        if index + 1 >= len(args):
            raise Error(t"{flag} needs a value")
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
        "--score-type",
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
            raise Error(t"Unknown option: {argument}")
        index += 1


def print_usage():
    var electorate_millions = ballots.NATIONAL_ELECTORATE // 1_000_000
    print(
        t"""Usage: scalingelections [OPTIONS]

  --method NAME        ballots, schulze (default), or kemeny
  --num-candidates N   Number of candidates (default: 128)
  --num-voters N       Number of voters (default: 2000); 0 draws matrix counts in [0, {electorate_millions}M]
  -k, --filter TEXT    Comma-separated backend substrings, case-insensitive (default: all)
                      Backends: CPU, GPU
  --warmup N           Warmup iterations (default: 1)
  --repeat N           Measured iterations (default: 1)
  --seed N             Reproducible input seed (default: 42)
  --score-type TYPE    Tally/solver arithmetic: auto, saturated64, uint64, uint32, or uint16 (default: auto)
  --help, -h           Show help
"""
    )


def main() raises:
    var args = argv()
    if has_flag(args, "--help") or has_flag(args, "-h"):
        print_usage()
        return
    reject_unknown_flags(args)
    var method = parse_text_arg(args, "--method", "schulze")
    var n = parse_int_arg(args, "--num-candidates", 128)
    var voters = parse_int_arg(args, "--num-voters", 2000)
    var warmup = parse_int_arg(args, "--warmup", 1)
    var repeat = parse_int_arg(args, "--repeat", 1)
    var seed = parse_int_arg(args, "--seed", 42)
    var requested_type = parse_text_arg(args, "--score-type", "auto")
    var score_type = ScoreType.parse(requested_type)
    var selector = parse_text_arg(args, "--filter", ".")
    selector = parse_text_arg(args, "-k", selector)
    if n < 1 or voters < 0 or warmup < 0 or repeat < 1:
        raise Error("Candidates and repeat must be positive; voters and warmup cannot be negative")
    if method != "ballots" and method != "schulze" and method != "kemeny":
        raise Error("--method must be ballots, schulze, or kemeny")
    if method == "ballots" and voters == 0:
        raise Error("--num-voters must be positive for ballot benchmarks")
    if method == "kemeny":
        kemeny.require_kemeny_width(n)
    print(t"Method: {method} Candidates: {n} Voters: {voters} Seed: {seed}")
    print(t"Warmup: {warmup} Repeat: {repeat} CPU threads: {num_logical_cores()}")
    if selected_by(selector, "GPU") and has_accelerator():
        var ctx = DeviceContext()
        print(t"GPU: {ctx.name()}")
    if method == "ballots":
        run_ballots(n, voters, seed, selector, warmup, repeat, score_type)
        return
    var preferences = ballots.generate_random_preferences[BENCHMARK_COUNT_TYPE](n, voters, seed)
    var solvers = StringDict[def(VoteMatrix[BENCHMARK_COUNT_TYPE], String, Int, Int, ScoreType) raises thin -> None]()
    solvers["schulze"] = run_schulze
    solvers["kemeny"] = run_kemeny
    solvers[method](preferences, selector, warmup, repeat, score_type)
