"""
Native command line for the ScalingElections backends, benchmarking each and reporting the vote.

Every backend computes the same strongest-paths closure, so the first one to run becomes the
baseline the rest are checked against, and a mismatch is reported rather than averaged away.
Timings are reported as cells per second over `n^3` cells, which is the figure comparable across
candidate counts.

`main` lives here rather than beside the kernels because Mojo refuses to emit a shared library
from a module that defines it.

## Usage

Run directly with Mojo via Pixi:

```bash
pixi run mojo cli.mojo
pixi run mojo cli.mojo --num-candidates 4096 --num-voters 4096
```

For proper benchmarking with large random-generated preference matrices:

```bash
pixi run mojo cli.mojo --num-candidates 2048 --num-voters 0 -k GPU --warmup 1 --repeat 20
pixi run mojo cli.mojo --num-candidates 4096 --num-voters 0 -k GPU --warmup 1 --repeat 10
pixi run mojo cli.mojo --num-candidates 8192 --num-voters 0 -k GPU --warmup 1 --repeat 5
pixi run mojo cli.mojo --num-candidates 16384 --num-voters 0 -k GPU --warmup 1 --repeat 3
pixi run mojo cli.mojo --num-candidates 32768 --num-voters 0 -k GPU --warmup 1 --repeat 1
```

Or compile and run:

```bash
pixi run mojo build cli.mojo -o build/scalingelections
./build/scalingelections
```

See: https://ashvardanian.com/posts/scaling-elections
"""

from std.sys import argv, exit, has_accelerator
from std.time import perf_counter_ns

from ballots import (
    PreferenceMatrix,
    SeedGraph,
    StrongestPathsMatrix,
    generate_random_preferences,
)
from schulze import (
    TILE_SIZE,
    ElectionOutcome,
    compute_election_results,
    compute_strongest_paths_gpu,
    compute_strongest_paths_serial,
    compute_strongest_paths_tiled_cpu,
    compute_strongest_paths_tiled_cpu_simd,
)


# region Benchmarking


@fieldwise_init
struct BaselineState(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """Whether a run has a result to check the next backend against."""

    var value: UInt8

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    comptime missing = Self(0)
    """No backend has succeeded yet, so this one's result becomes the baseline."""
    comptime recorded = Self(1)
    """A baseline is already held, so this one's result is checked against it."""


def run_warmup[
    implementation: def(PreferenceMatrix) raises thin -> StrongestPathsMatrix
](preferences: PreferenceMatrix, warmup: Int) raises:
    """Run warmup iterations and print timing."""
    for iteration in range(warmup):
        var start_time = perf_counter_ns()
        _ = implementation(preferences)
        var elapsed_ns = perf_counter_ns() - start_time
        var iteration_note = " {}/{}".format(iteration + 1, warmup) if warmup > 1 else String("")
        print("  Warm-up{}: {}".format(iteration_note, format_time(elapsed_ns)))


def measure_and_average[
    implementation: def(PreferenceMatrix) raises thin -> StrongestPathsMatrix
](preferences: PreferenceMatrix, repeat: Int, mut result: StrongestPathsMatrix) raises -> Int:
    """Run benchmark iterations and return avg_time_ns, storing result in mutable parameter."""
    var total_time: Int = 0
    for _ in range(repeat):
        var start_time = perf_counter_ns()
        result = implementation(preferences)
        total_time += perf_counter_ns() - start_time

    return total_time // repeat


def profile_and_report[
    implementation: def(PreferenceMatrix) raises thin -> StrongestPathsMatrix
](
    label: String,
    preferences: PreferenceMatrix,
    warmup: Int,
    repeat: Int,
    num_candidates: Int,
    mut baseline: StrongestPathsMatrix,
    baseline_state: BaselineState,
) raises -> ElectionOutcome:
    """Times one implementation, adopting its result as the baseline when none is held yet.

    Returns the winner and the full ranking.
    """
    print("→ {}".format(label))

    run_warmup[implementation](preferences, warmup)

    var result = StrongestPathsMatrix(0)
    var avg_time = measure_and_average[implementation](preferences, repeat, result)

    var average_note = " (avg of {})".format(repeat) if repeat > 1 else String("")
    print(
        "  Run:     {}{} │ {}".format(
            format_time(avg_time),
            average_note,
            format_throughput_from_ns(avg_time, num_candidates),
        )
    )

    if baseline_state == BaselineState.recorded:
        if validate_against_baseline(result, baseline):
            print("  ✓ Results validated")
        else:
            print("  ✗ Results don't match baseline!")

    var outcome = compute_election_results(result)

    if baseline_state == BaselineState.missing:
        baseline = result^

    print()
    return outcome^


def format_time(elapsed_ns: Int) -> String:
    """Formats a duration in milliseconds, switching to seconds past one thousand."""
    var elapsed_milliseconds = elapsed_ns // 1_000_000
    if elapsed_milliseconds < 1000:
        return "{} ms".format(elapsed_milliseconds)
    else:
        var elapsed_seconds = Float64(elapsed_ns) / 1_000_000_000.0
        var hundredths_of_second = Int(elapsed_seconds * 100.0)
        return "{}.{}{} s".format(
            hundredths_of_second // 100, (hundredths_of_second % 100) // 10, hundredths_of_second % 10
        )


def format_throughput(cells_per_sec: Float64) -> String:
    """Formats a cell rate, scaling the unit from kilo up to tera."""
    if cells_per_sec >= 1e12:
        var tenths_of_tera = Int(cells_per_sec / 1e11)
        return "{}.{} Tcells/s".format(tenths_of_tera // 10, tenths_of_tera % 10)
    elif cells_per_sec >= 1e9:
        var tenths_of_giga = Int(cells_per_sec / 1e8)
        return "{}.{} Gcells/s".format(tenths_of_giga // 10, tenths_of_giga % 10)
    elif cells_per_sec >= 1e6:
        var tenths_of_mega = Int(cells_per_sec / 1e5)
        return "{}.{} Mcells/s".format(tenths_of_mega // 10, tenths_of_mega % 10)
    else:
        var tenths_of_kilo = Int(cells_per_sec / 1e2)
        return "{}.{} Kcells/s".format(tenths_of_kilo // 10, tenths_of_kilo % 10)


def format_throughput_from_ns(elapsed_ns: Int, num_candidates: Int) -> String:
    """Formats the cell rate one timing implies, over N cubed cells."""
    if elapsed_ns <= 0:
        return "N/A"
    var elapsed_seconds = Float64(elapsed_ns) / 1_000_000_000.0
    if elapsed_seconds <= 0.0:
        return "N/A"
    var candidates_as_float = Float64(num_candidates)
    var total_cells = candidates_as_float * candidates_as_float * candidates_as_float
    var cells_per_sec = total_cells / elapsed_seconds
    return format_throughput(cells_per_sec)


# endregion Benchmarking


# region Command Line


def validate_against_baseline(result: StrongestPathsMatrix, baseline: StrongestPathsMatrix) -> Bool:
    """Check if two results match."""
    var num_candidates = result.num_candidates
    for row in range(num_candidates):
        for column in range(num_candidates):
            if result[row, column] != baseline[row, column]:
                return False
    return True


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
    return pattern == "." or pattern.lower() in name.lower()


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


comptime USAGE = """Usage: mojo cli.mojo [OPTIONS]

Options:
  --num-candidates N    Number of candidates (default: 128)
  --num-voters N        Number of voters (default: 2000)
                        Set to 0 for instant random preference matrix generation
  -k, --filter TEXT     Select backends whose name contains TEXT, case-insensitively
                        Names: Serial (Mojo), Tiled CPU (Mojo), Tiled CPU+SIMD (Mojo), Tiled GPU (Mojo)
  --warmup N            Number of warmup iterations (default: 1)
  --repeat N            Number of benchmark iterations (default: 1)
  --seed N              Seed for the preference generator (default: 42)
  --help, -h            Show this help message

Examples:
  pixi run mojo cli.mojo --num-candidates 256 --num-voters 4000
  pixi run mojo cli.mojo --num-candidates 4096 -k GPU
  pixi run mojo cli.mojo --num-candidates 16384 --num-voters 0 -k 'Tiled CPU'"""

comptime CONFIGURATION = """Configuration:
  Problem size: {} candidates × {}
  Warmup: {}, Repeat: {}
"""

comptime ELECTION_RESULTS = """Election Results

  Winner: Candidate #{}
  Top {}:  {}
"""

comptime NO_ELECTION_RESULTS = """Election Results

  No implementation was run, so there is no ranking to report.
"""


def main():
    """Benchmarks the selected backends and reports the election."""
    var args = argv()

    if has_flag(args, "--help") or has_flag(args, "-h"):
        print(USAGE)
        return
    var num_candidates = 128
    var num_voters = 2000
    var warmup = 1
    var repeat = 1
    var seed_value = 42
    var selector = String(".")
    try:
        reject_unknown_flags(args)
        num_candidates = parse_int_arg(args, "--num-candidates", num_candidates)
        num_voters = parse_int_arg(args, "--num-voters", num_voters)

        warmup = parse_int_arg(args, "--warmup", warmup)
        repeat = parse_int_arg(args, "--repeat", repeat)
        seed_value = parse_int_arg(args, "--seed", seed_value)
        selector = parse_text_arg(args, "--filter", selector)
        selector = parse_text_arg(args, "-k", selector)
        if num_candidates < 4:
            raise Error("--num-candidates must be at least 4")
        if num_voters < 0:
            raise Error("--num-voters cannot be negative")
        if warmup < 0:
            raise Error("--warmup cannot be negative")
        if repeat < 1:
            raise Error("--repeat must be at least 1")
    except error:
        print("Error: {}\n".format(error))
        print(USAGE)
        exit(2)

    print("Schulze Voting Algorithm (Mojo)\n")

    comptime serial_label = "Serial (Mojo)"
    comptime cpu_label = "Tiled CPU (Mojo)"
    comptime simd_label = "Tiled CPU+SIMD (Mojo)"
    comptime gpu_label = "Tiled GPU (Mojo)"
    var wants_gpu = selected_by(selector, gpu_label)
    if wants_gpu and not has_accelerator():
        print("✗ No GPU detected, so {} is skipped\n".format(gpu_label))
        wants_gpu = False

    var voters_description = "{} voters".format(num_voters) if num_voters > 0 else String("random")
    print(CONFIGURATION.format(num_candidates, voters_description, warmup, repeat))

    print("Generating preferences...")
    var preferences = generate_random_preferences(num_candidates, num_voters, seed_value)

    print("\nBenchmarking\n")

    # The first backend that succeeds becomes the baseline the rest validate against.
    var baseline = StrongestPathsMatrix(0)
    var baseline_state = BaselineState.missing
    var winner = 0
    var ranking = List[Int]()

    if selected_by(selector, serial_label):
        try:
            var outcome = profile_and_report[compute_strongest_paths_serial[SeedGraph.winning_votes]](
                serial_label,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                baseline_state,
            )
            if baseline_state == BaselineState.missing:
                winner = outcome.winner
                ranking = outcome.ranking.copy()
                baseline_state = BaselineState.recorded
        except error:
            print("  ✗ {} failed: {}\n".format(serial_label, error))

    if selected_by(selector, cpu_label):
        try:
            var outcome = profile_and_report[compute_strongest_paths_tiled_cpu[TILE_SIZE]](
                cpu_label,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                baseline_state,
            )
            if baseline_state == BaselineState.missing:
                winner = outcome.winner
                ranking = outcome.ranking.copy()
                baseline_state = BaselineState.recorded
        except error:
            print("  ✗ {} failed: {}\n".format(cpu_label, error))

    if selected_by(selector, simd_label):
        try:
            var outcome = profile_and_report[compute_strongest_paths_tiled_cpu_simd[TILE_SIZE]](
                simd_label,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                baseline_state,
            )
            if baseline_state == BaselineState.missing:
                winner = outcome.winner
                ranking = outcome.ranking.copy()
                baseline_state = BaselineState.recorded
        except error:
            print("  ✗ {} failed: {}\n".format(simd_label, error))

    if wants_gpu:
        try:
            var outcome = profile_and_report[compute_strongest_paths_gpu[TILE_SIZE]](
                gpu_label,
                preferences,
                warmup,
                repeat,
                num_candidates,
                baseline,
                baseline_state,
            )
            if baseline_state == BaselineState.missing:
                winner = outcome.winner
                ranking = outcome.ranking.copy()
                baseline_state = BaselineState.recorded
        except error:
            print("  ✗ {} failed: {}\n".format(gpu_label, error))

    if baseline_state == BaselineState.missing:
        print(NO_ELECTION_RESULTS)
        return

    var top_count = min(5, len(ranking))
    var top_shown = String("")
    for position in range(top_count):
        if position > 0:
            top_shown += ", "
        top_shown += "#{}".format(ranking[position])
    print(ELECTION_RESULTS.format(winner, top_count, top_shown))


# endregion Command Line
