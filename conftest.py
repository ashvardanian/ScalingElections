"""Environment knobs, the session banner, and the probes the suite skips on.

No test function and no oracle lives here: `test.py` holds both, and reads the seed through
`derived_seed` so a failing run reproduces from the header it printed.
"""

import functools
import importlib
import os
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
from numpy.typing import NDArray
import pytest

from kemeny import KemenyResult
from ballots import PairwiseCounts

sys.path.insert(0, str(Path(__file__).resolve().parent / "build"))

_RUN_SEED = int(os.environ.get("SCALINGELECTIONS_TESTS_SEED", int.from_bytes(os.urandom(4), "little")))
"""Base seed every profile in the suite offsets. Pin it with `SCALINGELECTIONS_TESTS_SEED`."""

randomized_repetitions_count: int = int(os.environ.get("SCALINGELECTIONS_TESTS_REPETITIONS", "12"))
"""How many profiles each randomized test draws. Override with `SCALINGELECTIONS_TESTS_REPETITIONS`."""

exhaustive_scale: int = max(1, int(os.environ.get("SCALINGELECTIONS_TESTS_SCALE", "1")))
"""How far the exhaustive Kemeny ladders climb, which costs `2^n` a rung.

Breadth and depth are separate knobs on purpose: raising the repetitions fuzzes wider, raising
the scale searches deeper, and one number cannot express both.
"""


def derived_seed(offset: int) -> int:
    """This run's base seed offset by a per-case number, so every draw is named and reproducible."""
    return _RUN_SEED + offset


# region Probes


@functools.cache
def installed_module_path(name: str) -> str | None:
    """Where an optional module was imported from, or None when this environment has none."""
    try:
        return importlib.import_module(name).__file__
    except ImportError:
        return None


@functools.cache
def cuda_device_ready() -> bool:
    """Whether a device backend can actually run, as opposed to a CPU-only build or an empty box."""
    try:
        import scalingelections_cuda as extension
    except ImportError:
        return False
    return "gpu" in extension.available_backends()


def pytest_report_header() -> list[str]:
    """What this run exercises, printed where pytest prints its own header."""
    return [
        f"seed: {_RUN_SEED}, pin with SCALINGELECTIONS_TESTS_SEED",
        f"repetitions: {randomized_repetitions_count}, exhaustive scale: {exhaustive_scale}",
        f"cuda: {installed_module_path('scalingelections_cuda') or 'extension not built'}",
        f"device: {'CUDA device visible' if cuda_device_ready() else 'none visible, device cases skip'}",
        f"mojo: {installed_module_path('scalingelections_mojo') or 'not built, run `pixi run build-bindings`'}",
        f"oracles: pref_voting {_oracle_state('pref_voting')}, igraph {_oracle_state('igraph')}",
    ]


def _oracle_state(name: str) -> str:
    """Whether a third-party oracle is importable, which decides if its cases run or skip."""
    return "ready" if installed_module_path(name) else "missing"


# endregion Probes


# region Fixtures


@pytest.fixture
def seed(__pytest_repeat_step_number: int | None) -> int:
    """A per-test seed that moves with the repeat step, so `--count` explores instead of replaying.

    The parameter carries no default on purpose: pytest builds a fixture's closure from the
    parameters that have none, so a defaulted one is never injected and every repeat replays the
    first step's draws. `pytest-repeat` hands `None` to a test that is not repeated. The stride is
    the repetition count, so a repeat step never lands on a seed a parametrized axis already used.
    """
    return derived_seed((__pytest_repeat_step_number or 0) * randomized_repetitions_count)


@dataclass(frozen=True)
class Implementation:
    """One implementation and execution target exposing the common operations."""

    tally_ballots: Callable[..., NDArray[np.uint16 | np.uint32 | np.uint64]]
    tally_pairwise_relations: Callable[..., PairwiseCounts]
    compute_strongest_paths: Callable[..., NDArray[np.uint16 | np.uint32 | np.uint64]]
    compute_kemeny_ranking: Callable[..., KemenyResult]
    enumerate_kemeny_rankings: Callable[..., Generator[list[int], None, None]]
    compute_split_cycle_winners: Callable[..., list[int]]


@pytest.fixture(scope="session", params=("python-cpu", "cpp-cpu", "cpp-gpu", "mojo-cpu", "mojo-gpu"))
def implementation(request: pytest.FixtureRequest) -> Implementation:
    """Bind the shared operation contract to one available execution target."""
    language, target = request.param.split("-")
    import scalingelections as module
    from scalingelections import Backend

    target = Backend(target)

    if language != "python":
        pytest.importorskip("scalingelections_cuda" if language == "cpp" else "scalingelections_mojo")
    if target not in module.available_backends(implementation=language):
        pytest.skip(f"{request.param} is unavailable")
    return Implementation(
        *(
            functools.partial(getattr(module, name), implementation=language, backend=target)
            for name in (
                "tally_ballots",
                "tally_pairwise_relations",
                "compute_strongest_paths",
                "compute_kemeny_ranking",
                "enumerate_kemeny_rankings",
                "compute_split_cycle_winners",
            )
        ),
    )


@pytest.fixture(scope="session")
def pref_voting() -> ModuleType:
    """Eric Pacuit's reference library, whose authors defined Split Cycle."""
    return pytest.importorskip("pref_voting", reason="Install it with `uv sync --extra cpu`")


@pytest.fixture(scope="session")
def igraph() -> ModuleType:
    """The graph library whose exact integer program stands in for Kemeny above ten candidates."""
    return pytest.importorskip("igraph", reason="Install it with `uv sync --extra cpu`")


# endregion Fixtures
