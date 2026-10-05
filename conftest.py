"""Environment knobs, the session banner, and the probes the suite skips on.

No test function and no oracle lives here: `test.py` holds both, and reads the seed through
`derived_seed` so a failing run reproduces from the header it printed.
"""

import functools
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

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
    try:
        extension.compute_strongest_paths(np.zeros((2, 2), dtype=np.uint32), backend="gpu_serial")
    except RuntimeError:
        return False
    return True


def pytest_report_header() -> list[str]:
    """What this run exercises, printed where pytest prints its own header."""
    return [
        f"seed: {_RUN_SEED}, pin with SCALINGELECTIONS_TESTS_SEED",
        f"repetitions: {randomized_repetitions_count}, exhaustive scale: {exhaustive_scale}",
        f"cuda: {installed_module_path('scalingelections_cuda') or 'extension not built'}",
        f"device: {'gpu_serial answers' if cuda_device_ready() else 'none visible, device cases skip'}",
        f"mojo: {installed_module_path('scalingelections_mojo') or 'not built, run `pixi run build-bindings`'}",
        f"oracles: pref_voting {_oracle_state('pref_voting')}, igraph {_oracle_state('igraph')}",
    ]


def _oracle_state(name: str) -> str:
    """Whether a third-party oracle is importable, which decides if its cases run or skip."""
    return "ready" if installed_module_path(name) else "missing"


# endregion Probes


# region Fixtures


@pytest.fixture
def seed(__pytest_repeat_step_number) -> int:
    """A per-test seed that moves with the repeat step, so `--count` explores instead of replaying.

    The parameter carries no default on purpose: pytest builds a fixture's closure from the
    parameters that have none, so a defaulted one is never injected and every repeat replays the
    first step's draws. `pytest-repeat` hands `None` to a test that is not repeated. The stride is
    the repetition count, so a repeat step never lands on a seed a parametrized axis already used.
    """
    return derived_seed((__pytest_repeat_step_number or 0) * randomized_repetitions_count)


@pytest.fixture(scope="session")
def gpu_ready() -> bool:
    """Whether a device backend can actually run, as opposed to a CPU-only build or an empty box."""
    return cuda_device_ready()


@pytest.fixture(scope="session")
def mojo():
    """The Mojo extension from `build/`, skipping the case when it has not been compiled."""
    return pytest.importorskip("scalingelections_mojo", reason="Build it with `pixi run build-bindings`")


@pytest.fixture(scope="session")
def pref_voting():
    """Eric Pacuit's reference library, whose authors defined Split Cycle."""
    return pytest.importorskip("pref_voting", reason="Install it with `uv sync --extra cpu`")


@pytest.fixture(scope="session")
def igraph():
    """The graph library whose exact integer program stands in for Kemeny above ten candidates."""
    return pytest.importorskip("igraph", reason="Install it with `uv sync --extra cpu`")


# endregion Fixtures
