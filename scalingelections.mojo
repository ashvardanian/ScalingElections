"""Python bindings for the Mojo implementations, so one test suite can cross-check every backend.

This module holds nothing but argument marshalling and the module table. `main` lives in
`cli.mojo`, because Mojo refuses to emit a shared library from a module that defines one.

Matrices cross the boundary element by element rather than as a buffer, which is fine for the
cross-language checks this exists to serve and wrong for benchmarking. Time the native binary
`cli.mojo` builds instead.
"""

from std.collections import StringDict
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import has_accelerator

import ballots
import kemeny
import schulze
from ballots import Backend, PreferenceMatrix, ScoreType

# region Python Bindings


def available_backends() raises -> PythonObject:
    """Execution targets visible to the Mojo runtime, without running a solver."""
    var names = Python().list()
    names.append("cpu")
    if has_accelerator():
        names.append("gpu")
    return names


def backend_from(kwargs: StringDict[PythonObject]) raises -> Backend:
    var backend = String(kwargs["backend"]) if "backend" in kwargs else String("cpu")
    for key in kwargs:
        if String(key) != "backend":
            raise Error("Unknown keyword: " + String(key))
    if backend != "cpu" and backend != "gpu":
        raise Error("Unknown backend: " + backend + "; expected cpu or gpu")
    if backend == "gpu" and not has_accelerator():
        raise Error("No GPU is available to the Mojo runtime")
    return Backend.gpu if backend == "gpu" else Backend.cpu


def score_type_from(mut kwargs: StringDict[PythonObject]) raises -> ScoreType:
    var requested_type = String(kwargs.pop("score_type")) if "score_type" in kwargs else String("auto")
    var score_type = ScoreType.auto
    if requested_type == "uint16":
        score_type = ScoreType.uint16
    elif requested_type == "uint32":
        score_type = ScoreType.uint32
    elif requested_type == "uint64":
        score_type = ScoreType.uint64
    elif requested_type != "auto":
        raise Error("score_type must be auto, uint16, uint32, or uint64")
    return score_type


def integer_from(value: PythonObject) raises -> Int:
    """Reads a Python integer through its decimal spelling, the one conversion the binding offers."""
    return Int(String(value))


def matrix_from(preferences: PythonObject) raises -> PreferenceMatrix:
    """Reads a square two-dimensional integer array into an owned matrix."""
    var num_candidates = len(preferences)
    if num_candidates < 1:
        raise Error("Preferences must have at least one candidate")

    var matrix = PreferenceMatrix(num_candidates)
    for row_index in range(num_candidates):
        var row = preferences[row_index]
        if len(row) != num_candidates:
            raise Error("Preferences must be a square matrix")
        for column_index in range(num_candidates):
            var value = integer_from(row[column_index])
            if value < 0 or UInt64(value) > UInt64(UInt32.MAX):
                raise Error("Entries must fit UInt32")
            matrix[row_index, column_index] = UInt32(value)
    return matrix^


def compute_strongest_paths(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Widest paths over winning votes, as a UInt32 NumPy matrix."""
    var score_type = score_type_from(kwargs)
    var backend = backend_from(kwargs)
    var matrix = matrix_from(preferences)
    var strengths = schulze.compute_strongest_paths(matrix, backend=backend, score_type=score_type)

    var rows = Python().list()
    for row_index in range(strengths.num_candidates):
        var row = Python().list()
        for column_index in range(strengths.num_candidates):
            row.append(PythonObject(Int(strengths[row_index, column_index])))
        rows.append(row)
    return Python.import_module("numpy").array(rows, dtype="uint32")


def compute_kemeny_ranking(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """The exact Kemeny-Young ranking and disagreement, using a compiled score type specialization."""
    var score_type = score_type_from(kwargs)
    var backend = backend_from(kwargs)
    var matrix = matrix_from(preferences)
    var solution = kemeny.compute_kemeny_ranking(matrix, backend=backend, score_type=score_type)
    var ranking = Python().list()
    for place in range(len(solution.ranking)):
        ranking.append(PythonObject(solution.ranking[place]))
    return Python().tuple(ranking, PythonObject(Int(solution.score)))


def tally_ballots(rankings: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts complete rankings into a UInt32 NumPy matrix."""
    var backend = backend_from(kwargs)
    var num_ballots = len(rankings)
    var array = Python.import_module("numpy").asarray(rankings)
    if integer_from(array.ndim) != 2:
        raise Error("Rankings must be a two-dimensional array")
    var num_candidates = integer_from(array.shape[1])
    if num_candidates < 1:
        raise Error("Every ballot must rank at least one candidate")

    var flat = List[UInt32]()
    flat.reserve(num_ballots * num_candidates)
    for ballot in range(num_ballots):
        var row = rankings[ballot]
        if len(row) != num_candidates:
            raise Error("Every ballot must rank the same candidates")
        for position in range(num_candidates):
            flat.append(UInt32(integer_from(row[position])))

    var counted = ballots.tally_ballots(flat, num_ballots, num_candidates, backend=backend)
    var rows = Python().list()
    for row_index in range(num_candidates):
        var row = Python().list()
        for column_index in range(num_candidates):
            row.append(PythonObject(Int(counted[row_index, column_index])))
        rows.append(row)
    return Python.import_module("numpy").array(rows, dtype="uint32")


def compute_split_cycle_winners(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """The Split Cycle winning set, which is every candidate nobody defeats."""
    var score_type = score_type_from(kwargs)
    var backend = backend_from(kwargs)
    var matrix = matrix_from(preferences)
    var undefeated = schulze.compute_split_cycle_winners(matrix, backend=backend, score_type=score_type)

    var winners = Python().list()
    for index in range(len(undefeated)):
        winners.append(PythonObject(undefeated[index]))
    return winners


@export
def PyInit_scalingelections_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("scalingelections_mojo")
        builder.def_function[available_backends]("available_backends")
        builder.def_function[tally_ballots]("tally_ballots")
        builder.def_function[compute_strongest_paths]("compute_strongest_paths")
        builder.def_function[compute_kemeny_ranking]("compute_kemeny_ranking")
        builder.def_function[compute_split_cycle_winners]("compute_split_cycle_winners")
        return builder.finalize()
    except error:
        abort(String("Failed to initialize scalingelections_mojo: ", error))


# endregion Python Bindings
