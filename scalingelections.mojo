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

from ballots import PreferenceMatrix, SeedGraph, tally_ballots_cpu, tally_ballots_gpu
from kemeny import kemeny_ranking, kemeny_ranking_gpu
from schulze import TILE_SIZE, compute_strongest_paths_gpu, compute_strongest_paths_tiled_cpu_simd, split_cycle_winners

# region Python Bindings


def available_backends() raises -> PythonObject:
    """Execution targets visible to the Mojo runtime, without running a solver."""
    var names = Python().list()
    names.append("cpu")
    if has_accelerator():
        names.append("gpu")
    return names


def use_gpu(kwargs: StringDict[PythonObject]) raises -> Bool:
    var backend = String(kwargs["backend"]) if "backend" in kwargs else String("cpu")
    for key in kwargs:
        if String(key) != "backend":
            raise Error("Unknown keyword: " + String(key))
    if backend != "cpu" and backend != "gpu":
        raise Error("Unknown backend: " + backend + "; expected cpu or gpu")
    if backend == "gpu" and not has_accelerator():
        raise Error("No GPU is available to the Mojo runtime")
    return backend == "gpu"


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
            matrix[row_index, column_index] = UInt32(integer_from(row[column_index]))
    return matrix^


def strongest_paths(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Widest paths over winning votes, as a list of rows."""
    var gpu = use_gpu(kwargs)
    var matrix = matrix_from(preferences)
    var strengths = compute_strongest_paths_gpu[TILE_SIZE](matrix) if gpu else compute_strongest_paths_tiled_cpu_simd[
        TILE_SIZE
    ](matrix)

    var rows = Python().list()
    for row_index in range(strengths.num_candidates):
        var row = Python().list()
        for column_index in range(strengths.num_candidates):
            row.append(PythonObject(Int(strengths[row_index, column_index])))
        rows.append(row)
    return rows


def kemeny_consensus(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """The exact Kemeny-Young ranking and the disagreement it achieves."""
    var gpu = use_gpu(kwargs)
    var matrix = matrix_from(preferences)
    var solution = kemeny_ranking_gpu(matrix) if gpu else kemeny_ranking(matrix)

    var ranking = Python().list()
    for place in range(len(solution.ranking)):
        ranking.append(PythonObject(solution.ranking[place]))
    var pair = Python().list()
    pair.append(ranking)
    pair.append(PythonObject(solution.score))
    return pair


def tally_ballots(rankings: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts complete rankings into a pairwise matrix, as a list of rows."""
    var gpu = use_gpu(kwargs)
    var num_ballots = len(rankings)
    if num_ballots < 1:
        raise Error("There must be at least one ballot")
    var num_candidates = len(rankings[0])
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

    var counted = tally_ballots_gpu(flat, num_ballots, num_candidates) if gpu else tally_ballots_cpu(
        flat, num_ballots, num_candidates
    )
    var rows = Python().list()
    for row_index in range(num_candidates):
        var row = Python().list()
        for column_index in range(num_candidates):
            row.append(PythonObject(Int(counted[row_index, column_index])))
        rows.append(row)
    return rows


def split_cycle(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """The Split Cycle winning set, which is every candidate nobody defeats."""
    var gpu = use_gpu(kwargs)
    var matrix = matrix_from(preferences)
    var undefeated = split_cycle_winners[compute_strongest_paths_gpu[TILE_SIZE, SeedGraph.positive_margins]](
        matrix
    ) if gpu else split_cycle_winners(matrix)

    var winners = Python().list()
    for index in range(len(undefeated)):
        winners.append(PythonObject(undefeated[index]))
    return winners


@export
def PyInit_scalingelections_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("scalingelections_mojo")
        builder.def_function[available_backends]("available_backends")
        builder.def_function[strongest_paths]("strongest_paths")
        builder.def_function[kemeny_consensus]("kemeny_consensus")
        builder.def_function[tally_ballots]("tally_ballots")
        builder.def_function[split_cycle]("split_cycle")
        return builder.finalize()
    except error:
        abort(String("Failed to initialize scalingelections_mojo: ", error))


# endregion Python Bindings
