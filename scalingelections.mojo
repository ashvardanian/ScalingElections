"""Python bindings for the Mojo implementations, so one test suite can cross-check every backend.

This module holds nothing but argument marshalling and the module table. `main` lives in
`cli.mojo`, because Mojo refuses to emit a shared library from a module that defines one.

Contiguous integer arrays cross the boundary through typed NumPy views and bulk copies.
Python sequences retain checked integer conversion before entering the solvers.
"""

from std.collections import Span, StringDict
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import ExceptionType, PythonModuleBuilder, raise_python_exception
from std.python.numpy import copy_to_numpy_tensor, from_numpy_tensor
from std.sys import has_accelerator
from std.utils.coord import Coord

import ballots
import kemeny
import schulze
from ballots import Backend, ScoreType, Unranked, VoteMatrix

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
    return ScoreType.parse(requested_type)


def integer_from(value: PythonObject) raises -> Int:
    """Reads a Python object implementing the integer index protocol."""
    return Int(py=value.__index__())


def unsigned_from[arithmetic_dtype: DType](value: PythonObject) raises -> SIMD[arithmetic_dtype, 1]:
    var count = UInt64(py=value.__index__())
    if count > UInt64(SIMD[arithmetic_dtype, 1].MAX):
        raise Error("Integer is outside the selected unsigned range")
    return count.cast[arithmetic_dtype]()


def matrix_from_array[stored_count_dtype: DType](array: PythonObject) raises -> VoteMatrix[DType.uint64]:
    var view = from_numpy_tensor[stored_count_dtype, 2](array)
    var matrix = VoteMatrix[DType.uint64](integer_from(array.shape[0]))
    for cell in range(len(view.data)):
        matrix.data[unsafe_offset=cell] = UInt64(view.data[cell])
    return matrix^


def matrix_from(preferences: PythonObject) raises -> VoteMatrix[DType.uint64]:
    """Reads square integer preferences without narrowing their counts."""
    var np = Python.import_module("numpy")
    var array = np.asarray(preferences) if Bool(
        Python.import_module("builtins").isinstance(preferences, np.ndarray)
    ) else np.asarray(preferences, dtype="object")
    if integer_from(array.ndim) != 2:
        raise Error("Preferences must be a square matrix")
    var num_candidates = integer_from(array.shape[0])
    if num_candidates < 1 or integer_from(array.shape[1]) != num_candidates:
        raise Error("Preferences must be a nonempty square matrix")
    var kind = String(array.dtype.kind)
    if kind == "O":
        var matrix = VoteMatrix[DType.uint64](num_candidates)
        for row in range(num_candidates):
            for column in range(num_candidates):
                matrix[row, column] = unsigned_from[DType.uint64](array[row][column])
        return matrix^
    if kind != "u" and kind != "i":
        raise Error("Entries must be integers")
    if kind == "i" and Bool(np.any(array < 0)):
        raise Error("Entries must be nonnegative")
    if integer_from(array.dtype.itemsize) <= 4:
        return matrix_from_array[DType.uint32](np.ascontiguousarray(array, dtype="uint32"))
    return matrix_from_array[DType.uint64](np.ascontiguousarray(array, dtype="uint64"))


def matrix_to_python[
    stored_count_dtype: DType
](matrix: VoteMatrix[stored_count_dtype], score_type: ScoreType) raises -> PythonObject:
    var n = matrix.num_candidates
    var values = Span(unsafe_ptr=matrix.data, length=n * n)
    var array = copy_to_numpy_tensor(values, Coord(n, n))
    return array.astype("uint" + String(score_type.width()), copy=False)


def compute_strongest_paths(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Widest paths over winning votes, preserving UInt64 counts."""
    var score_type = score_type_from(kwargs)
    var backend = backend_from(kwargs)
    var matrix = matrix_from(preferences)
    try:
        score_type = schulze.resolve_score_type(matrix, score_type)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
    var strengths = schulze.compute_strongest_paths(matrix, backend=backend, score_type=score_type)

    return matrix_to_python(strengths, score_type)


def compute_kemeny_ranking(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """The exact Kemeny-Young ranking and disagreement, using a compiled score type specialization."""
    var score_type = score_type_from(kwargs)
    var backend = backend_from(kwargs)
    var matrix = matrix_from(preferences)
    var solution = kemeny.compute_kemeny_ranking(matrix, backend=backend, score_type=score_type)
    if solution.score == UInt64.MAX:
        return PythonObject(
            from_owned=raise_python_exception(
                Error("Kemeny optimum reached the saturation sentinel"), ExceptionType("PyExc_OverflowError")
            )
        )
    var ranking = Python().list()
    for place in range(len(solution.ranking)):
        ranking.append(PythonObject(solution.ranking[place]))
    var winners = Python().list()
    for candidate in solution.winners:
        winners.append(PythonObject(candidate))
    return Python().tuple(ranking, PythonObject(solution.score), winners, PythonObject(solution.unique))


def tally_ballots(rankings: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts dense or CSR integer-weighted rankings with explicit ties and omission semantics."""
    var offsets_arg = kwargs.pop("offsets") if "offsets" in kwargs else PythonObject(None)
    var count_arg = kwargs.pop("num_candidates") if "num_candidates" in kwargs else PythonObject(None)
    var ranks_arg = kwargs.pop("ranks") if "ranks" in kwargs else PythonObject(None)
    var weights_arg = kwargs.pop("weights") if "weights" in kwargs else PythonObject(None)
    var unranked_name = String(kwargs.pop("unranked")) if "unranked" in kwargs else String("unknown")
    if unranked_name != "unknown" and unranked_name != "worse":
        raise Error("unranked must be unknown or worse")
    var unranked = Unranked.worse if unranked_name == "worse" else Unranked.unknown
    var score_type = score_type_from(kwargs)
    var backend = backend_from(kwargs)
    var flat = List[UInt32]()
    var offsets = List[UInt64]()
    var ranks = List[UInt32]()
    var weights = List[UInt64]()
    var num_candidates: Int
    var num_ballots: Int
    var dense_width = 0
    if offsets_arg is PythonObject(None):
        var array = Python.import_module("numpy").asarray(rankings)
        if integer_from(array.ndim) != 2:
            raise Error("Rankings must be a two-dimensional array or CSR entries")
        num_ballots = integer_from(array.shape[0])
        dense_width = integer_from(array.shape[1])
        num_candidates = dense_width if count_arg is PythonObject(None) else integer_from(count_arg)
        offsets.append(0)
        for ballot in range(num_ballots):
            var row = rankings[ballot]
            if len(row) != dense_width:
                raise Error("Dense rows must have the same width")
            for position in range(dense_width):
                flat.append(unsigned_from[DType.uint32](row[position]))
            offsets.append(UInt64(len(flat)))
        if ranks_arg is not PythonObject(None):
            if len(ranks_arg) != num_ballots:
                raise Error("Ranks must match the rankings shape")
            for ballot in range(num_ballots):
                if len(ranks_arg[ballot]) != dense_width:
                    raise Error("Ranks must match the rankings shape")
                for position in range(dense_width):
                    ranks.append(unsigned_from[DType.uint32](ranks_arg[ballot][position]))
    else:
        if count_arg is PythonObject(None):
            raise Error("CSR ballots require num_candidates")
        num_candidates = integer_from(count_arg)
        num_ballots = len(offsets_arg) - 1
        if num_ballots < 0:
            raise Error("Offsets must include their initial zero")
        for entry in range(len(rankings)):
            flat.append(unsigned_from[DType.uint32](rankings[entry]))
        for index in range(num_ballots + 1):
            offsets.append(unsigned_from[DType.uint64](offsets_arg[index]))
        if ranks_arg is not PythonObject(None):
            if len(ranks_arg) != len(flat):
                raise Error("Ranks must match the entries length")
            for entry in range(len(flat)):
                ranks.append(unsigned_from[DType.uint32](ranks_arg[entry]))
    if num_candidates < 1 or UInt64(num_candidates) > UInt64(UInt32.MAX):
        raise Error("num_candidates must be a positive UInt32 integer")
    if offsets[0] != 0 or offsets[num_ballots] != UInt64(len(flat)):
        raise Error("Offsets must start at zero and end at the entries length")
    for ballot in range(num_ballots):
        if offsets[ballot] > offsets[ballot + 1] or offsets[ballot + 1] > UInt64(len(flat)):
            raise Error("Offsets must be monotone and within the entries length")
    var seen = List[Int]()
    seen.resize(num_candidates, -1)
    for ballot in range(num_ballots):
        var start = Int(offsets[ballot])
        var end = Int(offsets[ballot + 1])
        for entry in range(start, end):
            var candidate = Int(flat[entry])
            if candidate >= num_candidates or seen[candidate] == ballot:
                raise Error("Every ballot must list distinct candidates within the candidate range")
            seen[candidate] = ballot
    if (
        offsets_arg is PythonObject(None)
        and ranks_arg is PythonObject(None)
        and weights_arg is PythonObject(None)
        and dense_width == num_candidates
    ):
        try:
            score_type = ballots.resolve_tally_score_type(UInt64(num_ballots), score_type)
        except error:
            return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
        if UInt64(num_ballots) <= UInt64(UInt32.MAX) and (backend == Backend.cpu or num_candidates <= 64):
            var counted = ballots.tally_ballots(flat, num_ballots, num_candidates, backend=backend)
            return matrix_to_python(counted, score_type)
    if ranks_arg is PythonObject(None):
        for ballot in range(num_ballots):
            for position in range(Int(offsets[ballot + 1] - offsets[ballot])):
                ranks.append(UInt32(position))
    if weights_arg is not PythonObject(None) and len(weights_arg) != num_ballots:
        raise Error("Weights must match the ballot count")
    var bound = UInt64(0)
    for ballot in range(num_ballots):
        weights.append(1 if weights_arg is PythonObject(None) else unsigned_from[DType.uint64](weights_arg[ballot]))
        bound = ballots.add_counts[ballots.Arithmetic.saturated](bound, weights[ballot])
    try:
        score_type = ballots.resolve_tally_score_type(bound, score_type)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
    var counted = ballots.tally_ragged_ballots(
        flat, offsets, ranks, weights, num_candidates, unranked=unranked, backend=backend, score_type=score_type
    )
    for cell in range(num_candidates * num_candidates):
        if counted.data[unsafe_offset=cell] == UInt64.MAX:
            return PythonObject(
                from_owned=raise_python_exception(
                    Error("Tally reached the saturation sentinel"), ExceptionType("PyExc_OverflowError")
                )
            )
    return matrix_to_python(counted, score_type)


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
