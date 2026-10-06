"""Python bindings for the Mojo implementations, so one test suite can cross-check every backend.

This module holds nothing but argument marshalling and the module table. `main` lives in
`cli.mojo`, because Mojo refuses to emit a shared library from a module that defines one.

Contiguous integer arrays cross the boundary through typed NumPy views, and every result is
allocated as a NumPy array first so the solvers write into it directly. Python sequences retain
checked integer conversion before entering the solvers.
"""

from std.collections import Span, StringDict
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import ExceptionType, PythonModuleBuilder, raise_python_exception
from std.python.numpy import from_numpy_tensor
from std.sys import has_accelerator

from max.gpu.host import DeviceContext

import ballots
import kemeny
import schulze
from ballots import Arithmetic, Backend, PairwiseRelation, ScoreType, Unranked, VoteMatrixView, with_score_type

# region Python Bindings


def available_backends() raises -> PythonObject:
    """Execution targets visible to the Mojo runtime, without running a solver."""
    var names = Python().list()
    names.append("cpu")
    if has_accelerator():
        names.append("gpu")
    return names


def score_type_from(mut kwargs: StringDict[PythonObject]) raises -> ScoreType:
    var requested_type = String(kwargs.pop("score_type")) if "score_type" in kwargs else String("auto")
    return ScoreType.parse(requested_type)


def integer_from(value: PythonObject) raises -> Int:
    """Reads a Python object implementing the integer index protocol."""
    return Int(py=value.__index__())


def python_error(message: String, exception: ExceptionType) -> PythonObject:
    return PythonObject(from_owned=raise_python_exception(Error(message), exception))


def numpy_array[DataType: DType](shape: PythonObject) raises -> PythonObject:
    """A fresh NumPy array a solver fills in place."""
    return Python.import_module("numpy").empty(shape, dtype=String(DataType))


def numpy_data[DataType: DType](array: PythonObject) raises -> Pointer[Scalar[DataType], MutUntrackedOrigin]:
    """The first element of an array `numpy_array` allocated, so contiguous and writable."""
    return Pointer[Scalar[DataType], MutUntrackedOrigin](unsafe_from_address=Int(py=array.ctypes.data))


def unsigned_integers(mut array: PythonObject) raises -> PythonObject:
    """
    Checks `array` holds nonnegative integers below 2^64, converting object entries once through
    `__index__` into a `uint64` array.

    Returns None on success, otherwise the raised Python exception for the caller to return.
    """
    var np = Python.import_module("numpy")
    if Bool(np.issubdtype(array.dtype, np.object_)):
        var builtins = Python.import_module("builtins")
        var index = Python.import_module("operator").index
        var boolean_types = Python().tuple(builtins.bool, np.bool_)
        var converted = np.empty(array.shape, dtype="uint64")
        for cell in range(integer_from(array.size)):
            var value = array.flat[cell]
            if Bool(builtins.isinstance(value, boolean_types)) or not Bool(
                builtins.hasattr(builtins.type(value), "__index__")
            ):
                return python_error("Entries must be integers", ExceptionType("PyExc_TypeError"))
            var integer = index(value)
            if Bool(integer < 0) or Bool(integer > PythonObject(UInt64.MAX)):
                return python_error("Entries must fit UInt64", ExceptionType("PyExc_OverflowError"))
            converted.flat[cell] = integer
        array = converted
        return PythonObject(None)
    if not Bool(np.issubdtype(array.dtype, np.integer)):
        if integer_from(array.size) == 0:
            return PythonObject(None)
        return python_error("Entries must be integers", ExceptionType("PyExc_TypeError"))
    if Bool(np.issubdtype(array.dtype, np.signedinteger)) and integer_from(array.size) and Bool(array.min() < 0):
        return python_error("Entries must be nonnegative", ExceptionType("PyExc_OverflowError"))
    return PythonObject(None)


@fieldwise_init
struct SolverOperation(Copyable, Equatable, ImplicitlyCopyable, Movable, TrivialRegisterPassable):
    """The solver result selected when specializing the shared Python input boundary."""

    var value: UInt8
    comptime strongest_paths = Self(0)
    comptime kemeny_ranking = Self(1)
    comptime kemeny_costs = Self(2)
    comptime split_cycle_winners = Self(3)

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value


def solve[
    Operation: SolverOperation
](preferences: PythonObject, var kwargs: StringDict[PythonObject]) raises -> PythonObject:
    var score_type: ScoreType
    try:
        score_type = score_type_from(kwargs)
    except error:
        return python_error(String(error), ExceptionType("PyExc_ValueError"))
    for key in kwargs:
        if String(key) != "backend":
            return python_error(String(t"Unknown keyword: {key}"), ExceptionType("PyExc_TypeError"))
    var backend_name = String(kwargs["backend"]) if "backend" in kwargs else String("cpu")
    if backend_name != "cpu" and backend_name != "gpu":
        return python_error("backend must be cpu or gpu", ExceptionType("PyExc_ValueError"))
    var backend = Backend.gpu if backend_name == "gpu" else Backend.cpu
    if backend == Backend.gpu and not has_accelerator():
        return python_error("No GPU is available to the Mojo runtime", ExceptionType("PyExc_RuntimeError"))
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var array = np.asarray(preferences) if Bool(builtins.isinstance(preferences, np.ndarray)) else np.asarray(
        preferences, dtype="object"
    )
    if integer_from(array.ndim) != 2:
        return python_error("Preferences must be a nonempty square matrix", ExceptionType("PyExc_ValueError"))
    var n = integer_from(array.shape[0])
    if n < 1 or integer_from(array.shape[1]) != n:
        return python_error("Preferences must be a nonempty square matrix", ExceptionType("PyExc_ValueError"))
    if UInt64(n) > UInt64(ballots.CandidateIndex.MAX) - UInt64(schulze.TILE_SIZE) or n > Int.MAX // n // 8:
        return python_error("Matrix size exceeds the addressable range", ExceptionType("PyExc_OverflowError"))
    comptime if Operation == SolverOperation.kemeny_ranking or Operation == SolverOperation.kemeny_costs:
        try:
            kemeny.require_kemeny_width(n)
        except error:
            return python_error(String(error), ExceptionType("PyExc_ValueError"))
    var failure = unsigned_integers(array)
    if failure is not PythonObject(None):
        return failure
    if Bool(array.dtype == np.uint64):
        return solve_matrix[Operation, DType.uint64](np.ascontiguousarray(array, dtype="uint64"), backend, score_type)
    if Bool(array.max() <= PythonObject(UInt32.MAX)):
        return solve_matrix[Operation, DType.uint32](np.ascontiguousarray(array, dtype="uint32"), backend, score_type)
    return solve_matrix[Operation, DType.uint64](np.ascontiguousarray(array, dtype="uint64"), backend, score_type)


def solve_matrix[
    Operation: SolverOperation, StoredCountDataType: DType
](array: PythonObject, backend: Backend, var score_type: ScoreType) raises -> PythonObject:
    var view = from_numpy_tensor[StoredCountDataType, 2](array)
    var matrix = VoteMatrixView[StoredCountDataType, origin_of(array)](
        Span[SIMD[StoredCountDataType, 1], origin_of(array)](
            unsafe_ptr=view.data.unsafe_ptr().unsafe_origin_cast[origin_of(array)](), length=len(view.data)
        ),
        integer_from(array.shape[0]),
    )
    try:
        comptime if Operation == SolverOperation.strongest_paths:
            score_type = schulze.resolve_score_type[schulze.SeedGraph.winning_votes](matrix, score_type)
        elif Operation == SolverOperation.split_cycle_winners:
            score_type = schulze.resolve_score_type[schulze.SeedGraph.positive_margins](matrix, score_type)
        else:
            score_type = kemeny.resolve_score_type(matrix, score_type)
    except error:
        return python_error(String(error), ExceptionType("PyExc_OverflowError"))

    def solve_with[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> PythonObject:
        return solve_typed[Operation, ArithmeticDataType, ArithmeticMode](matrix, backend)

    return with_score_type(score_type, solve_with)


def solve_typed[
    StoredCountDataType: DType,
    //,
    Operation: SolverOperation,
    ArithmeticDataType: DType,
    ArithmeticMode: Arithmetic,
](matrix: VoteMatrixView[StoredCountDataType, _], backend: Backend) raises -> PythonObject:
    comptime if Operation == SolverOperation.strongest_paths:
        var paths = numpy_array[ArithmeticDataType](Python().tuple(matrix.num_candidates, matrix.num_candidates))
        schulze.strongest_paths_typed[ArithmeticDataType, schulze.SeedGraph.winning_votes](
            matrix, backend, numpy_data[ArithmeticDataType](paths)
        )
        return paths
    elif Operation == SolverOperation.split_cycle_winners:
        var undefeated = schulze.split_cycle_winners_typed[ArithmeticDataType](matrix, backend)
        var winners = Python().list()
        for candidate in undefeated:
            winners.append(PythonObject(candidate))
        return winners
    elif Operation == SolverOperation.kemeny_costs:
        return kemeny_costs_to_python[ArithmeticDataType, ArithmeticMode](matrix, backend)
    else:
        var trace: List[kemeny.KemenyTraceWord]
        try:
            trace = kemeny.compute_kemeny_trace_gpu[ArithmeticDataType, ArithmeticMode](
                matrix
            ) if backend == Backend.gpu else kemeny.compute_kemeny_trace_cpu[ArithmeticDataType, ArithmeticMode](matrix)
        except error:
            return python_error(String(error), ExceptionType("PyExc_RuntimeError"))
        var solution: kemeny.KemenySolution
        try:
            solution = kemeny.kemeny_solution_from_trace[ArithmeticDataType](trace)
        except failure:
            if failure == kemeny.KemenyTraceError.overflow:
                return python_error(String(failure), ExceptionType("PyExc_OverflowError"))
            return python_error(String(failure), ExceptionType("PyExc_RuntimeError"))
        var ranking = Python().list()
        for candidate in solution.ranking:
            ranking.append(PythonObject(candidate))
        var winners = Python().list()
        for candidate in solution.winners:
            winners.append(PythonObject(candidate))
        return Python().tuple(
            ranking, PythonObject(solution.score), winners, PythonObject(solution.multiplicity.name())
        )


def kemeny_costs_to_python[
    StoredCountDataType: DType, //, ArithmeticDataType: DType, ArithmeticMode: Arithmetic
](matrix: VoteMatrixView[StoredCountDataType, _], backend: Backend) raises -> PythonObject:
    var states = 1 << matrix.num_candidates
    var sums = kemeny.KemenySums[ArithmeticDataType, ArithmeticMode](matrix)
    var costs = numpy_array[ArithmeticDataType](PythonObject(states))
    var costs_ptr = numpy_data[ArithmeticDataType](costs)
    if backend == Backend.gpu:
        var ctx = DeviceContext()
        var (device_costs, _) = kemeny.compute_kemeny_costs_gpu(ctx, sums)
        ctx.enqueue_copy(dst_ptr=costs_ptr, src_buf=device_costs)
        ctx.synchronize()
    else:
        kemeny.compute_kemeny_costs_cpu(sums, costs_ptr)
    if costs_ptr[unsafe_offset=states - 1] == SIMD[ArithmeticDataType, 1].MAX:
        return python_error(String(kemeny.KemenyTraceError.overflow), ExceptionType("PyExc_OverflowError"))
    return costs


def compute_strongest_paths(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Compute widest paths with selected-width storage and arithmetic."""
    return solve[SolverOperation.strongest_paths](preferences, kwargs^)


def _compute_kemeny_costs(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Retain one selected-width cost table for enumerating all optimal orderings."""
    return solve[SolverOperation.kemeny_costs](preferences, kwargs^)


def compute_kemeny_ranking(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Compute an exact ranking, disagreement score, winners, and multiplicity."""
    return solve[SolverOperation.kemeny_ranking](preferences, kwargs^)


def compute_split_cycle_winners(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Return every candidate undefeated under Split Cycle."""
    return solve[SolverOperation.split_cycle_winners](preferences, kwargs^)


def ballot_values[
    StorageDataType: DType
](array: PythonObject,) raises -> Span[SIMD[StorageDataType, 1], ImmUntrackedOrigin]:
    if array is PythonObject(None):
        return Span[SIMD[StorageDataType, 1], ImmUntrackedOrigin]()
    var view = from_numpy_tensor[StorageDataType, 1](array.reshape(-1))
    return Span(unsafe_ptr=view.data.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](), length=len(view.data))


def tally_prepared[
    Relation: PairwiseRelation
](candidates: PythonObject, var kwargs: StringDict[PythonObject]) raises -> PythonObject:
    """Counts dense or CSR integer-weighted candidates with explicit ties and omission semantics."""
    comptime planes = Relation.planes()
    var offsets_arg = kwargs.pop("offsets") if "offsets" in kwargs else PythonObject(None)
    var count_arg = kwargs.pop("num_candidates") if "num_candidates" in kwargs else PythonObject(None)
    var ranks_arg = kwargs.pop("ranks") if "ranks" in kwargs else PythonObject(None)
    var weights_arg = kwargs.pop("weights") if "weights" in kwargs else PythonObject(None)
    var unranked_arg = kwargs.pop("unranked") if "unranked" in kwargs else PythonObject("unknown")
    var score_type: ScoreType
    try:
        score_type = score_type_from(kwargs)
    except error:
        return python_error(String(error), ExceptionType("PyExc_ValueError"))
    for key in kwargs:
        if String(key) != "backend":
            return python_error(String(t"Unknown keyword: {key}"), ExceptionType("PyExc_TypeError"))
    var backend_name = String(kwargs["backend"]) if "backend" in kwargs else String("cpu")
    if backend_name != "cpu" and backend_name != "gpu":
        return python_error("backend must be cpu or gpu", ExceptionType("PyExc_ValueError"))
    var backend = Backend.gpu if backend_name == "gpu" else Backend.cpu
    if backend == Backend.gpu and not has_accelerator():
        return python_error("No GPU is available to the Mojo runtime", ExceptionType("PyExc_RuntimeError"))
    var np = Python.import_module("numpy")
    var builtins = Python.import_module("builtins")
    var integer_types = Python().tuple(builtins.int, np.integer)
    var boolean_types = Python().tuple(builtins.bool, np.bool_)
    if count_arg is not PythonObject(None):
        if not Bool(builtins.isinstance(count_arg, integer_types)) or Bool(
            builtins.isinstance(count_arg, boolean_types)
        ):
            return python_error("num_candidates must be an integer", ExceptionType("PyExc_TypeError"))
        if Bool(count_arg < 0) or Bool(count_arg > PythonObject(UInt32.MAX)):
            return python_error("num_candidates must fit UInt32", ExceptionType("PyExc_OverflowError"))
    var inputs = Python().tuple(candidates, offsets_arg, ranks_arg, weights_arg)
    var storage_types = Python().tuple(
        String(ballots.CandidateIndex.dtype),
        String(ballots.BallotOffset.dtype),
        String(ballots.RankLabel.dtype),
        String(ballots.VoterWeight.dtype),
    )
    var arrays = Python().list()
    for input_index in range(4):
        var values = inputs[input_index]
        if values is PythonObject(None):
            arrays.append(values)
            continue
        var array = np.asarray(values) if Bool(builtins.isinstance(values, np.ndarray)) else np.asarray(
            values, dtype="object"
        )
        var dimensions = 2 if offsets_arg is PythonObject(None) and (input_index == 0 or input_index == 2) else 1
        if integer_from(array.ndim) != dimensions:
            return python_error("Ballot arrays have incompatible dimensions", ExceptionType("PyExc_ValueError"))
        var failure = unsigned_integers(array)
        if failure is not PythonObject(None):
            return failure
        var storage = storage_types[input_index]
        if integer_from(array.size) and Bool(array.max() > np.iinfo(storage).max):
            return python_error(
                "Ballot entries exceed their unsigned storage range", ExceptionType("PyExc_OverflowError")
            )
        arrays.append(np.ascontiguousarray(array, dtype=storage))
    var candidate_array = arrays[0]
    offsets_arg = arrays[1]
    ranks_arg = arrays[2]
    weights_arg = arrays[3]
    var flat = ballot_values[ballots.CandidateIndex.dtype](candidate_array.reshape(-1))
    var offset_values = List[ballots.BallotOffset]()
    var offsets = ballot_values[ballots.BallotOffset.dtype](offsets_arg)
    var num_candidates: Int
    var dense_width = 0
    if offsets_arg is PythonObject(None):
        dense_width = integer_from(candidate_array.shape[1])
        num_candidates = dense_width if count_arg is PythonObject(None) else integer_from(count_arg)
        for ballot in range(integer_from(candidate_array.shape[0]) + 1):
            offset_values.append(ballots.BallotOffset(ballot) * ballots.BallotOffset(dense_width))
        offsets = ballots.ballot_span(offset_values)
        if ranks_arg is not PythonObject(None) and not Bool(ranks_arg.shape == candidate_array.shape):
            return python_error("Ranks must match the candidates shape", ExceptionType("PyExc_ValueError"))
    else:
        if count_arg is PythonObject(None):
            return python_error("CSR ballots require num_candidates", ExceptionType("PyExc_ValueError"))
        num_candidates = integer_from(count_arg)
    var policy_values = List[ballots.PolicyCode]()
    var unranked = Unranked.unknown
    try:
        if Bool(builtins.isinstance(unranked_arg, builtins.str)):
            unranked = Unranked.parse(String(unranked_arg))
        else:
            if len(unranked_arg) != len(offsets) - 1:
                raise Error("Unranked policies must match the ballot count")
            for ballot in range(len(unranked_arg)):
                policy_values.append(Unranked.parse(String(unranked_arg[ballot])).value)
    except error:
        return python_error(String(error), ExceptionType("PyExc_ValueError"))
    var prepared = ballots.RaggedBallots(
        flat,
        offsets,
        ballot_values[ballots.RankLabel.dtype](ranks_arg),
        ballot_values[ballots.VoterWeight.dtype](weights_arg),
        ballots.ballot_span(policy_values),
        num_candidates,
        unranked,
    )
    var bound: ballots.VoterWeight
    try:
        bound = ballots.validate_ragged_ballots(prepared)
    except error:
        return python_error(String(error), ExceptionType("PyExc_ValueError"))
    if num_candidates > Int.MAX // num_candidates // (8 * planes):
        return python_error("Matrix size exceeds the addressable range", ExceptionType("PyExc_OverflowError"))
    try:
        score_type = ballots.resolve_tally_score_type(bound, score_type)
    except error:
        return python_error(String(error), ExceptionType("PyExc_OverflowError"))

    def tally_shape() raises {imm} -> PythonObject:
        comptime if Relation == PairwiseRelation.all:
            return Python().tuple(planes, num_candidates, num_candidates)
        else:
            return Python().tuple(num_candidates, num_candidates)

    def tally[ArithmeticDataType: DType, ArithmeticMode: Arithmetic]() raises {imm} -> PythonObject:
        var counts = numpy_array[ArithmeticDataType](tally_shape())
        var counts_ptr = numpy_data[ArithmeticDataType](counts)
        # Complete rankings take the dense kernel wherever it can count them in this arithmetic.
        if (
            Relation == PairwiseRelation.preference
            and dense_width == num_candidates
            and len(prepared.ranks) == 0
            and len(prepared.weights) == 0
            and (
                backend == Backend.cpu
                or ballots.tally_dense_serves_gpu[ArithmeticDataType, ArithmeticMode](DeviceContext(), num_candidates)
            )
        ):
            ballots.tally_ballots[ArithmeticDataType, ArithmeticMode](
                flat, len(offsets) - 1, num_candidates, counts_ptr, backend=backend
            )
        else:
            ballots.tally_ragged_typed[ArithmeticDataType, ArithmeticMode, Relation](prepared, backend, counts_ptr)
        comptime if ArithmeticMode == Arithmetic.saturated:
            for cell in range(num_candidates * num_candidates * planes):
                if counts_ptr[unsafe_offset=cell] == SIMD[ArithmeticDataType, 1].MAX:
                    return python_error("Tally reaches the overflow sentinel", ExceptionType("PyExc_OverflowError"))
        comptime if Relation == PairwiseRelation.all:
            return Python.import_module("builtins").tuple(counts)
        else:
            return counts

    var result = with_score_type(score_type, tally)
    _ = arrays^
    _ = offset_values^
    _ = policy_values^
    return result^


def tally_ballots(candidates: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts one dense or CSR ballot relation with integer weights and explicit ties."""
    var relation = String(kwargs.pop("relation")) if "relation" in kwargs else String("preference")
    if relation == "preference":
        return tally_prepared[PairwiseRelation.preference](candidates, kwargs^)
    if relation == "indifference":
        return tally_prepared[PairwiseRelation.indifference](candidates, kwargs^)
    if relation == "unknown":
        return tally_prepared[PairwiseRelation.unknown](candidates, kwargs^)
    return python_error("relation must be preference, indifference, or unknown", ExceptionType("PyExc_ValueError"))


def tally_pairwise_relations(candidates: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts preferences, effective indifference, and unknown comparisons in one pass."""
    if "relation" in kwargs:
        return python_error(
            "tally_pairwise_relations always returns all three relations", ExceptionType("PyExc_TypeError")
        )
    return tally_prepared[PairwiseRelation.all](candidates, kwargs^)


@export
def PyInit_scalingelections_mojo() abi("C") -> PythonObject:
    try:
        var builder = PythonModuleBuilder("scalingelections_mojo")
        builder.def_function[available_backends]("available_backends")
        builder.def_function[tally_ballots]("tally_ballots")
        builder.def_function[tally_pairwise_relations]("tally_pairwise_relations")
        builder.def_function[compute_strongest_paths]("compute_strongest_paths")
        builder.def_function[compute_kemeny_ranking]("compute_kemeny_ranking")
        builder.def_function[_compute_kemeny_costs]("_compute_kemeny_costs")
        builder.def_function[compute_split_cycle_winners]("compute_split_cycle_winners")
        return builder.finalize()
    except error:
        abort(String(t"Failed to initialize scalingelections_mojo: {error}"))


# endregion Python Bindings
