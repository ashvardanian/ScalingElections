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

from max.gpu.host import DeviceContext

import ballots
import kemeny
import schulze
from ballots import Backend, PairwiseRelation, ScoreType, Unranked, VoteMatrix, VoteMatrixView

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


def matrix_to_python[StoredCountDataType: DType](matrix: VoteMatrix[StoredCountDataType]) raises -> PythonObject:
    var n = matrix.num_candidates
    var values = Span(unsafe_ptr=matrix.data, length=n * n)
    return copy_to_numpy_tensor(values, Coord(n, n))


def solve[
    Operation: SolverOperation
](preferences: PythonObject, var kwargs: StringDict[PythonObject]) raises -> PythonObject:
    var score_type: ScoreType
    try:
        score_type = score_type_from(kwargs)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_ValueError")))
    for key in kwargs:
        if String(key) != "backend":
            return python_error("Unknown keyword: " + String(key), ExceptionType("PyExc_TypeError"))
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
    if UInt64(n) > UInt64(UInt32.MAX) - UInt64(schulze.TILE_SIZE) or n > Int.MAX // n // 8:
        return python_error("Matrix size exceeds the addressable range", ExceptionType("PyExc_OverflowError"))
    comptime if Operation == SolverOperation.kemeny_ranking or Operation == SolverOperation.kemeny_costs:
        if n > kemeny.KEMENY_MAX_CANDIDATES:
            return python_error(
                "Kemeny supports at most " + String(kemeny.KEMENY_MAX_CANDIDATES) + " candidates",
                ExceptionType("PyExc_ValueError"),
            )
    var kind = String(array.dtype.kind)
    if kind == "O":
        var index = Python.import_module("operator").index
        var boolean_types = Python().tuple(builtins.bool, np.bool_)
        var converted = np.empty(array.shape, dtype="uint64")
        for cell in range(n * n):
            var value = array.flat[cell]
            if Bool(builtins.isinstance(value, boolean_types)) or not Bool(
                builtins.hasattr(builtins.type(value), "__index__")
            ):
                return python_error(
                    "Entries must be integers representable by UInt64", ExceptionType("PyExc_TypeError")
                )
            var integer = index(value)
            if Bool(integer < 0) or Bool(integer > PythonObject(UInt64.MAX)):
                return python_error("Entries must fit UInt64", ExceptionType("PyExc_OverflowError"))
            converted.flat[cell] = integer
        return solve_matrix[Operation, DType.uint64](converted, backend, score_type)
    if (kind != "u" and kind != "i") or integer_from(array.itemsize) > 8:
        return python_error("Entries must be integers representable by UInt64", ExceptionType("PyExc_TypeError"))
    if kind == "i" and Bool(array.min() < 0):
        return python_error("Entries must be nonnegative", ExceptionType("PyExc_OverflowError"))
    if kind == "u" and integer_from(array.itemsize) == 8:
        return solve_matrix[Operation, DType.uint64](np.ascontiguousarray(array, dtype="uint64"), backend, score_type)
    if integer_from(array.itemsize) <= 4 or Bool(array.max() <= PythonObject(UInt32.MAX)):
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
            score_type = schulze.resolve_score_type(matrix, score_type)
        elif Operation == SolverOperation.split_cycle_winners:
            score_type = schulze.resolve_score_type[schulze.SeedGraph.positive_margins](matrix, score_type)
        else:
            score_type = kemeny.resolve_score_type(matrix, score_type)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
    var result: PythonObject
    if score_type == ScoreType.uint16:
        result = solve_typed[Operation, DType.uint16](matrix, backend)
    elif score_type == ScoreType.uint32:
        result = solve_typed[Operation, DType.uint32](matrix, backend)
    elif score_type == ScoreType.uint64:
        result = solve_typed[Operation, DType.uint64](matrix, backend)
    else:
        result = solve_typed[Operation, DType.uint64, ballots.Arithmetic.saturated](matrix, backend)
    return result^


def solve_typed[
    Operation: SolverOperation,
    ArithmeticDataType: DType,
    ArithmeticMode: ballots.Arithmetic = ballots.Arithmetic.exact,
    StoredCountDataType: DType = DType.uint32,
](matrix: VoteMatrixView[StoredCountDataType, _], backend: Backend) raises -> PythonObject:
    comptime if Operation == SolverOperation.strongest_paths:
        var paths = schulze.strongest_paths_typed[ArithmeticDataType, schulze.SeedGraph.winning_votes](matrix, backend)
        return matrix_to_python(paths)
    elif Operation == SolverOperation.split_cycle_winners:
        var undefeated = schulze.split_cycle_winners_typed[ArithmeticDataType](matrix, backend)
        var winners = Python().list()
        for candidate in undefeated:
            winners.append(PythonObject(candidate))
        return winners
    else:
        try:
            kemeny.require_kemeny_score_range[ArithmeticDataType, ArithmeticMode](matrix)
        except error:
            return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
        comptime if Operation == SolverOperation.kemeny_costs:
            return kemeny_costs_to_python[ArithmeticDataType, ArithmeticMode](matrix, backend)
        else:
            var solution = kemeny.compute_kemeny_ranking_gpu[ArithmeticDataType, ArithmeticMode](
                matrix
            ) if backend == Backend.gpu else kemeny.compute_kemeny_ranking_cpu[ArithmeticDataType, ArithmeticMode](
                matrix
            )
            if solution.score == UInt64.MAX:
                return python_error(
                    "Kemeny optimum reached the saturation sentinel", ExceptionType("PyExc_OverflowError")
                )
            var ranking = Python().list()
            for candidate in solution.ranking:
                ranking.append(PythonObject(candidate))
            var winners = Python().list()
            for candidate in solution.winners:
                winners.append(PythonObject(candidate))
            return Python().tuple(
                ranking, PythonObject(solution.score), winners, PythonObject(solution.multiplicity.name())
            )


def compute_strongest_paths(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Compute widest paths with selected-width storage and arithmetic."""
    return solve[SolverOperation.strongest_paths](preferences, kwargs^)


def kemeny_costs_to_python[
    ArithmeticDataType: DType,
    ArithmeticMode: ballots.Arithmetic = ballots.Arithmetic.exact,
    StoredCountDataType: DType = DType.uint64,
](matrix: VoteMatrixView[StoredCountDataType, _], backend: Backend) raises -> PythonObject:
    var n = matrix.num_candidates
    var sums = kemeny.KemenySums[ArithmeticDataType, ArithmeticMode](matrix)
    if backend == Backend.cpu:
        var costs = kemeny.compute_kemeny_costs_cpu(matrix, sums)
        var values = Span(unsafe_ptr=costs.unsafe_ptr(), length=len(costs))
        return copy_to_numpy_tensor(values, Coord(len(costs)))
    var ctx = DeviceContext()
    var (costs, _) = kemeny.compute_kemeny_costs_gpu(ctx, matrix, sums)
    var host_costs = ctx.enqueue_create_host_buffer[ArithmeticDataType](1 << n)
    costs.enqueue_copy_to(host_costs)
    ctx.synchronize()
    return copy_to_numpy_tensor(host_costs.as_span(), Coord(1 << n))


def _compute_kemeny_costs(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Retain one selected-width cost table for enumerating all optimal orderings."""
    return solve[SolverOperation.kemeny_costs](preferences, kwargs^)


def compute_kemeny_ranking(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Compute an exact ranking, disagreement score, winners, and multiplicity."""
    return solve[SolverOperation.kemeny_ranking](preferences, kwargs^)


def python_error(message: String, exception: ExceptionType) -> PythonObject:
    return PythonObject(from_owned=raise_python_exception(Error(message), exception))


def ballot_values[
    StorageDataType: DType
](array: PythonObject,) raises -> Span[SIMD[StorageDataType, 1], ImmUntrackedOrigin]:
    if array is PythonObject(None):
        return Span[SIMD[StorageDataType, 1], ImmUntrackedOrigin]()
    var view = from_numpy_tensor[StorageDataType, 1](array.reshape(-1))
    return Span(unsafe_ptr=view.data.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](), length=len(view.data))


def tally_to_python[
    ArithmeticDataType: DType, ArithmeticMode: ballots.Arithmetic, RelationCount: Int
](prepared: ballots.RaggedBallots, relation: PairwiseRelation, backend: Backend) raises -> PythonObject:
    var counted = ballots.tally_ragged_typed[ArithmeticDataType, ArithmeticMode, RelationCount](
        prepared, relation, backend
    )
    var outputs = Python().list()
    for output in range(RelationCount):
        comptime if ArithmeticMode == ballots.Arithmetic.saturated:
            for cell in range(prepared.num_candidates * prepared.num_candidates):
                if counted[output].data[unsafe_offset=cell] == SIMD[ArithmeticDataType, 1].MAX:
                    return python_error("Tally reached the saturation sentinel", ExceptionType("PyExc_OverflowError"))
        var values = Span(unsafe_ptr=counted[output].data, length=prepared.num_candidates * prepared.num_candidates)
        outputs.append(copy_to_numpy_tensor(values, Coord(prepared.num_candidates, prepared.num_candidates)))
    comptime if RelationCount == 1:
        return outputs[0]
    else:
        return Python.import_module("builtins").tuple(outputs)


def tally_prepared[
    RelationCount: Int
](candidates: PythonObject, var kwargs: StringDict[PythonObject]) raises -> PythonObject:
    """Counts dense or CSR integer-weighted candidates with explicit ties and omission semantics."""
    var offsets_arg = kwargs.pop("offsets") if "offsets" in kwargs else PythonObject(None)
    var count_arg = kwargs.pop("num_candidates") if "num_candidates" in kwargs else PythonObject(None)
    var ranks_arg = kwargs.pop("ranks") if "ranks" in kwargs else PythonObject(None)
    var weights_arg = kwargs.pop("weights") if "weights" in kwargs else PythonObject(None)
    var unranked_arg = kwargs.pop("unranked") if "unranked" in kwargs else PythonObject("unknown")
    var relation_name = String(kwargs.pop("relation")) if "relation" in kwargs else String("preference")
    var relation = PairwiseRelation.preference
    if relation_name == "indifference":
        relation = PairwiseRelation.indifference
    elif relation_name == "unknown":
        relation = PairwiseRelation.unknown
    elif relation_name != "preference":
        return python_error("Unknown pairwise relation", ExceptionType("PyExc_ValueError"))
    var score_type: ScoreType
    try:
        score_type = score_type_from(kwargs)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_ValueError")))
    for key in kwargs:
        if String(key) != "backend":
            return python_error("Unknown keyword: " + String(key), ExceptionType("PyExc_TypeError"))
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
        if Bool(count_arg == 0):
            return python_error("num_candidates must be positive", ExceptionType("PyExc_ValueError"))
    var inputs = Python().tuple(candidates, offsets_arg, ranks_arg, weights_arg)
    var arrays = Python().list()
    for input_index in range(4):
        var values = inputs[input_index]
        if values is PythonObject(None):
            arrays.append(values)
            continue
        var array = np.asarray(values) if Bool(builtins.isinstance(values, np.ndarray)) else np.asarray(
            values, dtype="object"
        )
        var kind = String(array.dtype.kind)
        var dimensions = 2 if offsets_arg is PythonObject(None) and (input_index == 0 or input_index == 2) else 1
        if integer_from(array.ndim) != dimensions:
            return python_error("Ballot arrays have incompatible dimensions", ExceptionType("PyExc_ValueError"))
        if kind == "O":
            for entry in range(integer_from(array.size)):
                var value = array.flat[entry]
                if not Bool(builtins.isinstance(value, integer_types)) or Bool(
                    builtins.isinstance(value, boolean_types)
                ):
                    return python_error("Ballot entries must be integers", ExceptionType("PyExc_TypeError"))
        elif kind != "u" and kind != "i" and integer_from(array.size) != 0:
            return python_error("Ballot entries must be integers", ExceptionType("PyExc_TypeError"))
        var maximum = PythonObject(UInt32.MAX) if input_index == 0 or input_index == 2 else PythonObject(UInt64.MAX)
        if integer_from(array.size) and ((kind != "u" and Bool(array.min() < 0)) or Bool(array.max() > maximum)):
            return python_error(
                "Ballot entries exceed their unsigned storage range", ExceptionType("PyExc_OverflowError")
            )
        var dtype = "uint32" if input_index == 0 or input_index == 2 else "uint64"
        arrays.append(np.ascontiguousarray(array, dtype=dtype))
    var candidate_array = arrays[0]
    offsets_arg = arrays[1]
    ranks_arg = arrays[2]
    weights_arg = arrays[3]
    var flat = ballot_values[DType.uint32](candidate_array.reshape(-1))
    var offset_values = List[ballots.BallotOffset]()
    var offsets = ballot_values[DType.uint64](offsets_arg)
    var ranks = ballot_values[DType.uint32](ranks_arg)
    var weights = ballot_values[DType.uint64](weights_arg)
    var num_candidates: Int
    var num_ballots: Int
    var dense_width = 0
    if offsets_arg is PythonObject(None):
        num_ballots = integer_from(candidate_array.shape[0])
        dense_width = integer_from(candidate_array.shape[1])
        num_candidates = dense_width if count_arg is PythonObject(None) else integer_from(count_arg)
        for ballot in range(num_ballots + 1):
            offset_values.append(UInt64(ballot) * UInt64(dense_width))
        offsets = ballots.ballot_span(offset_values)
        if ranks_arg is not PythonObject(None) and not Bool(ranks_arg.shape == candidate_array.shape):
            return python_error("Ranks must match the candidates shape", ExceptionType("PyExc_ValueError"))
    else:
        if count_arg is PythonObject(None):
            return python_error("CSR ballots require num_candidates", ExceptionType("PyExc_ValueError"))
        num_candidates = integer_from(count_arg)
        num_ballots = len(offsets) - 1
        if num_ballots < 0:
            return python_error("Offsets must include their initial zero", ExceptionType("PyExc_ValueError"))
    if ranks_arg is not PythonObject(None):
        if len(ranks) != len(flat):
            return python_error("Ranks must match the entries length", ExceptionType("PyExc_ValueError"))
    var policy_values = List[ballots.PolicyCode]()
    var unranked = Unranked.unknown
    try:
        if Bool(builtins.isinstance(unranked_arg, builtins.str)):
            unranked = Unranked.parse(String(unranked_arg))
        else:
            if len(unranked_arg) != num_ballots:
                return python_error("Unranked policies must match the ballot count", ExceptionType("PyExc_ValueError"))
            for ballot in range(num_ballots):
                policy_values.append(Unranked.parse(String(unranked_arg[ballot])).value)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_ValueError")))
    if num_candidates < 1 or UInt64(num_candidates) > UInt64(UInt32.MAX):
        return python_error("num_candidates must be a positive UInt32 integer", ExceptionType("PyExc_ValueError"))
    if num_candidates > Int.MAX // num_candidates // (8 * RelationCount):
        return python_error("Matrix size exceeds the addressable range", ExceptionType("PyExc_OverflowError"))
    if offsets[0] != 0 or offsets[num_ballots] != UInt64(len(flat)):
        return python_error(
            "Offsets must start at zero and end at the entries length", ExceptionType("PyExc_ValueError")
        )
    for ballot in range(num_ballots):
        if offsets[ballot] > offsets[ballot + 1] or offsets[ballot + 1] > UInt64(len(flat)):
            return python_error(
                "Offsets must be monotone and within the entries length", ExceptionType("PyExc_ValueError")
            )
    var seen = List[Int]()
    seen.resize(num_candidates, -1)
    for ballot in range(num_ballots):
        var start = Int(offsets[ballot])
        var end = Int(offsets[ballot + 1])
        for entry in range(start, end):
            var candidate = Int(flat[entry])
            if candidate >= num_candidates or seen[candidate] == ballot:
                return PythonObject(
                    from_owned=raise_python_exception(
                        Error("Every ballot must list distinct candidates within the candidate range"),
                        ExceptionType("PyExc_ValueError"),
                    )
                )
            seen[candidate] = ballot
    if (
        RelationCount == 1
        and offsets_arg is PythonObject(None)
        and ranks_arg is PythonObject(None)
        and weights_arg is PythonObject(None)
        and dense_width == num_candidates
        and relation == PairwiseRelation.preference
    ):
        try:
            score_type = ballots.resolve_tally_score_type(UInt64(num_ballots), score_type)
        except error:
            return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
        if score_type == ScoreType.uint32 and (backend == Backend.cpu or num_candidates <= 64):
            var counted = ballots.tally_ballots(flat, num_ballots, num_candidates, backend=backend)
            var result = matrix_to_python(counted)
            _ = arrays^
            return result^
    if weights_arg is not PythonObject(None) and len(weights) != num_ballots:
        return python_error("Weights must match the ballot count", ExceptionType("PyExc_ValueError"))
    var prepared = ballots.RaggedBallots(
        flat,
        offsets,
        ranks,
        weights,
        ballots.ballot_span(policy_values),
        num_candidates,
        unranked,
    )
    var bound = UInt64(num_ballots)
    if len(weights):
        bound = 0
        for ballot in range(num_ballots):
            bound = ballots.add_counts[ballots.Arithmetic.saturated](bound, weights[ballot])
    try:
        score_type = ballots.resolve_tally_score_type(bound, score_type)
    except error:
        return PythonObject(from_owned=raise_python_exception(error, ExceptionType("PyExc_OverflowError")))
    var result: PythonObject
    if score_type == ScoreType.uint16:
        result = tally_to_python[DType.uint16, ballots.Arithmetic.exact, RelationCount](prepared, relation, backend)
    elif score_type == ScoreType.uint32:
        result = tally_to_python[DType.uint32, ballots.Arithmetic.exact, RelationCount](prepared, relation, backend)
    elif score_type == ScoreType.saturated64:
        result = tally_to_python[DType.uint64, ballots.Arithmetic.saturated, RelationCount](prepared, relation, backend)
    else:
        result = tally_to_python[DType.uint64, ballots.Arithmetic.exact, RelationCount](prepared, relation, backend)
    _ = arrays^
    _ = offset_values^
    _ = policy_values^
    return result^


def tally_ballots(candidates: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts one dense or CSR ballot relation with integer weights and explicit ties."""
    return tally_prepared[1](candidates, kwargs^)


def tally_pairwise_relations(candidates: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Counts preferences, effective indifference, and unknown comparisons in one pass."""
    if "relation" in kwargs:
        return python_error(
            "tally_pairwise_relations always returns all three relations", ExceptionType("PyExc_TypeError")
        )
    return tally_prepared[3](candidates, kwargs^)


def compute_split_cycle_winners(preferences: PythonObject, var **kwargs: PythonObject) raises -> PythonObject:
    """Return every candidate undefeated under Split Cycle."""
    return solve[SolverOperation.split_cycle_winners](preferences, kwargs^)


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
        abort(String("Failed to initialize scalingelections_mojo: ", error))


# endregion Python Bindings
