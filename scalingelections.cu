/**
 *  @brief Python bindings for the ballot tallies and the Schulze, Split Cycle, and Kemeny-Young solvers.
 *  @file scalingelections.cu
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#include <memory> // `std::make_unique`
#include <mutex>  // `std::mutex`, `std::lock_guard`
#include <span>   // `std::span`

#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "types.cuh"
#include "ballots.cuh"
#include "schulze.cuh"
#include "kemeny.cuh"

namespace py = pybind11;

#if defined(SCALING_ELECTIONS_WITH_GPU)

/** How many devices the runtime can see, without launching a kernel. */
inline int gpu_device_count_() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

#else

inline int gpu_device_count_() noexcept { return 0; }

#endif

/** Resolves the execution target shared by every operation. */
inline backend_t backend_from_name(std::string_view name) {
    if (name == "cpu") return backend_t::cpu_k;
    if (name != "gpu") throw std::invalid_argument("backend must be cpu or gpu");
    if (gpu_device_count_() == 0) throw std::runtime_error("No GPU devices available");
    return backend_t::gpu_k;
}

inline score_type_t score_type_from_name(std::string_view name) {
    if (name == "auto") return score_type_t::auto_k;
    if (name == "saturated64") return score_type_t::saturated64_k;
    if (name == "uint64") return score_type_t::uint64_k;
    if (name == "uint32") return score_type_t::uint32_k;
    if (name == "uint16") return score_type_t::uint16_k;
    throw std::invalid_argument("score_type must be auto, saturated64, uint64, uint32, or uint16");
}

inline unranked_t unranked_from_name(std::string_view name) {
    if (name == "unknown") return unranked_t::unknown_k;
    if (name == "worse") return unranked_t::worse_k;
    throw std::invalid_argument("unranked must be unknown or worse");
}

inline pairwise_relation_t relation_from_name(std::string_view name) {
    if (name == "preference") return pairwise_relation_t::preference_k;
    if (name == "indifference") return pairwise_relation_t::indifference_k;
    if (name == "unknown") return pairwise_relation_t::unknown_k;
    throw std::invalid_argument("relation must be preference, indifference, or unknown");
}

/** Stores the interrupt signal status. */
volatile std::sig_atomic_t global_signal_status = 0;

void signal_handler(int signal) { global_signal_status = signal; }

/**
 *  Routes SIGINT into `global_signal_status` while any solver runs, then hands it back to Python.
 *  Concurrent calls share one installation, so only the last to leave restores the prior handler.
 */
struct interrupt_scope_t {
    interrupt_scope_t() {
        std::lock_guard<std::mutex> const lock(mutex_);
        if (depth_++ != 0) return;
        global_signal_status = 0;
        previous_ = std::signal(SIGINT, signal_handler);
    }
    ~interrupt_scope_t() noexcept {
        std::lock_guard<std::mutex> const lock(mutex_);
        if (--depth_ != 0) return;
        std::signal(SIGINT, previous_);
        // Replaying the signal lets Python raise its own `KeyboardInterrupt` once the solver unwinds.
        if (global_signal_status) std::raise(global_signal_status);
    }
    interrupt_scope_t(interrupt_scope_t const&) = delete;
    interrupt_scope_t& operator=(interrupt_scope_t const&) = delete;

  private:
    using handler_t = void (*)(int);
    static inline std::mutex mutex_;
    static inline std::size_t depth_ = 0;
    static inline handler_t previous_ = SIG_DFL;
};

#if defined(SCALING_ELECTIONS_WITH_OPENMP)

/** Fixes the team size for one call, restoring the runtime's prior settings on scope exit. */
struct openmp_team_t {
    int const previous_threads;
    int const previous_dynamic;

    explicit openmp_team_t(unsigned threads)
        : previous_threads(omp_get_max_threads()), previous_dynamic(omp_get_dynamic()) {
        omp_set_dynamic(0);
        // `hardware_concurrency` is permitted to answer zero, which OpenMP rejects.
        if (threads > 0) omp_set_num_threads(static_cast<int>(threads));
    }
    ~openmp_team_t() noexcept {
        omp_set_num_threads(previous_threads);
        omp_set_dynamic(previous_dynamic);
    }
    openmp_team_t(openmp_team_t const&) = delete;
    openmp_team_t& operator=(openmp_team_t const&) = delete;
};

#else

/** Without OpenMP every solver runs on the calling thread, so there is no team to fix. */
struct openmp_team_t {
    explicit openmp_team_t(unsigned) noexcept {}
};

#endif

#pragma region Python bindings

/** How NumPy stores the entries of an input array. */
enum class integer_storage_t : std::uint8_t {
    /** Python objects, each converted through its own integer protocol. */
    objects_k,
    /** A NumPy integer type, at most 64 bits wide. */
    integers_k,
};

/** A validated integer array before narrowing, with its largest entry when NumPy stores it natively. */
struct integer_source_t {
    py::array array;
    integer_storage_t storage;
    std::uint64_t maximum;
};

/** Whether the entries of @p array belong to the NumPy abstract type named @p kind . */
static bool is_numpy_subtype_(py::module_ const& numpy, py::array const& array, char const* kind) {
    return numpy.attr("issubdtype")(array.dtype(), numpy.attr(kind)).cast<bool>();
}

static integer_source_t integer_source_(py::module_ const& numpy, py::handle values) {
    py::array array = py::isinstance<py::array>(values)
                          ? py::reinterpret_borrow<py::array>(values)
                          : numpy.attr("asarray")(values, py::arg("dtype") = "object").cast<py::array>();
    if (is_numpy_subtype_(numpy, array, "object_")) return {std::move(array), integer_storage_t::objects_k, 0};
    if (!is_numpy_subtype_(numpy, array, "integer"))
        throw py::type_error("Entries must be integers representable by UInt64");
    if (!array.size()) return {std::move(array), integer_storage_t::integers_k, 0};
    if (is_numpy_subtype_(numpy, array, "signedinteger") && array.attr("min")().cast<std::int64_t>() < 0)
        throw std::overflow_error("Entries must be nonnegative");
    std::uint64_t const maximum = array.attr("max")().cast<std::uint64_t>();
    return {std::move(array), integer_storage_t::integers_k, maximum};
}

template <typename count_type_>
static py::array_t<count_type_> integer_array_(py::module_ const& numpy, py::handle values) {
    using count_t = count_type_;
    auto const [source, storage, maximum] = integer_source_(numpy, values);
    if (storage == integer_storage_t::objects_k) {
        py::array_t<count_t> result(std::vector<py::ssize_t>(source.shape(), source.shape() + source.ndim()));
        auto output = result.mutable_data();
        auto const numpy_bool = numpy.attr("bool_");
        for (auto value : source.attr("flat")) {
            if (PyBool_Check(value.ptr()) || py::isinstance(value, numpy_bool) || !PyIndex_Check(value.ptr()))
                throw py::type_error("Entries must be integers representable by UInt64");
            auto integer = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
            if (!integer) throw py::error_already_set();
            auto const count = PyLong_AsUnsignedLongLong(integer.ptr());
            if (PyErr_Occurred()) throw py::error_already_set();
            if (count > std::numeric_limits<count_t>::max())
                throw std::overflow_error("Entries exceed the requested integer representation");
            *output++ = static_cast<count_t>(count);
        }
        return result;
    }
    if (maximum > std::numeric_limits<count_t>::max())
        throw std::overflow_error("Entries exceed the requested integer representation");
    return py::array_t<count_t, py::array::c_style | py::array::forcecast>::ensure(source);
}

/** Stores pairwise counts as uint32 where every entry fits, else as uint64, which callers tell apart by item size. */
static py::array counts_array_(py::module_ const& numpy, py::handle values) {
    auto const [source, storage, maximum] = integer_source_(numpy, values);
    if (storage == integer_storage_t::objects_k) return integer_array_<std::uint64_t>(numpy, source);
    if (maximum > std::numeric_limits<std::uint32_t>::max() || is_numpy_subtype_(numpy, source, "uint64"))
        return py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>::ensure(source);
    return py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>::ensure(source);
}

static candidate_index_t matrix_size_(py::array const& preferences) {
    if (preferences.ndim() != 2 || preferences.shape(0) != preferences.shape(1) || preferences.shape(0) < 1)
        throw std::invalid_argument("Preferences must be a nonempty square matrix");
    auto const n = static_cast<std::size_t>(preferences.shape(0));
    if (n > std::numeric_limits<candidate_index_t>::max() - tile_size_k)
        throw std::overflow_error("Candidate count exceeds the index range");
    checked_product(checked_product(n, n), preferences.itemsize());
    return static_cast<candidate_index_t>(n);
}

#pragma region Schulze

template <seed_graph_t seed_, tally_arithmetic arithmetic_type_, tally_arithmetic output_count_type_,
          typename stored_count_type_>
static py::array strongest_paths_typed(strided_matrix<stored_count_type_ const> preferences, backend_t backend) {
    using output_count_t = output_count_type_;
    std::size_t const n = preferences.rows;
    py::array_t<output_count_t> result({n, n});
    strided_matrix<output_count_t> const paths = square_view(result.mutable_data(), n, n);
    {
        py::gil_scoped_release release;
        switch (backend) {
        // The CPU sweeps in place over the result, so it never narrows.
        case backend_t::cpu_k:
            compute_strongest_paths_tiled_cpu<tile_size_k, seed_>(preferences, paths, &global_signal_status);
            break;
        case backend_t::gpu_k:
            compute_strongest_paths_gpu<tile_size_k, seed_, arithmetic_type_>(preferences, paths);
            break;
        }
    }
    return result;
}

template <seed_graph_t seed_, typename stored_count_type_>
static py::array strongest_paths_over(strided_matrix<stored_count_type_ const> preferences, backend_t backend,
                                      score_type_t score_type) {
    switch (schulze_resolve_score_type<seed_>(preferences, score_type)) {
    case score_type_t::saturated64_k:
    case score_type_t::uint64_k:
        return strongest_paths_typed<seed_, std::uint64_t, std::uint64_t>(preferences, backend);
    case score_type_t::uint32_k:
        // An automatic GPU request sweeps packed 16-bit pairs when every edge fits, and still returns 32 bits.
        if (score_type == score_type_t::auto_k && backend == backend_t::gpu_k &&
            packed_min_max_available_<tile_size_k>() &&
            schulze_widest_edge_<seed_>(preferences) <= std::numeric_limits<std::uint16_t>::max())
            return strongest_paths_typed<seed_, std::uint16_t, std::uint32_t>(preferences, backend);
        return strongest_paths_typed<seed_, std::uint32_t, std::uint32_t>(preferences, backend);
    case score_type_t::uint16_k:
        return strongest_paths_typed<seed_, std::uint16_t, std::uint16_t>(preferences, backend);
    case score_type_t::auto_k: break;
    }
    throw std::invalid_argument("score_type must be auto, saturated64, uint64, uint32, or uint16");
}

static py::array compute_strongest_paths(py::handle values, std::string_view backend_name,
                                         std::string_view score_name) {
    py::module_ const numpy = py::module_::import("numpy");
    py::array const preferences = counts_array_(numpy, values);
    candidate_index_t const n = matrix_size_(preferences);
    backend_t const backend = backend_from_name(backend_name);
    score_type_t const score_type = score_type_from_name(score_name);
    interrupt_scope_t const interrupts;
    openmp_team_t const team(std::thread::hardware_concurrency());
    switch (preferences.itemsize()) {
    case sizeof(std::uint32_t):
        return strongest_paths_over<seed_graph_t::winning_votes_k>(
            square_view(static_cast<std::uint32_t const*>(preferences.data()), n, n), backend, score_type);
    case sizeof(std::uint64_t):
        return strongest_paths_over<seed_graph_t::winning_votes_k>(
            square_view(static_cast<std::uint64_t const*>(preferences.data()), n, n), backend, score_type);
    }
    throw std::invalid_argument("Preferences must be stored in 32 or 64 bits");
}

template <typename stored_count_type_>
static std::vector<candidate_index_t> split_cycle_over(strided_matrix<stored_count_type_ const> preferences,
                                                       backend_t backend, score_type_t score_type) {
    auto const paths = py::array_t<stored_count_type_>(
        strongest_paths_over<seed_graph_t::positive_margins_k>(preferences, backend, score_type));
    std::size_t const n = preferences.rows;
    return select_split_cycle_winners(preferences, square_view(paths.data(), n, n));
}

static std::vector<candidate_index_t> compute_split_cycle_winners(py::handle values, std::string_view backend_name,
                                                                  std::string_view score_name) {
    py::module_ const numpy = py::module_::import("numpy");
    py::array const preferences = counts_array_(numpy, values);
    candidate_index_t const n = matrix_size_(preferences);
    backend_t const backend = backend_from_name(backend_name);
    score_type_t const score_type = score_type_from_name(score_name);
    interrupt_scope_t const interrupts;
    openmp_team_t const team(std::thread::hardware_concurrency());
    switch (preferences.itemsize()) {
    case sizeof(std::uint32_t):
        return split_cycle_over(square_view(static_cast<std::uint32_t const*>(preferences.data()), n, n), backend,
                                score_type);
    case sizeof(std::uint64_t):
        return split_cycle_over(square_view(static_cast<std::uint64_t const*>(preferences.data()), n, n), backend,
                                score_type);
    }
    throw std::invalid_argument("Preferences must be stored in 32 or 64 bits");
}

#pragma endregion Schulze

#pragma region Kemeny

template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
static py::tuple kemeny_ranking_typed(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                      backend_t backend) {
    kemeny_solution_t solution;
    {
        py::gil_scoped_release release;
        solution = compute_kemeny_ranking<arithmetic_type_>(preferences, backend, &global_signal_status);
    }
    return py::make_tuple(solution.ranking, solution.score, solution.winners,
                          solution.multiplicity == ranking_multiplicity_t::unique_k ? "unique" : "multiple");
}

template <typename stored_count_type_>
static py::tuple kemeny_ranking_over(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                     backend_t backend, score_type_t score_type) {
    switch (kemeny_resolve_score_type(preferences, score_type)) {
    case score_type_t::saturated64_k: return kemeny_ranking_typed<saturated<std::uint64_t>>(preferences, backend);
    case score_type_t::uint64_k: return kemeny_ranking_typed<std::uint64_t>(preferences, backend);
    case score_type_t::uint32_k: return kemeny_ranking_typed<std::uint32_t>(preferences, backend);
    case score_type_t::uint16_k: return kemeny_ranking_typed<std::uint16_t>(preferences, backend);
    case score_type_t::auto_k: break;
    }
    throw std::invalid_argument("score_type must be auto, saturated64, uint64, uint32, or uint16");
}

static py::tuple compute_kemeny_ranking_py(py::handle values, std::string_view backend_name,
                                           std::string_view score_name) {
    py::module_ const numpy = py::module_::import("numpy");
    py::array const preferences = counts_array_(numpy, values);
    candidate_index_t const n = matrix_size_(preferences);
    backend_t const backend = backend_from_name(backend_name);
    score_type_t const score_type = score_type_from_name(score_name);
    interrupt_scope_t const interrupts;
    openmp_team_t const team(std::thread::hardware_concurrency());
    switch (preferences.itemsize()) {
    case sizeof(std::uint32_t):
        return kemeny_ranking_over(square_view<std::uint32_t const, candidate_index_t>(
                                       static_cast<std::uint32_t const*>(preferences.data()), n, n),
                                   backend, score_type);
    case sizeof(std::uint64_t):
        return kemeny_ranking_over(square_view<std::uint64_t const, candidate_index_t>(
                                       static_cast<std::uint64_t const*>(preferences.data()), n, n),
                                   backend, score_type);
    }
    throw std::invalid_argument("Preferences must be stored in 32 or 64 bits");
}

/** Hands an owned cost table to NumPy without a copy, from a caller that released the GIL. */
template <tally_arithmetic output_count_type_, typename costs_type_>
static py::array kemeny_costs_to_python_(costs_type_ costs) {
    using costs_t = costs_type_;
    using output_count_t = output_count_type_;
    using arithmetic_t = typename costs_t::value_type;
    static_assert(sizeof(output_count_t) == sizeof(arithmetic_t), "NumPy reads the table in place");
    if (costs.back() == std::numeric_limits<arithmetic_t>::max())
        throw std::overflow_error("Kemeny optimum reaches the overflow sentinel");
    py::gil_scoped_acquire const acquire;
    auto owner = std::make_unique<costs_t>(std::move(costs));
    py::capsule lifetime(owner.get(), [](void* data) { delete static_cast<costs_t*>(data); });
    auto const retained = owner.release();
    return py::array(py::dtype::of<output_count_t>(), {retained->size()}, {sizeof(output_count_t)}, retained->data(),
                     lifetime);
}

template <tally_arithmetic arithmetic_type_, tally_arithmetic output_count_type_, typename stored_count_type_>
static py::array kemeny_costs_typed(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                    backend_t backend) {
    using arithmetic_t = arithmetic_type_;
    using output_count_t = output_count_type_;
    kemeny_sums<arithmetic_t> const sums(preferences);
    py::gil_scoped_release release;
    switch (backend) {
    case backend_t::cpu_k:
        return kemeny_costs_to_python_<output_count_t>(
            compute_kemeny_costs_cpu<arithmetic_t>(preferences, sums, &global_signal_status));
    case backend_t::gpu_k:
        return kemeny_costs_to_python_<output_count_t>(compute_kemeny_costs_gpu<arithmetic_t>(preferences, sums));
    }
    throw std::invalid_argument("backend must be cpu or gpu");
}

template <typename stored_count_type_>
static py::array kemeny_costs_over(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                   backend_t backend, score_type_t score_type) {
    switch (kemeny_resolve_score_type(preferences, score_type)) {
    case score_type_t::saturated64_k:
        return kemeny_costs_typed<saturated<std::uint64_t>, std::uint64_t>(preferences, backend);
    case score_type_t::uint64_k: return kemeny_costs_typed<std::uint64_t, std::uint64_t>(preferences, backend);
    case score_type_t::uint32_k: return kemeny_costs_typed<std::uint32_t, std::uint32_t>(preferences, backend);
    case score_type_t::uint16_k: return kemeny_costs_typed<std::uint16_t, std::uint16_t>(preferences, backend);
    case score_type_t::auto_k: break;
    }
    throw std::invalid_argument("score_type must be auto, saturated64, uint64, uint32, or uint16");
}

static py::array compute_kemeny_costs_py(py::handle values, std::string_view backend_name,
                                         std::string_view score_name) {
    py::module_ const numpy = py::module_::import("numpy");
    py::array const preferences = counts_array_(numpy, values);
    candidate_index_t const n = matrix_size_(preferences);
    backend_t const backend = backend_from_name(backend_name);
    score_type_t const score_type = score_type_from_name(score_name);
    interrupt_scope_t const interrupts;
    openmp_team_t const team(std::thread::hardware_concurrency());
    switch (preferences.itemsize()) {
    case sizeof(std::uint32_t):
        return kemeny_costs_over(square_view<std::uint32_t const, candidate_index_t>(
                                     static_cast<std::uint32_t const*>(preferences.data()), n, n),
                                 backend, score_type);
    case sizeof(std::uint64_t):
        return kemeny_costs_over(square_view<std::uint64_t const, candidate_index_t>(
                                     static_cast<std::uint64_t const*>(preferences.data()), n, n),
                                 backend, score_type);
    }
    throw std::invalid_argument("Preferences must be stored in 32 or 64 bits");
}

#pragma endregion Kemeny

#pragma region Tally

/** Whether ballots arrive as complete rankings that the dense kernels read row by row. */
enum class ballot_layout_t : std::uint8_t {
    /** Every ballot lists every candidate once, with positions as ranks and unit weights. */
    complete_k,
    /** Ballots may omit candidates, tie them, or carry weights. */
    ragged_k,
};

/** Counts into @p counts with the dense kernels where the layout and their capacity allow, else ragged. */
template <tally_arithmetic arithmetic_type_>
static void tally_into_(ragged_ballots ballots, ballot_layout_t layout, strided_matrix<arithmetic_type_> counts,
                        backend_t backend, pairwise_relation_t relation) {
    py::gil_scoped_release release;
    std::size_t const n = ballots.num_candidates;
    if (layout == ballot_layout_t::complete_k && relation == pairwise_relation_t::preference_k &&
        (backend == backend_t::cpu_k || tally_dense_fits_<arithmetic_type_>(ballots.num_candidates)))
        return tally_ballots(strided_view<candidate_index_t const>(ballots.candidates, ballots.num_ballots, n, n),
                             counts, backend, &global_signal_status);
    tally_ballots(ballots, counts, backend, relation, &global_signal_status);
}

template <tally_arithmetic arithmetic_type_, tally_arithmetic output_count_type_>
static py::object tally_typed(ragged_ballots ballots, ballot_layout_t layout, backend_t backend,
                              pairwise_relation_t relation) {
    using arithmetic_t = arithmetic_type_;
    using output_count_t = output_count_type_;
    std::size_t const n = ballots.num_candidates;
    std::size_t const planes = relation_planes_(relation);
    std::size_t const rows = checked_product(n, planes);
    std::size_t const cells = checked_product(rows, n);
    checked_product(cells, sizeof(arithmetic_t));
    py::array_t<output_count_t> result({rows, n});
    output_count_t* const output = result.mutable_data();
    if constexpr (std::is_same_v<arithmetic_t, output_count_t>) {
        std::fill(output, output + cells, output_count_t {0});
        tally_into_(ballots, layout, strided_view(output, rows, n, n), backend, relation);
    }
    else {
        std::vector<arithmetic_t> counts(cells, arithmetic_t {0});
        tally_into_(ballots, layout, strided_view(counts.data(), rows, n, n), backend, relation);
        std::transform(counts.begin(), counts.end(), output,
                       [](arithmetic_t count) { return static_cast<output_count_t>(count); });
    }
    if constexpr (std::is_same_v<arithmetic_t, saturated<std::uint64_t>>)
        if (std::find(output, output + cells, std::numeric_limits<output_count_t>::max()) != output + cells)
            throw std::overflow_error("Tally reaches the overflow sentinel");
    if (relation != pairwise_relation_t::all_k) return result;
    py::tuple matrices(planes);
    for (std::size_t plane = 0; plane < planes; ++plane)
        matrices[plane] = py::array_t<output_count_t>({n, n}, {sizeof(output_count_t) * n, sizeof(output_count_t)},
                                                      output + plane * n * n, result);
    return matrices;
}

static py::object tally_ragged_py(py::handle values, py::object offsets_arg, py::object num_candidates_arg,
                                  py::object ranks_arg, py::object weights_arg, py::object unranked_arg,
                                  pairwise_relation_t relation, std::string_view score_name,
                                  std::string_view backend_name) {
    py::module_ const numpy = py::module_::import("numpy");
    auto candidates = integer_array_<candidate_index_t>(numpy, values);
    std::span<candidate_index_t const> const candidate_ids(candidates.data(),
                                                           static_cast<std::size_t>(candidates.size()));
    backend_t const backend = backend_from_name(backend_name);
    score_type_t score_type = score_type_from_name(score_name);
    unranked_t unranked = unranked_t::unknown_k;
    std::vector<unranked_t> policies;
    if (py::isinstance<py::str>(unranked_arg)) unranked = unranked_from_name(py::cast<std::string>(unranked_arg));
    else
        for (auto value : unranked_arg) policies.push_back(unranked_from_name(py::cast<std::string>(value)));

    std::optional<py::array_t<ballot_offset_t>> offsets;
    std::span<ballot_offset_t const> ballot_offsets;
    std::size_t num_ballots = 0;
    std::size_t n = 0;
    if (!num_candidates_arg.is_none()) {
        auto const count = integer_array_<candidate_index_t>(numpy, num_candidates_arg);
        if (count.ndim() != 0) throw py::type_error("num_candidates must be an integer scalar");
        n = *count.data();
    }
    std::size_t dense_width = 0;
    if (offsets_arg.is_none()) {
        if (candidates.ndim() != 2) throw std::invalid_argument("Dense rankings must be two-dimensional");
        num_ballots = candidates.shape(0);
        dense_width = candidates.shape(1);
        if (num_candidates_arg.is_none()) n = dense_width;
    }
    else {
        if (candidates.ndim() != 1) throw std::invalid_argument("CSR candidates must be one-dimensional");
        offsets = integer_array_<ballot_offset_t>(numpy, offsets_arg);
        ballot_offsets = {offsets->data(), static_cast<std::size_t>(offsets->size())};
        if (offsets->ndim() != 1 || ballot_offsets.empty() || ballot_offsets.back() != candidate_ids.size())
            throw std::invalid_argument("Offsets must span the candidate entries");
        num_ballots = ballot_offsets.size() - 1;
        if (num_candidates_arg.is_none()) throw std::invalid_argument("CSR ballots require num_candidates");
    }
    if (n < 1 || n > std::numeric_limits<candidate_index_t>::max())
        throw std::invalid_argument("Candidate count exceeds the supported index range");
    if (!py::isinstance<py::str>(unranked_arg) && policies.size() != num_ballots)
        throw std::invalid_argument("Unranked policies must match the ballot count");
    checked_product(checked_product(n, n), sizeof(voter_weight_t));

    std::optional<py::array_t<rank_label_t>> ranks;
    std::optional<py::array_t<voter_weight_t>> weights;
    if (!ranks_arg.is_none()) {
        ranks = integer_array_<rank_label_t>(numpy, ranks_arg);
        if (ranks->size() != candidates.size() ||
            (ranks->ndim() != 1 && (candidates.ndim() != 2 || ranks->ndim() != 2)) ||
            (ranks->ndim() == 2 && (ranks->shape(0) != candidates.shape(0) || ranks->shape(1) != candidates.shape(1))))
            throw std::invalid_argument("Ranks must match the candidate entries");
    }
    if (!weights_arg.is_none()) {
        weights = integer_array_<voter_weight_t>(numpy, weights_arg);
        if (weights->ndim() != 1 || static_cast<std::size_t>(weights->size()) != num_ballots)
            throw std::invalid_argument("Weights must have one entry per voter");
    }
    if (!offsets) {
        checked_product(num_ballots + 1, sizeof(ballot_offset_t));
        offsets.emplace(num_ballots + 1);
        ballot_offset_t* const output = offsets->mutable_data();
        for (std::size_t ballot = 0; ballot <= num_ballots; ++ballot) output[ballot] = ballot * dense_width;
        ballot_offsets = {output, num_ballots + 1};
    }
    ragged_ballots const ballots {candidate_ids.data(),
                                  ballot_offsets.data(),
                                  ranks ? ranks->data() : nullptr,
                                  weights ? weights->data() : nullptr,
                                  num_ballots,
                                  static_cast<candidate_index_t>(n),
                                  unranked,
                                  policies.empty() ? nullptr : policies.data()};
    validate_ragged_ballots_(ballots);
    score_type = tally_resolve_score_type(ballots, score_type);

    ballot_layout_t const layout = candidates.ndim() == 2 && dense_width == n && !ranks && !weights
                                       ? ballot_layout_t::complete_k
                                       : ballot_layout_t::ragged_k;
    interrupt_scope_t const interrupts;
    openmp_team_t const team(std::thread::hardware_concurrency());
    switch (score_type) {
    case score_type_t::saturated64_k:
        return tally_typed<saturated<std::uint64_t>, std::uint64_t>(ballots, layout, backend, relation);
    case score_type_t::uint64_k: return tally_typed<std::uint64_t, std::uint64_t>(ballots, layout, backend, relation);
    case score_type_t::uint32_k: return tally_typed<std::uint32_t, std::uint32_t>(ballots, layout, backend, relation);
    case score_type_t::uint16_k:
        switch (backend) {
        case backend_t::cpu_k: return tally_typed<std::uint16_t, std::uint16_t>(ballots, layout, backend, relation);
        // Devices have no 16-bit atomics, so a GPU tally counts in 32 bits and narrows on copy-out.
        case backend_t::gpu_k: return tally_typed<std::uint32_t, std::uint16_t>(ballots, layout, backend, relation);
        }
        break;
    case score_type_t::auto_k: break;
    }
    throw std::invalid_argument("score_type must be auto, saturated64, uint64, uint32, or uint16");
}

static py::object tally_ballots_py(py::handle candidates, py::object offsets, py::object num_candidates,
                                   py::object ranks, py::object weights, py::object unranked,
                                   std::string_view relation_name, std::string_view score_name,
                                   std::string_view backend_name) {
    return tally_ragged_py(candidates, offsets, num_candidates, ranks, weights, unranked,
                           relation_from_name(relation_name), score_name, backend_name);
}

static py::object tally_pairwise_relations_py(py::handle candidates, py::object offsets, py::object num_candidates,
                                              py::object ranks, py::object weights, py::object unranked,
                                              std::string_view score_name, std::string_view backend_name) {
    return tally_ragged_py(candidates, offsets, num_candidates, ranks, weights, unranked, pairwise_relation_t::all_k,
                           score_name, backend_name);
}

#pragma endregion Tally

/** Execution targets visible to this build and runtime, without launching a kernel. */
static std::vector<std::string> available_backends() {
    std::vector<std::string> backends {"cpu"};
    if (gpu_device_count_() > 0) backends.emplace_back("gpu");
    return backends;
}

PYBIND11_MODULE(scalingelections_cuda, m) {
    m.def("available_backends", &available_backends);
    m.def("tally_ballots", &tally_ballots_py, //
          py::arg("candidates"), py::kw_only(), py::arg("offsets") = py::none(), py::arg("num_candidates") = py::none(),
          py::arg("ranks") = py::none(), py::arg("weights") = py::none(), py::arg("unranked") = "unknown",
          py::arg("relation") = "preference", py::arg("score_type") = "auto", py::arg("backend") = "cpu");
    m.def("tally_pairwise_relations", &tally_pairwise_relations_py, //
          py::arg("candidates"), py::kw_only(), py::arg("offsets") = py::none(), py::arg("num_candidates") = py::none(),
          py::arg("ranks") = py::none(), py::arg("weights") = py::none(), py::arg("unranked") = "unknown",
          py::arg("score_type") = "auto", py::arg("backend") = "cpu");
    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("compute_kemeny_ranking", &compute_kemeny_ranking_py, //
          py::arg("preferences"), py::kw_only(),                //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("_compute_kemeny_costs", &compute_kemeny_costs_py, //
          py::arg("preferences"), py::kw_only(),             //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("compute_split_cycle_winners", &compute_split_cycle_winners, //
          py::arg("preferences"), py::kw_only(),                       //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
}

#pragma endregion Python bindings
