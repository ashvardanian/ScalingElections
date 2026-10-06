/**
 *  @brief CUDA-accelerated Schulze voting algorithm implementation.
 *  @file scalingelections.cu
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
#include <memory> // `std::make_unique`
#include <ranges> // `std::views::iota`
#include <span>   // `std::span`

#include <pybind11/numpy.h> // `array_t`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "types.cuh"
#include "ballots.cuh"
#include "schulze.cuh"
#include "kemeny.cuh"

/** Resolves the execution target shared by every operation. */
inline backend_t backend_from_name(std::string_view name) {
    if (name == "cpu") return backend_t::cpu_k;
    if (name == "gpu") {
#if defined(SCALING_ELECTIONS_WITH_CUDA)
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
            throw std::runtime_error("No GPU devices available");
        return backend_t::gpu_k;
#else
        throw std::runtime_error("This build has no GPU support compiled in");
#endif
    }
    throw std::invalid_argument("Backend must be one of: cpu, gpu");
}

inline score_type_t score_type_from_name(std::string_view name) {
    if (name == "auto") return score_type_t::auto_k;
    if (name == "uint16") return score_type_t::uint16_k;
    if (name == "uint32") return score_type_t::uint32_k;
    if (name == "uint64") return score_type_t::uint64_k;
    if (name == "saturated64") return score_type_t::saturated64_k;
    throw std::invalid_argument("score_type must be auto, uint16, uint32, uint64, or saturated64");
}

/** Stores the interrupt signal status. */
volatile std::sig_atomic_t global_signal_status = 0;

void signal_handler(int signal) { global_signal_status = signal; }

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

#endif

#pragma region Python bindings

static std::pair<py::array, std::uint64_t> integer_source(py::handle values) {
    auto array =
        py::isinstance<py::array>(values)
            ? py::reinterpret_borrow<py::array>(values)
            : py::module_::import("numpy").attr("asarray")(values, py::arg("dtype") = "object").cast<py::array>();
    auto const kind = array.dtype().kind();
    if (kind != 'O' && ((kind != 'u' && kind != 'i') || array.itemsize() > 8))
        throw py::type_error("Entries must be integers representable by UInt64");
    std::uint64_t maximum = 0;
    if (kind != 'O' && array.size()) {
        if (kind == 'i' && array.attr("min")().cast<std::int64_t>() < 0)
            throw std::overflow_error("Entries must be nonnegative");
        maximum = array.attr("max")().cast<std::uint64_t>();
    }
    return {std::move(array), maximum};
}

template <typename count_type_>
static py::array_t<count_type_> integer_array(py::handle values) {
    using count_t = count_type_;
    auto const [source, maximum] = integer_source(values);
    if (source.dtype().kind() == 'O') {
        py::array_t<count_t> result(std::vector<py::ssize_t>(source.shape(), source.shape() + source.ndim()));
        auto output = result.mutable_data();
        auto const numpy_bool = py::module_::import("numpy").attr("bool_");
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

static py::array counts_array(py::handle values) {
    auto const [source, maximum] = integer_source(values);
    if (source.dtype().kind() == 'O') return integer_array<std::uint64_t>(source);
    if (maximum <= std::numeric_limits<std::uint32_t>::max() &&
        !(source.dtype().kind() == 'u' && source.itemsize() == sizeof(std::uint64_t)))
        return py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast>::ensure(source);
    return py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>::ensure(source);
}

static candidate_index_t matrix_size(py::array const& preferences) {
    if (preferences.ndim() != 2 || preferences.shape(0) != preferences.shape(1) || preferences.shape(0) < 1)
        throw std::invalid_argument("Preferences must be a nonempty square matrix");
    auto const n = static_cast<std::size_t>(preferences.shape(0));
    if (n > std::numeric_limits<candidate_index_t>::max() - tile_size_k)
        throw std::overflow_error("Candidate count exceeds the index range");
    checked_product(checked_product(n, n), preferences.itemsize());
    return static_cast<candidate_index_t>(n);
}

template <tally_arithmetic arithmetic_type_, typename stored_count_type_>
static py::array strongest_paths_typed(strided_matrix<stored_count_type_ const> preferences, backend_t backend,
                                       seed_graph_t seed, score_type_t score_type) {
    using arithmetic_t = arithmetic_type_;
    using stored_count_t = stored_count_type_;
    auto const n = static_cast<candidate_index_t>(preferences.extent(0));
    auto const preferences_view = preferences;
    for (candidate_index_t row = 0; row < n; ++row)
        for (candidate_index_t column = 0; column < n; ++column) {
            stored_count_t const forward = preferences_view(row, column);
            stored_count_t const backward = preferences_view(column, row);
            if (score_type == score_type_t::saturated64_k && forward == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("Schulze input reaches the overflow sentinel");
            if (row == column) continue;
            stored_count_t const edge = seed == seed_graph_t::positive_margins_k
                                            ? (forward > backward ? forward - backward : 0)
                                            : forward;
            if (edge > std::numeric_limits<arithmetic_t>::max())
                throw std::overflow_error("Schulze edge exceeds the selected arithmetic type");
        }
    py::array_t<arithmetic_t> result({n, n});
    auto const result_view = square_view(result.mutable_data(), n, n);
#if defined(SCALING_ELECTIONS_WITH_CUDA)
    if (backend == backend_t::gpu_k) {
        candidate_index_t const stride = divide_round_up(n, tile_size_k) * tile_size_k;
        managed_vector<arithmetic_t> graph(checked_product(stride, stride));
        if (cudaMemset(graph.data(), 0, checked_product(graph.size(), sizeof(arithmetic_t))) != cudaSuccess)
            throw std::runtime_error("Failed to clear device memory");
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("Failed to clear device memory");
        compute_strongest_paths_gpu<tile_size_k>(preferences_view, square_view(graph.data(), stride, stride), seed,
                                                 score_type);
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("CUDA solver did not complete");
        if (cudaMemcpy2D(result.mutable_data(), std::size_t(n) * sizeof(arithmetic_t), graph.data(),
                         std::size_t(stride) * sizeof(arithmetic_t), std::size_t(n) * sizeof(arithmetic_t), n,
                         cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Failed to copy paths from device");
        return result;
    }
#else
    if (backend == backend_t::gpu_k) throw std::runtime_error("This build has no GPU support compiled in");
#endif
#if defined(SCALING_ELECTIONS_WITH_OPENMP)
    openmp_team_t const team(std::thread::hardware_concurrency());
#endif
    if (n % tile_size_k == 0)
        compute_strongest_paths_tiled_cpu<tile_size_k, tile_march_t::fast_k>(preferences_view, result_view,
                                                                             &global_signal_status, seed);
    else
        compute_strongest_paths_tiled_cpu<tile_size_k, tile_march_t::checked_k>(preferences_view, result_view,
                                                                                &global_signal_status, seed);
    return result;
}

template <typename stored_count_type_>
static py::array strongest_paths_over(strided_matrix<stored_count_type_ const> preferences, backend_t backend,
                                      seed_graph_t seed, score_type_t score_type) {
    if (score_type == score_type_t::auto_k && sizeof(stored_count_type_) > sizeof(std::uint32_t)) {
        auto const n = preferences.extent(0);
        auto const preferences_view = preferences;
        for (candidate_index_t row = 0; row < n; ++row)
            for (candidate_index_t column = 0; column < n; ++column)
                if (row != column && preferences_view(row, column) > std::numeric_limits<std::uint32_t>::max())
                    return strongest_paths_typed<std::uint64_t>(preferences, backend, seed, score_type_t::uint64_k);
    }
    switch (score_type) {
    case score_type_t::uint16_k: return strongest_paths_typed<std::uint16_t>(preferences, backend, seed, score_type);
    case score_type_t::auto_k:
    case score_type_t::uint32_k: return strongest_paths_typed<std::uint32_t>(preferences, backend, seed, score_type);
    case score_type_t::uint64_k:
    case score_type_t::saturated64_k:
        return strongest_paths_typed<std::uint64_t>(preferences, backend, seed, score_type);
    }
    throw std::invalid_argument("Invalid score type");
}

static py::array strongest_paths_over(py::handle values, std::string_view backend_name, seed_graph_t seed,
                                      std::string_view score_name) {
    auto preferences = counts_array(values);
    auto const n = matrix_size(preferences);
    auto const backend = backend_from_name(backend_name);
    auto const score_type = score_type_from_name(score_name);
    if (preferences.itemsize() == sizeof(std::uint32_t))
        return strongest_paths_over(square_view(static_cast<std::uint32_t const*>(preferences.data()), n, n), backend,
                                    seed, score_type);
    return strongest_paths_over(square_view(static_cast<std::uint64_t const*>(preferences.data()), n, n), backend, seed,
                                score_type);
}

static py::array compute_strongest_paths(py::handle preferences, std::string_view backend,
                                         std::string_view score_type) {
    return strongest_paths_over(preferences, backend, seed_graph_t::winning_votes_k, score_type);
}

template <typename stored_count_type_>
static std::vector<candidate_index_t> split_cycle_over(strided_matrix<stored_count_type_ const> preferences,
                                                       backend_t backend, score_type_t score_type) {
    auto paths = py::array_t<stored_count_type_>(
        strongest_paths_over(preferences, backend, seed_graph_t::positive_margins_k, score_type));
    auto const n = preferences.extent(0);
    return select_split_cycle_winners(preferences, square_view(paths.data(), n, n));
}

static std::vector<candidate_index_t> compute_split_cycle_winners(py::handle values, std::string_view backend_name,
                                                                  std::string_view score_name) {
    auto preferences = counts_array(values);
    auto const n = matrix_size(preferences);
    auto const backend = backend_from_name(backend_name);
    auto const score_type = score_type_from_name(score_name);
    if (preferences.itemsize() == sizeof(std::uint32_t))
        return split_cycle_over(square_view(static_cast<std::uint32_t const*>(preferences.data()), n, n), backend,
                                score_type);
    return split_cycle_over(square_view(static_cast<std::uint64_t const*>(preferences.data()), n, n), backend,
                            score_type);
}

static py::tuple compute_kemeny_ranking_py(py::handle values, std::string_view backend_name,
                                           std::string_view score_name) {
    auto preferences = counts_array(values);
    auto const n = matrix_size(preferences);
    auto const backend = backend_from_name(backend_name);
    auto const score_type = score_type_from_name(score_name);
    auto solution = preferences.itemsize() == sizeof(std::uint32_t)
                        ? compute_kemeny_ranking(square_view<std::uint32_t const, candidate_index_t>(
                                                     static_cast<std::uint32_t const*>(preferences.data()), n, n),
                                                 backend, score_type)
                        : compute_kemeny_ranking(square_view<std::uint64_t const, candidate_index_t>(
                                                     static_cast<std::uint64_t const*>(preferences.data()), n, n),
                                                 backend, score_type);
    return py::make_tuple(solution.ranking, solution.score, solution.winners,
                          solution.multiplicity == ranking_multiplicity_t::unique_k ? "unique" : "multiple");
}

template <typename costs_type_>
static py::array kemeny_costs_to_python(costs_type_ costs) {
    using costs_t = costs_type_;
    using arithmetic_t = typename costs_t::value_type;
    using stored_count_t =
        std::conditional_t<std::is_same_v<arithmetic_t, saturated<std::uint64_t>>, std::uint64_t, arithmetic_t>;
    if (costs.back() == std::numeric_limits<arithmetic_t>::max())
        throw std::overflow_error("The optimal Kemeny score exceeds the representable range");
    auto owner = std::make_unique<costs_t>(std::move(costs));
    py::capsule lifetime(owner.get(), [](void* data) { delete static_cast<costs_t*>(data); });
    auto const retained = owner.release();
    return py::array(py::dtype::of<stored_count_t>(), {retained->size()}, {sizeof(stored_count_t)}, retained->data(),
                     lifetime);
}

template <typename stored_count_type_>
static py::array kemeny_costs_over(strided_matrix<stored_count_type_ const, candidate_index_t> preferences,
                                   backend_t backend, score_type_t score_type) {
    if (score_type == score_type_t::auto_k) {
        auto const bound = kemeny_score_bound_(preferences);
        score_type = bound < std::numeric_limits<std::uint32_t>::max()   ? score_type_t::uint32_k
                     : bound < std::numeric_limits<std::uint64_t>::max() ? score_type_t::uint64_k
                                                                         : score_type_t::saturated64_k;
    }
    auto run = [&](auto score) -> py::array {
        using arithmetic_t = decltype(score);
        kemeny_require_supported_width_(preferences.extent(0));
        kemeny_require_score_range_<arithmetic_t>(preferences);
        kemeny_sums<arithmetic_t> const sums(preferences);
        if (backend == backend_t::cpu_k)
            return kemeny_costs_to_python(compute_kemeny_costs_cpu<arithmetic_t>(preferences, sums));
#if defined(SCALING_ELECTIONS_WITH_CUDA)
        return kemeny_costs_to_python(compute_kemeny_costs_gpu<arithmetic_t>(preferences, sums));
#else
        throw std::runtime_error("This build has no GPU support compiled in");
#endif
    };
    switch (score_type) {
    case score_type_t::uint16_k: return run(std::uint16_t {});
    case score_type_t::uint32_k: return run(std::uint32_t {});
    case score_type_t::uint64_k: return run(std::uint64_t {});
    case score_type_t::saturated64_k: return run(saturated<std::uint64_t> {0});
    default: throw std::invalid_argument("Invalid score type");
    }
}

static py::array compute_kemeny_costs_py(py::handle values, std::string_view backend_name,
                                         std::string_view score_name) {
    auto preferences = counts_array(values);
    auto const n = matrix_size(preferences);
    auto const backend = backend_from_name(backend_name);
    auto const score_type = score_type_from_name(score_name);
    if (preferences.itemsize() == sizeof(std::uint32_t))
        return kemeny_costs_over(square_view<std::uint32_t const, candidate_index_t>(
                                     static_cast<std::uint32_t const*>(preferences.data()), n, n),
                                 backend, score_type);
    return kemeny_costs_over(square_view<std::uint64_t const, candidate_index_t>(
                                 static_cast<std::uint64_t const*>(preferences.data()), n, n),
                             backend, score_type);
}

template <tally_arithmetic arithmetic_type_, typename output_count_type_ = arithmetic_type_>
static py::object tally_over(ragged_ballots ballots, backend_t backend, pairwise_relation_t relation) {
    using arithmetic_t = arithmetic_type_;
    using output_count_t = output_count_type_;
    auto const n = ballots.num_candidates;
    std::size_t const planes = relation == pairwise_relation_t::all_k ? 3 : 1;
    std::size_t const rows = checked_product(n, planes);
    std::size_t const cells = checked_product(rows, n);
    checked_product(cells, sizeof(arithmetic_t));
    py::array_t<output_count_t> result({rows, std::size_t(n)});
    auto const output = std::span(result.mutable_data(), cells);
    if constexpr (std::is_same_v<arithmetic_t, output_count_t>) {
        std::ranges::fill(output, output_count_t {0});
        py::gil_scoped_release release;
        tally_ballots(ballots, strided_view(output.data(), rows, std::size_t(n), std::size_t(n)), backend, relation);
    }
    else {
        std::vector<arithmetic_t> counts(cells, arithmetic_t {0});
        {
            py::gil_scoped_release release;
            tally_ballots(ballots, strided_view(counts.data(), rows, std::size_t(n), std::size_t(n)), backend,
                          relation);
        }
        std::ranges::transform(counts, output.begin(),
                               [](arithmetic_t count) { return static_cast<output_count_t>(count); });
    }
    if constexpr (sizeof(output_count_t) == sizeof(std::uint64_t))
        if (std::ranges::find(output, std::numeric_limits<output_count_t>::max()) != output.end())
            throw std::overflow_error("Tally reaches the overflow sentinel");
    if (relation != pairwise_relation_t::all_k) return result;
    py::tuple matrices(planes);
    for (std::size_t plane = 0; plane < planes; ++plane)
        matrices[plane] = py::array_t<output_count_t>({n, n}, {sizeof(output_count_t) * n, sizeof(output_count_t)},
                                                      output.data() + plane * n * n, result);
    return matrices;
}

static py::object tally_ballots_py(py::handle values, py::object offsets_arg, py::object num_candidates_arg,
                                   py::object ranks_arg, py::object weights_arg, py::object unranked_arg,
                                   pairwise_relation_t relation, std::string_view score_name,
                                   std::string_view backend_name) {
    auto candidates = integer_array<candidate_index_t>(values);
    auto const candidate_ids = std::span(candidates.data(), static_cast<std::size_t>(candidates.size()));
    auto const backend = backend_from_name(backend_name);
    auto score_type = score_type_from_name(score_name);
    auto parse_unranked = [](py::handle value) {
        auto const name = py::cast<std::string>(value);
        if (name == "unknown") return unranked_t::unknown_k;
        if (name == "worse") return unranked_t::worse_k;
        throw std::invalid_argument("unranked must be unknown or worse");
    };
    auto unranked = unranked_t::unknown_k;
    std::vector<unranked_t> policies;
    if (py::isinstance<py::str>(unranked_arg)) unranked = parse_unranked(unranked_arg);
    else
        for (auto value : unranked_arg) policies.push_back(parse_unranked(value));

    std::optional<py::array_t<std::uint64_t>> offsets;
    std::span<std::uint64_t const> ballot_offsets;
    std::size_t num_ballots = 0;
    std::size_t n = 0;
    if (!num_candidates_arg.is_none()) {
        auto const count = integer_array<ballot_offset_t>(num_candidates_arg);
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
        offsets = integer_array<std::uint64_t>(offsets_arg);
        ballot_offsets = {offsets->data(), static_cast<std::size_t>(offsets->size())};
        if (offsets->ndim() != 1 || ballot_offsets.empty() || ballot_offsets.front() != 0 ||
            ballot_offsets.back() != candidate_ids.size())
            throw std::invalid_argument("Offsets must span the candidate entries");
        if (!std::ranges::is_sorted(ballot_offsets)) throw std::invalid_argument("Offsets must be nondecreasing");
        num_ballots = ballot_offsets.size() - 1;
        if (num_candidates_arg.is_none()) throw std::invalid_argument("CSR ballots require num_candidates");
    }
    if (n < 1 || n > std::numeric_limits<candidate_index_t>::max())
        throw std::invalid_argument("Candidate count exceeds the supported index range");
    if (!py::isinstance<py::str>(unranked_arg) && policies.size() != num_ballots)
        throw std::invalid_argument("Unranked policies must match the ballot count");
    std::size_t const cells = checked_product(n, n);
    checked_product(cells, sizeof(std::uint64_t));

    std::optional<py::array_t<candidate_index_t>> ranks;
    std::optional<py::array_t<std::uint64_t>> weights;
    std::span<candidate_index_t const> rank_values;
    std::span<std::uint64_t const> voter_weights;
    if (!ranks_arg.is_none()) {
        ranks = integer_array<candidate_index_t>(ranks_arg);
        if (ranks->size() != candidates.size() ||
            (ranks->ndim() != 1 && (candidates.ndim() != 2 || ranks->ndim() != 2)) ||
            (ranks->ndim() == 2 && (ranks->shape(0) != candidates.shape(0) || ranks->shape(1) != candidates.shape(1))))
            throw std::invalid_argument("Ranks must match the candidate entries");
        rank_values = {ranks->data(), static_cast<std::size_t>(ranks->size())};
    }
    if (!weights_arg.is_none()) {
        weights = integer_array<std::uint64_t>(weights_arg);
        if (weights->ndim() != 1 || static_cast<std::size_t>(weights->size()) != num_ballots)
            throw std::invalid_argument("Weights must have one entry per voter");
        voter_weights = {weights->data(), static_cast<std::size_t>(weights->size())};
    }
    std::vector<std::size_t> seen(n, std::numeric_limits<std::size_t>::max());
    for (std::size_t ballot = 0; ballot < num_ballots; ++ballot) {
        auto const first = ballot_offsets.empty() ? ballot * dense_width : ballot_offsets[ballot];
        auto const length = ballot_offsets.empty() ? dense_width : ballot_offsets[ballot + 1] - first;
        for (auto candidate : candidate_ids.subspan(first, length)) {
            if (candidate >= n || seen[candidate] == ballot)
                throw std::invalid_argument("Candidate IDs must be in range and distinct within each ballot");
            seen[candidate] = ballot;
        }
    }
    auto const bound = voter_weights.empty()
                           ? saturated<std::uint64_t>(num_ballots)
                           : std::accumulate(voter_weights.begin(), voter_weights.end(), saturated<std::uint64_t> {0});
    if (score_type == score_type_t::auto_k) {
        score_type = score_type_t::uint32_k;
        if (bound.value > std::numeric_limits<std::uint32_t>::max()) score_type = score_type_t::uint64_k;
        if (bound.value == std::numeric_limits<std::uint64_t>::max()) score_type = score_type_t::saturated64_k;
    }
    std::uint64_t limit = std::numeric_limits<std::uint64_t>::max() - 1;
    switch (score_type) {
    case score_type_t::uint16_k: limit = std::numeric_limits<std::uint16_t>::max(); break;
    case score_type_t::uint32_k: limit = std::numeric_limits<std::uint32_t>::max(); break;
    case score_type_t::saturated64_k: limit = std::numeric_limits<std::uint64_t>::max(); break;
    default: break;
    }
    if (bound.value > limit) throw std::overflow_error("Tally bound exceeds the selected arithmetic type");
    if (candidates.ndim() == 2 && dense_width == n && !ranks && !weights && score_type == score_type_t::uint32_k &&
        relation == pairwise_relation_t::preference_k) {
        py::array_t<std::uint32_t> result({n, n});
        auto const output = std::span(result.mutable_data(), cells);
        std::ranges::fill(output, 0);
        if (num_ballots) {
            py::gil_scoped_release release;
            tally_ballots(strided_view<candidate_index_t const>(candidate_ids.data(), num_ballots, n, n),
                          square_view(output.data(), n, n), backend);
        }
        return result;
    }
    if (!offsets) {
        checked_product(num_ballots + 1, sizeof(std::uint64_t));
        offsets.emplace(num_ballots + 1);
        auto const output = std::span(offsets->mutable_data(), num_ballots + 1);
        std::ranges::transform(std::views::iota(std::size_t {0}, num_ballots + 1), output.begin(),
                               [dense_width](auto ballot) { return ballot * dense_width; });
        ballot_offsets = output;
    }
    ragged_ballots const ballots {candidate_ids.data(),
                                  ballot_offsets.data(),
                                  rank_values.data(),
                                  voter_weights.data(),
                                  num_ballots,
                                  static_cast<candidate_index_t>(n),
                                  unranked,
                                  policies.empty() ? nullptr : policies.data()};
    switch (score_type) {
    case score_type_t::uint16_k: return tally_over<std::uint32_t, std::uint16_t>(ballots, backend, relation);
    case score_type_t::uint32_k: return tally_over<std::uint32_t>(ballots, backend, relation);
    case score_type_t::uint64_k: return tally_over<std::uint64_t>(ballots, backend, relation);
    case score_type_t::saturated64_k:
        return tally_over<saturated<std::uint64_t>, std::uint64_t>(ballots, backend, relation);
    default: throw std::invalid_argument("Invalid score type");
    }
}

/** Execution targets visible to this build and runtime, without launching a kernel. */
static std::vector<std::string> available_backends() {
    std::vector<std::string> backends {"cpu"};
#if defined(SCALING_ELECTIONS_WITH_CUDA)
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0) backends.emplace_back("gpu");
#endif
    return backends;
}

PYBIND11_MODULE(scalingelections_cuda, m) {

    std::signal(SIGINT, signal_handler);

    m.def("available_backends", &available_backends);
    m.def(
        "tally_ballots",
        [](py::handle candidates, py::object offsets, py::object num_candidates, py::object ranks, py::object weights,
           py::object unranked, std::string_view relation_name, std::string_view score_type, std::string_view backend) {
            auto relation = pairwise_relation_t::preference_k;
            if (relation_name == "indifference") relation = pairwise_relation_t::indifference_k;
            else if (relation_name == "unknown") relation = pairwise_relation_t::unknown_k;
            else if (relation_name != "preference")
                throw std::invalid_argument("relation must be preference, indifference, or unknown");
            return tally_ballots_py(candidates, offsets, num_candidates, ranks, weights, unranked, relation, score_type,
                                    backend);
        },
        py::arg("candidates"), py::kw_only(), py::arg("offsets") = py::none(), py::arg("num_candidates") = py::none(),
        py::arg("ranks") = py::none(), py::arg("weights") = py::none(), py::arg("unranked") = "unknown",
        py::arg("relation") = "preference", py::arg("score_type") = "auto", py::arg("backend") = "cpu");
    m.def(
        "tally_pairwise_relations",
        [](py::handle candidates, py::object offsets, py::object num_candidates, py::object ranks, py::object weights,
           py::object unranked, std::string_view score_type, std::string_view backend) {
            return tally_ballots_py(candidates, offsets, num_candidates, ranks, weights, unranked,
                                    pairwise_relation_t::all_k, score_type, backend);
        },
        py::arg("candidates"), py::kw_only(), py::arg("offsets") = py::none(), py::arg("num_candidates") = py::none(),
        py::arg("ranks") = py::none(), py::arg("weights") = py::none(), py::arg("unranked") = "unknown",
        py::arg("score_type") = "auto", py::arg("backend") = "cpu");
    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("compute_kemeny_ranking", &compute_kemeny_ranking_py, //
          py::arg("preferences"), py::kw_only(),                //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("_compute_kemeny_costs", &compute_kemeny_costs_py, py::arg("preferences"), py::kw_only(),
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("compute_split_cycle_winners", &compute_split_cycle_winners, //
          py::arg("preferences"), py::kw_only(),                       //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
}

#pragma endregion Python bindings
