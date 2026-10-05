/**
 *  @brief CUDA-accelerated Schulze voting algorithm implementation.
 *  @file scalingelections.cu
 *  @author Ash Vardanian
 *  @date July 12, 2024
 *  @see https://ashvardanian.com/posts/scaling-elections
 */
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
    if (name == "gpu") return backend_t::gpu_k;
    throw std::invalid_argument("Backend must be one of: cpu, gpu");
}

inline score_type_t score_type_from_name(std::string_view name) {
    if (name == "auto") return score_type_t::auto_k;
    if (name == "uint16") return score_type_t::uint16_k;
    if (name == "uint32") return score_type_t::uint32_k;
    if (name == "uint64") return score_type_t::uint64_k;
    throw std::invalid_argument("score_type must be auto, uint16, uint32, or uint64");
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

/**
 *  @brief Computes the strongest paths for the block-parallel Schulze voting algorithm.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu` or `gpu`.
 *  @return A NumPy array containing the strongest paths matrix.
 *
 *  @note A backend the build or the device cannot serve raises rather than downgrading.
 */
template <typename score_type_>
static py::array_t<votes_count_t> strongest_paths_over_(        //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    std::string_view backend_name, seed_graph_t seed, score_type_t score_type) {

    using score_t = score_type_;
    backend_t const backend = backend_from_name(backend_name);

    auto buffer = preferences.request();
    if (buffer.ndim != 2) throw std::invalid_argument("Number of dimensions must be two");
    if (buffer.shape[0] != buffer.shape[1]) throw std::invalid_argument("Preferences matrix must be square");
    auto preferences_ptr = reinterpret_cast<votes_count_t*>(buffer.ptr);
    auto num_candidates = static_cast<candidate_index_t>(buffer.shape[0]);
    auto row_stride = static_cast<candidate_index_t>(buffer.strides[0] / sizeof(votes_count_t));
    const_matrix_t const preferences_view = square_view(preferences_ptr, num_candidates, row_stride);

    if constexpr (sizeof(score_t) < sizeof(votes_count_t)) {
        for (candidate_index_t row = 0; row != num_candidates; ++row)
            for (candidate_index_t column = 0; column != num_candidates; ++column) {
                votes_count_t const forward = preferences_ptr[row * row_stride + column];
                votes_count_t const backward = preferences_ptr[column * row_stride + row];
                votes_count_t const edge = seed == seed_graph_t::positive_margins_k
                                               ? (forward > backward ? forward - backward : 0)
                                               : forward;
                if (edge > std::numeric_limits<score_t>::max())
                    throw std::overflow_error("Schulze edge exceeds the selected arithmetic type");
            }
    }
    auto result = py::array_t<score_t>({num_candidates, num_candidates});
    auto result_ptr = static_cast<score_t*>(result.request().ptr);

#if defined(SCALING_ELECTIONS_WITH_CUDA)

    if (backend != backend_t::cpu_k) {

        // Rounding the matrix up to a whole number of tiles keeps the kernels free of tail
        // checks: the padding is zero, which is the identity of the max-min semiring.
        candidate_index_t const graph_stride = (num_candidates + tile_size_k - 1) / tile_size_k * tile_size_k;
        managed_vector<score_t> graph(static_cast<std::size_t>(graph_stride) * graph_stride);
        cudaError_t error = cudaMemset(graph.data(), 0, graph.size() * sizeof(score_t));
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");

        compute_strongest_paths_gpu<tile_size_k>(
            preferences_view, square_view(graph.data(), graph_stride, graph_stride), seed, score_type);

        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("CUDA operations did not complete successfully");

        // Copy the leading sub-block back, dropping the padding.
        error = cudaMemcpy2D(result_ptr, num_candidates * sizeof(score_t), graph.data(), graph_stride * sizeof(score_t),
                             num_candidates * sizeof(score_t), num_candidates, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) throw std::runtime_error("Failed to copy data from device to host");

        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("CUDA transfers did not complete successfully");
        return py::array_t<votes_count_t, py::array::forcecast>::ensure(result);
    }

#else

    if (backend != backend_t::cpu_k) throw std::runtime_error("This build has no GPU support compiled in");

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#if defined(SCALING_ELECTIONS_WITH_OPENMP)
    openmp_team_t const team(std::thread::hardware_concurrency());
#endif

    // A tile wider than the electorate is not an error: `checked_k` zero-fills the tail, and
    // zero is the identity of the max-min semiring, so the padding can never win a comparison.
    strided_matrix<score_t> const result_view = square_view(result_ptr, num_candidates, num_candidates);
    if (num_candidates % tile_size_k == 0)
        compute_strongest_paths_tiled_cpu<tile_size_k, tile_march_t::fast_k>( //
            preferences_view, result_view, &global_signal_status, seed);
    else
        compute_strongest_paths_tiled_cpu<tile_size_k, tile_march_t::checked_k>( //
            preferences_view, result_view, &global_signal_status, seed);
    return py::array_t<votes_count_t, py::array::forcecast>::ensure(result);
}

static py::array_t<votes_count_t> strongest_paths_over_(py::array_t<votes_count_t, py::array::c_style> preferences,
                                                        std::string_view backend_name, seed_graph_t seed,
                                                        std::string_view score_name) {
    score_type_t const score_type = score_type_from_name(score_name);
    switch (score_type) {
    case score_type_t::uint16_k:
        return strongest_paths_over_<std::uint16_t>(preferences, backend_name, seed, score_type);
    case score_type_t::auto_k:
    case score_type_t::uint32_k:
        return strongest_paths_over_<std::uint32_t>(preferences, backend_name, seed, score_type);
    case score_type_t::uint64_k:
        return strongest_paths_over_<std::uint64_t>(preferences, backend_name, seed, score_type);
    }
    throw std::invalid_argument("Unknown score type");
}

/**
 *  @brief Widest paths over winning votes, which is the variant Schulze runs on here.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu` or `gpu`.
 *  @return A NumPy array containing the strongest paths matrix.
 */
static py::array_t<votes_count_t> compute_strongest_paths(      //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    std::string_view backend_name, std::string_view score_name) {
    return strongest_paths_over_(preferences, backend_name, seed_graph_t::winning_votes_k, score_name);
}

/**
 *  @brief The Split Cycle winning set, which is every candidate nobody defeats.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu` or `gpu`.
 *  @return The undefeated candidates, in increasing order.
 *
 *  The same max-min kernel serves both methods; only the graph it closes over differs, which is
 *  what being a C2 rule buys.
 */
static std::vector<candidate_index_t> compute_split_cycle_winners( //
    py::array_t<votes_count_t, py::array::c_style> preferences,    //
    std::string_view backend_name, std::string_view score_name) {

    py::array_t<votes_count_t> const margin_paths = strongest_paths_over_(preferences, backend_name,
                                                                          seed_graph_t::positive_margins_k, score_name);

    py::buffer_info const preferences_buffer = preferences.request();
    py::buffer_info const paths_buffer = margin_paths.request();
    auto const num_candidates = static_cast<candidate_index_t>(preferences_buffer.shape[0]);
    auto const row_stride = static_cast<candidate_index_t>(preferences_buffer.strides[0] / sizeof(votes_count_t));

    py::gil_scoped_release release;
    return select_split_cycle_winners(
        square_view(reinterpret_cast<votes_count_t const*>(preferences_buffer.ptr), num_candidates, row_stride),
        square_view(reinterpret_cast<votes_count_t const*>(paths_buffer.ptr), num_candidates, num_candidates));
}

/**
 *  @brief Computes the exact Kemeny-Young consensus ranking and its disagreement score.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu` or `gpu`.
 *  @return A tuple of the ranking, best first, and the disagreement it achieves.
 */
static py::tuple compute_kemeny_ranking_py(py::array_t<votes_count_t, py::array::c_style> const& preferences,
                                           std::string_view backend_name, std::string_view score_name) {
    backend_t const backend = backend_from_name(backend_name);
    py::buffer_info const buffer = preferences.request();
    if (buffer.ndim != 2 || buffer.shape[0] != buffer.shape[1])
        throw std::invalid_argument("Preferences must be a square matrix");
    auto const num_candidates = static_cast<candidate_index_t>(buffer.shape[0]);
    score_type_t const score_type = score_type_from_name(score_name);
    auto solution = compute_kemeny_ranking(static_cast<votes_count_t const*>(buffer.ptr), num_candidates, backend,
                                           score_type);
    return py::make_tuple(solution.ranking, solution.score);
}

/**
 *  @brief Folds one chunk of complete rankings into a pairwise preference matrix.
 *
 *  @param[in] rankings A two-dimensional array of complete rankings, best candidate first.
 *  @param[in] backend_name One of `cpu` or `gpu`.
 *  @return The square matrix counting, for each ordered pair, the ballots preferring the first.
 *
 *  Every ballot must rank every candidate, so a caller with partial ballots completes them first.
 */
static py::array_t<votes_count_t> tally_ballots_py(py::array_t<candidate_index_t, py::array::c_style> const& rankings,
                                                   std::string const& backend_name) {

    backend_t const backend = backend_from_name(backend_name);

    py::buffer_info buffer = rankings.request();
    if (buffer.ndim != 2) throw std::invalid_argument("Rankings must be a two-dimensional array");
    std::size_t const num_ballots = static_cast<std::size_t>(buffer.shape[0]);
    candidate_index_t const num_candidates = static_cast<candidate_index_t>(buffer.shape[1]);
    if (num_candidates < 1) throw std::invalid_argument("Every ballot must rank at least one candidate");

    py::array_t<votes_count_t> preferences({num_candidates, num_candidates});
    votes_count_t* preferences_ptr = static_cast<votes_count_t*>(preferences.request().ptr);
    candidate_index_t const* rankings_ptr = static_cast<candidate_index_t const*>(buffer.ptr);
    std::size_t const cells = static_cast<std::size_t>(num_candidates) * num_candidates;
    std::fill(preferences_ptr, preferences_ptr + cells, votes_count_t {0});

    ballots_t const rankings_view = strided_view<candidate_index_t const, std::size_t>(rankings_ptr, num_ballots,
                                                                                       num_candidates, num_candidates);
    matrix_t const preferences_view = square_view(preferences_ptr, num_candidates, num_candidates);
    {
        py::gil_scoped_release release;
        tally_ballots(rankings_view, preferences_view, backend);
    }
    return preferences;
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
    m.def("tally_ballots", &tally_ballots_py, //
          py::arg("rankings"), py::kw_only(), //
          py::arg("backend") = "cpu");
    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("compute_kemeny_ranking", &compute_kemeny_ranking_py, //
          py::arg("preferences"), py::kw_only(),                //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
    m.def("compute_split_cycle_winners", &compute_split_cycle_winners, //
          py::arg("preferences"), py::kw_only(),                       //
          py::arg("backend") = "cpu", py::arg("score_type") = "auto");
}

#pragma endregion Python bindings
