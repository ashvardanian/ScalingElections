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

/** Resolves the name the Python layer passes, throwing on anything unrecognized. */
inline backend_t backend_from_name(std::string_view name) {
    if (name == "cpu_openmp") return backend_t::cpu_openmp_k;
    if (name == "gpu_serial") return backend_t::gpu_serial_k;
    if (name == "gpu_hopper") return backend_t::gpu_hopper_k;
    throw std::invalid_argument("Backend must be one of: cpu_openmp, gpu_serial, gpu_hopper");
}

/** Resolves the Kemeny backend name the Python layer passes, throwing on anything unrecognized. */
inline tally_backend_t tally_backend_from_name(std::string_view name) {
    if (name == "auto") return tally_backend_t::automatic_k;
    if (name == "cpu_openmp") return tally_backend_t::cpu_openmp_k;
    if (name == "gpu_privatized") return tally_backend_t::gpu_privatized_k;
    throw std::invalid_argument("Tally backend must be one of: auto, cpu_openmp, gpu_privatized");
}

inline kemeny_backend_t kemeny_backend_from_name(std::string_view name) {
    if (name == "auto") return kemeny_backend_t::automatic_k;
    if (name == "cpu_openmp") return kemeny_backend_t::cpu_openmp_k;
    if (name == "gpu_layered") return kemeny_backend_t::gpu_layered_k;
    throw std::invalid_argument("Kemeny backend must be one of: auto, cpu_openmp, gpu_layered");
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
 *  @param[in] backend_name One of `cpu_openmp`, `gpu_serial`, or `gpu_hopper`.
 *  @return A NumPy array containing the strongest paths matrix.
 *
 *  @note A backend the build or the device cannot serve raises rather than downgrading.
 */
static py::array_t<votes_count_t> strongest_paths_over_(        //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    std::string_view backend_name, seed_graph_t seed) {

    backend_t const backend = backend_from_name(backend_name);

    auto buffer = preferences.request();
    if (buffer.ndim != 2) throw std::runtime_error("Number of dimensions must be two");
    if (buffer.shape[0] != buffer.shape[1]) throw std::runtime_error("Preferences matrix must be square");
    auto preferences_ptr = reinterpret_cast<votes_count_t*>(buffer.ptr);
    auto num_candidates = static_cast<candidate_index_t>(buffer.shape[0]);
    auto row_stride = static_cast<candidate_index_t>(buffer.strides[0] / sizeof(votes_count_t));
    const_matrix_t const preferences_view = square_view(preferences_ptr, num_candidates, row_stride);

    // Allocate NumPy array for the result
    auto result = py::array_t<votes_count_t>({num_candidates, num_candidates});
    auto result_buf = result.request();
    auto result_ptr = reinterpret_cast<votes_count_t*>(result_buf.ptr);
    auto result_row_stride = static_cast<candidate_index_t>(result_buf.strides[0] / sizeof(votes_count_t));
    if (result_row_stride != num_candidates) throw std::runtime_error("Result matrix must be contiguous");

#if defined(SCALING_ELECTIONS_WITH_CUDA)

    if (backend != backend_t::cpu_openmp_k) {

        // Rounding the matrix up to a whole number of tiles keeps the kernels free of tail
        // checks: the padding is zero, which is the identity of the max-min semiring.
        candidate_index_t const graph_stride = (num_candidates + tile_size_k - 1) / tile_size_k * tile_size_k;
        managed_vector<votes_count_t> graph(static_cast<std::size_t>(graph_stride) * graph_stride);
        cudaError_t error = cudaMemset(graph.data(), 0, graph.size() * sizeof(votes_count_t));
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("Failed to clear device memory");

        compute_strongest_paths_cuda<tile_size_k>(preferences_view,
                                                  square_view(graph.data(), graph_stride, graph_stride), backend, seed);

        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("CUDA operations did not complete successfully");

        // Copy the leading sub-block back, dropping the padding.
        error = cudaMemcpy2D(result_ptr, num_candidates * sizeof(votes_count_t), graph.data(),
                             graph_stride * sizeof(votes_count_t), num_candidates * sizeof(votes_count_t),
                             num_candidates, cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) throw std::runtime_error("Failed to copy data from device to host");

        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) throw std::runtime_error("CUDA transfers did not complete successfully");
        return result;
    }

#else

    if (backend != backend_t::cpu_openmp_k) throw std::runtime_error("This build has no GPU support compiled in");

#endif // defined(SCALING_ELECTIONS_WITH_CUDA)

#if defined(SCALING_ELECTIONS_WITH_OPENMP)
    openmp_team_t const team(std::thread::hardware_concurrency());
#endif

    // A tile wider than the electorate is not an error: `checked_k` zero-fills the tail, and
    // zero is the identity of the max-min semiring, so the padding can never win a comparison.
    matrix_t const result_view = square_view(result_ptr, num_candidates, num_candidates);
    if (num_candidates % tile_size_k == 0)
        compute_strongest_paths_openmp<tile_size_k, tile_march_t::fast_k>( //
            preferences_view, result_view, &global_signal_status, seed);
    else
        compute_strongest_paths_openmp<tile_size_k, tile_march_t::checked_k>( //
            preferences_view, result_view, &global_signal_status, seed);
    return result;
}

/**
 *  @brief Widest paths over winning votes, which is the variant Schulze runs on here.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu_openmp`, `gpu_serial`, or `gpu_hopper`.
 *  @return A NumPy array containing the strongest paths matrix.
 */
static py::array_t<votes_count_t> compute_strongest_paths(      //
    py::array_t<votes_count_t, py::array::c_style> preferences, //
    std::string_view backend_name) {
    return strongest_paths_over_(preferences, backend_name, seed_graph_t::winning_votes_k);
}

/**
 *  @brief The Split Cycle winning set, which is every candidate nobody defeats.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `cpu_openmp`, `gpu_serial`, or `gpu_hopper`.
 *  @return The undefeated candidates, in increasing order.
 *
 *  The same max-min kernel serves both methods; only the graph it closes over differs, which is
 *  what being a C2 rule buys.
 */
static std::vector<candidate_index_t> compute_split_cycle_winners( //
    py::array_t<votes_count_t, py::array::c_style> preferences,    //
    std::string_view backend_name) {

    py::array_t<votes_count_t> const margin_paths = strongest_paths_over_(preferences, backend_name,
                                                                          seed_graph_t::positive_margins_k);

    py::buffer_info const preferences_buffer = preferences.request();
    py::buffer_info const paths_buffer = margin_paths.request();
    auto const num_candidates = static_cast<candidate_index_t>(preferences_buffer.shape[0]);
    auto const row_stride = static_cast<candidate_index_t>(preferences_buffer.strides[0] / sizeof(votes_count_t));

    py::gil_scoped_release release;
    return split_cycle_winners(
        square_view(reinterpret_cast<votes_count_t const*>(preferences_buffer.ptr), num_candidates, row_stride),
        square_view(reinterpret_cast<votes_count_t const*>(paths_buffer.ptr), num_candidates, num_candidates));
}

/**
 *  @brief Prints what each visible CUDA device is, through Python's own stdout.
 *
 *  @note Raises on a build without CUDA, rather than reporting an empty device list.
 */
#if defined(SCALING_ELECTIONS_WITH_CUDA)
static void log_gpus() {
    int device_count;
    cudaDeviceProp device_properties;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess) throw std::runtime_error("Failed to get device count");
    for (int index = 0; index < device_count; index++) {
        error = cudaGetDeviceProperties(&device_properties, index);
        if (error != cudaSuccess) throw std::runtime_error("Failed to get device properties");
        py::print(py::str("Device {}: {}").format(index, device_properties.name));
        py::print(py::str("\tSMs: {}").format(device_properties.multiProcessorCount));
        py::print(py::str("\tGlobal mem: {:.2f}GB")
                      .format(static_cast<float>(device_properties.totalGlobalMem) / (1024 * 1024 * 1024)));
        py::print(py::str("\tCUDA Cap: {}.{}").format(device_properties.major, device_properties.minor));
    }
}
#else
static void log_gpus() { throw std::runtime_error("No CUDA devices available"); }
#endif

/**
 *  @brief Computes the exact Kemeny-Young consensus ranking and its disagreement score.
 *
 *  @param[in] preferences The preferences matrix.
 *  @param[in] backend_name One of `auto`, `cpu_openmp`, or `gpu_layered`.
 *  @return A tuple of the ranking, best first, and the disagreement it achieves.
 */
static py::tuple compute_kemeny_ranking(py::array_t<votes_count_t, py::array::c_style> const& preferences,
                                        std::string_view backend_name) {
    kemeny_backend_t const backend = kemeny_backend_from_name(backend_name);

    py::buffer_info buffer = preferences.request();
    if (buffer.ndim != 2 || buffer.shape[0] != buffer.shape[1])
        throw std::runtime_error("Preferences must be a square matrix");
    candidate_index_t const num_candidates = static_cast<candidate_index_t>(buffer.shape[0]);

    kemeny_solution_t const solution = kemeny_solve(static_cast<votes_count_t const*>(buffer.ptr), num_candidates,
                                                    backend);
    return py::make_tuple(solution.ranking, solution.score);
}

/**
 *  @brief Folds one chunk of complete rankings into a pairwise preference matrix.
 *
 *  @param[in] rankings A two-dimensional array of complete rankings, best candidate first.
 *  @param[in] backend_name One of `cpu_openmp` or `gpu_privatized`.
 *  @return The square matrix counting, for each ordered pair, the ballots preferring the first.
 *
 *  Every ballot must rank every candidate, so a caller with partial ballots completes them first.
 */
static py::array_t<votes_count_t> tally_ballots_py(py::array_t<candidate_index_t, py::array::c_style> const& rankings,
                                                   std::string const& backend_name) {

    tally_backend_t const backend = tally_backend_from_name(backend_name);

    py::buffer_info buffer = rankings.request();
    if (buffer.ndim != 2) throw std::runtime_error("Rankings must be a two-dimensional array");
    std::size_t const num_ballots = static_cast<std::size_t>(buffer.shape[0]);
    candidate_index_t const num_candidates = static_cast<candidate_index_t>(buffer.shape[1]);
    if (num_candidates < 1) throw std::runtime_error("Every ballot must rank at least one candidate");

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
    m.def("log_gpus", &log_gpus);
    m.def("tally_ballots", &tally_ballots_py, //
          py::arg("rankings"), py::kw_only(), //
          py::arg("backend") = "auto");
    m.def("compute_kemeny_ranking", &compute_kemeny_ranking, //
          py::arg("preferences"), py::kw_only(),             //
          py::arg("backend") = "auto");
    m.def("compute_strongest_paths", &compute_strongest_paths, //
          py::arg("preferences"), py::kw_only(),               //
          py::arg("backend") = "cpu_openmp");
    m.def("compute_split_cycle_winners", &compute_split_cycle_winners, //
          py::arg("preferences"), py::kw_only(),                       //
          py::arg("backend") = "cpu_openmp");
}

#pragma endregion Python bindings
