import os
import platform
import shutil
import sys
import sysconfig

import pybind11
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


def cuda_home():
    """Toolkit root of the `nvcc` on PATH, so headers and libraries match the compiler."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    return os.environ.get("CUDA_HOME", "/usr/local/cuda")


CUDA_HOME = cuda_home()


# Detect CUDA availability
def has_cuda():
    """Check if CUDA is available."""
    # Check for nvcc
    if os.system("which nvcc > /dev/null 2>&1") != 0:
        return False
    # Check for CUDA headers
    return os.path.exists(os.path.join(CUDA_HOME, "include", "cuda.h"))


# Detect ROCm/HIP availability
def has_rocm():
    """Check if ROCm/HIP is available."""
    # Check for hipcc
    if os.system("which hipcc > /dev/null 2>&1") != 0:
        return False
    # Check for HIP headers in common ROCm locations
    hip_paths = [
        "/opt/rocm/include/hip/hip_runtime.h",
        "/opt/rocm/hip/include/hip/hip_runtime.h",
    ]
    return any(os.path.exists(path) for path in hip_paths)


# Ampere, Ada, Hopper, Blackwell datacenter, Blackwell consumer. Newest last: it supplies the PTX.
DEFAULT_CUDA_ARCHS = ("80", "89", "90", "100", "120")


def detect_cuda_archs():
    """Compute capabilities to emit, overridable via `SCALING_ELECTIONS_CUDA_ARCH`."""
    override = os.environ.get("SCALING_ELECTIONS_CUDA_ARCH", "")
    requested = [code.strip() for code in override.split(",") if code.strip()]
    return requested or list(DEFAULT_CUDA_ARCHS)


HEADERS = ["types.cuh", "ballots.cuh", "schulze.cuh", "kemeny.cuh"]


def cccl_include() -> list:
    """A host-only build needs libcu++'s `mdspan` where the standard library has none of its own."""
    roots = [
        os.path.join(CUDA_HOME, "include"),
        os.path.join(CUDA_HOME, "include", "cccl"),
    ]
    return [root for root in roots if os.path.exists(os.path.join(root, "cuda", "std", "mdspan"))][:1]


CCCL_INCLUDE = cccl_include()

cuda_available = has_cuda()
rocm_available = has_rocm()
is_macos = platform.system() == "Darwin"
python_include = sysconfig.get_paths()["include"]


class BuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append(".cu")
        nvcc_available = self.is_nvcc_available()
        hipcc_available = self.is_hipcc_available()

        for ext in self.extensions:
            if any(source.endswith(".cu") for source in ext.sources):
                if nvcc_available:
                    self.build_cuda_extension(ext)
                elif hipcc_available:
                    self.build_hip_extension(ext)
                else:
                    self.build_gcc_extension(ext)
            else:
                super().build_extension(ext)

    def is_nvcc_available(self):
        return os.system("which nvcc > /dev/null 2>&1") == 0

    def is_hipcc_available(self):
        return os.system("which hipcc > /dev/null 2>&1") == 0

    def build_cuda_extension(self, ext):
        # Compile CUDA source files
        for source in ext.sources:
            if source.endswith(".cu"):
                self.compile_cuda(source)

        # Compile non-CUDA source files
        objects = []
        for source in ext.sources:
            if not source.endswith(".cu"):
                obj = self.compiler.compile([source], output_dir=self.build_temp, extra_postargs=["-fPIC"])
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "scalingelections.o")],
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def build_hip_extension(self, ext):
        # Compile HIP source files using hipcc
        for source in ext.sources:
            if source.endswith(".cu"):
                self.compile_hip(source)

        # Compile non-HIP source files
        objects = []
        for source in ext.sources:
            if not source.endswith(".cu"):
                obj = self.compiler.compile([source], output_dir=self.build_temp, extra_postargs=["-fPIC"])
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "scalingelections.o")],
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def build_gcc_extension(self, ext):
        # Compile all source files with GCC, including treating .cu files as .cpp files
        objects = []
        # Aggressive optimization flags for CPU performance
        if is_macos:
            # macOS with clang doesn't support some GCC flags
            opt_flags = [
                "-std=c++20",  # C++20 standard, required for `cuda::std::mdspan`
                "-fPIC",  # Position Independent Code
                "-O3",  # Maximum optimization
                "-ffast-math",  # Aggressive floating-point optimizations
                "-march=native",  # Use all available CPU instructions
                "-funroll-loops",  # Loop unrolling
            ]
        else:
            # Linux with GCC
            opt_flags = [
                "-std=c++20",  # C++20 standard, required for `cuda::std::mdspan`
                "-fPIC",  # Position Independent Code
                "-fopenmp",  # OpenMP support
                "-O3",  # Maximum optimization
                "-ffast-math",  # Aggressive floating-point optimizations
                "-march=native",  # Use all available CPU instructions (AVX, AVX2, AVX-512, etc.)
                "-mtune=native",  # Tune for the specific CPU
                "-funroll-loops",  # Loop unrolling
                "-ftree-vectorize",  # Enable vectorization
                "-fopt-info-vec-optimized",  # Report successful vectorizations
            ]
        for source in ext.sources:
            if source.endswith(".cu"):
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_preargs=["-x", "c++"],
                    extra_postargs=opt_flags,
                    include_dirs=ext.include_dirs,
                )
            else:
                obj = self.compiler.compile(
                    [source],
                    output_dir=self.build_temp,
                    extra_postargs=opt_flags,
                    include_dirs=ext.include_dirs,
                )
            objects.extend(obj)

        # Link all object files with libraries from extension config
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=ext.extra_link_args,
            target_lang=ext.language,
        )

    def compile_cuda(self, source):
        # Compile CUDA source file using nvcc
        ext = self.extensions[0]
        output_dir = self.build_temp
        os.makedirs(output_dir, exist_ok=True)
        include_dirs = self.compiler.include_dirs + ext.include_dirs
        include_dirs = " ".join(f"-I{dir}" for dir in include_dirs)
        output_file = os.path.join(output_dir, "scalingelections.o")

        arch_codes = detect_cuda_archs()
        # SASS for every target, PTX only for the newest so older toolkits can still JIT forward.
        gencodes = " ".join(f"-gencode arch=compute_{arch},code=sm_{arch}" for arch in arch_codes)
        gencodes += f" -gencode arch=compute_{arch_codes[-1]},code=compute_{arch_codes[-1]}"

        cmd = (
            f"nvcc -ccbin g++ -c {source} -o {output_file} -std=c++20 "
            f"{gencodes} "
            f"-Xcompiler -fPIC,-fopenmp,-march=native {include_dirs} -O3 -g"
        )

        if os.system(cmd) != 0:
            raise RuntimeError(f"nvcc compilation of {source} failed")

    def compile_hip(self, source):
        # Compile HIP source file using hipcc
        ext = self.extensions[0]
        output_dir = self.build_temp
        os.makedirs(output_dir, exist_ok=True)
        include_dirs = self.compiler.include_dirs + ext.include_dirs
        include_dirs = " ".join(f"-I{dir}" for dir in include_dirs)
        output_file = os.path.join(output_dir, "scalingelections.o")

        # Detect AMD GPU architecture
        # Common AMD architectures:
        # gfx900: Vega 10 (MI25)
        # gfx906: Vega 20 (MI50, MI60)
        # gfx908: CDNA1 (MI100)
        # gfx90a: CDNA2 (MI200 series)
        # gfx940/gfx941/gfx942: CDNA3 (MI300 series)
        # gfx1030: RDNA2 (RX 6000 series)
        # gfx1100: RDNA3 (RX 7000 series)

        # Try to detect the GPU architecture
        arch_code = None
        try:
            import subprocess

            result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse rocminfo output to find gfx architecture
                for line in result.stdout.split("\n"):
                    if "Name:" in line and "gfx" in line:
                        # Extract gfx code (e.g., gfx90a)
                        parts = line.split()
                        for part in parts:
                            if part.startswith("gfx"):
                                arch_code = part.strip()
                                break
                        if arch_code:
                            break
        except Exception:
            pass

        # Fall back to common architectures if detection fails
        if not arch_code:
            # Try environment variable
            arch_code = os.environ.get("HIP_ARCHITECTURES", "gfx90a,gfx906,gfx908")

        # Build the hipcc command
        # HIP can often compile CUDA code directly with --cuda-gpu-arch for compatibility
        cmd = (
            f"hipcc -c {source} -o {output_file} -std=c++20 "
            f"--offload-arch={arch_code} "
            f"-fPIC -fopenmp {include_dirs} -O3 -g "
            f"-D__HIP_PLATFORM_AMD__"
        )

        if os.system(cmd) != 0:
            raise RuntimeError(f"hipcc compilation of {source} failed")


__version__ = "0.2.0"

long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Try to get the actual Python library directory
# Use sysconfig which is more reliable than sys.prefix for finding libraries
python_lib_dir = sysconfig.get_config_var("LIBDIR")
if not python_lib_dir or not os.path.exists(
    os.path.join(python_lib_dir, f"libpython{sys.version_info.major}.{sys.version_info.minor}.so")
):
    # Fallback: resolve the real path of the Python executable and use its lib directory
    python_executable = os.path.realpath(sys.executable)
    python_base_dir = os.path.dirname(os.path.dirname(python_executable))
    python_lib_dir = os.path.join(python_base_dir, "lib")

python_lib_name = f"python{sys.version_info.major}.{sys.version_info.minor}"


# Build extension based on GPU availability and platform
if cuda_available:
    print("Building with CUDA support")
    ext_modules = [
        Extension(
            "scalingelections_cuda",
            ["scalingelections.cu"],
            depends=HEADERS,
            include_dirs=[
                pybind11.get_include(),
                python_include,
                os.path.join(CUDA_HOME, "include"),
            ],
            library_dirs=[
                os.path.join(CUDA_HOME, "lib64"),
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/wsl/lib",
                python_lib_dir,
            ],
            libraries=[
                "cudart",
                "cuda",
                "cublas",
                "gomp",  # OpenMP
                python_lib_name,
            ],
            extra_link_args=[
                f"-Wl,-rpath,{python_lib_dir}",
                "-fopenmp",
            ],
            language="c++",
        ),
    ]
elif rocm_available:
    print("Building with ROCm/HIP support")
    ext_modules = [
        Extension(
            "scalingelections_cuda",
            ["scalingelections.cu"],  # HIP can compile CUDA-style code
            depends=HEADERS,
            include_dirs=[
                pybind11.get_include(),
                python_include,
                "/opt/rocm/include",
                "/opt/rocm/hip/include",
            ],
            library_dirs=[
                "/opt/rocm/lib",
                "/opt/rocm/hip/lib",
                "/usr/lib/x86_64-linux-gnu",
                python_lib_dir,
            ],
            libraries=[
                "amdhip64",  # HIP runtime (required)
                "gomp",  # OpenMP
                python_lib_name,
            ],
            extra_link_args=[
                f"-Wl,-rpath,{python_lib_dir}",
                "-Wl,-rpath,/opt/rocm/lib",
                "-fopenmp",
            ],
            language="c++",
        ),
    ]
else:
    # CPU-only build
    if is_macos:
        print("Building CPU-only (macOS, no OpenMP) - No GPU support available")
        ext_modules = [
            Extension(
                "scalingelections_cuda",
                ["scalingelections.cu"],  # Will be compiled as C++ with clang
                depends=HEADERS,
                include_dirs=[
                    pybind11.get_include(),
                    python_include,
                    *CCCL_INCLUDE,
                ],
                extra_link_args=["-undefined", "dynamic_lookup"],
                language="c++",
            ),
        ]
    else:
        print("Building CPU-only (OpenMP) - No GPU support (CUDA/ROCm) available")
        ext_modules = [
            Extension(
                "scalingelections_cuda",
                ["scalingelections.cu"],  # Will be compiled as C++ with GCC
                depends=HEADERS,
                include_dirs=[
                    pybind11.get_include(),
                    python_include,
                    *CCCL_INCLUDE,
                ],
                library_dirs=[
                    "/usr/lib/x86_64-linux-gnu",
                    python_lib_dir,
                ],
                libraries=[
                    "gomp",  # OpenMP
                    python_lib_name,
                ],
                extra_link_args=[
                    f"-Wl,-rpath,{python_lib_dir}",
                    "-fopenmp",
                ],
                language="c++",
            ),
        ]


setup(
    name="ScalingElections",
    version=__version__,
    author="Ash Vardanian",
    author_email="1983160+ashvardanian@users.noreply.github.com",
    url="https://ashvardanian.com/posts/scaling-elections",
    project_urls={"Repository": "https://github.com/ashvardanian/ScalingElections"},
    description="ScalingElections: Condorcet Voting at GPU Speed — Schulze as semiring matrix multiplication, Kemeny-Young as exact NP-hard search",
    long_description=long_description,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.12",
)
