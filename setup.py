import os
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc, get_python_lib

import pybind11
import numpy as np


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
                obj = self.compiler.compile(
                    [source], output_dir=self.build_temp, extra_postargs=["-fPIC"]
                )
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "scaling_elections.o")],
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
                obj = self.compiler.compile(
                    [source], output_dir=self.build_temp, extra_postargs=["-fPIC"]
                )
                objects.extend(obj)

        # Link all object files
        self.compiler.link_shared_object(
            objects + [os.path.join(self.build_temp, "scaling_elections.o")],
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
        opt_flags = [
            "-fPIC",
            "-fopenmp",
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

        # Link all object files
        self.compiler.link_shared_object(
            objects,
            self.get_ext_fullpath(ext.name),
            libraries=[lib for lib in ext.libraries if not lib.startswith("cu")],
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
        output_file = os.path.join(output_dir, "scaling_elections.o")

        # Let's try inferring the compute capability from the GPU
        # Kepler: -arch=sm_30
        # Turing: -arch=sm_75
        # Ampere: -arch=sm_86
        # Ada: -arch=sm_89
        # Hopper: -arch=sm_90
        # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
        arch_code = "90"
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit

            device = cuda.Device(0)  # Get the default device
            major, minor = device.compute_capability()
            arch_code = f"{major}{minor}"
        except ImportError:
            pass

        cmd = (
            f"nvcc -ccbin g++ -c {source} -o {output_file} -std=c++17 "
            f"-gencode arch=compute_{arch_code},code=sm_{arch_code} "  # produce real SASS binary
            f"-gencode arch=compute_{arch_code},code=compute_{arch_code} "  # plus PTX for future devices
            f"-Xcompiler -fPIC {include_dirs} -O3 -g"
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
        output_file = os.path.join(output_dir, "scaling_elections.o")

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
            result = subprocess.run(
                ["rocminfo"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Parse rocminfo output to find gfx architecture
                for line in result.stdout.split('\n'):
                    if 'Name:' in line and 'gfx' in line:
                        # Extract gfx code (e.g., gfx90a)
                        parts = line.split()
                        for part in parts:
                            if part.startswith('gfx'):
                                arch_code = part.strip()
                                break
                        if arch_code:
                            break
        except Exception:
            pass

        # Fall back to common architectures if detection fails
        if not arch_code:
            # Try environment variable
            arch_code = os.environ.get('HIP_ARCHITECTURES', 'gfx90a,gfx906,gfx908')

        # Build the hipcc command
        # HIP can often compile CUDA code directly with --cuda-gpu-arch for compatibility
        cmd = (
            f"hipcc -c {source} -o {output_file} -std=c++17 "
            f"--offload-arch={arch_code} "
            f"-fPIC {include_dirs} -O3 -g "
            f"-D__HIP_PLATFORM_AMD__"
        )

        if os.system(cmd) != 0:
            raise RuntimeError(f"hipcc compilation of {source} failed")


__version__ = "0.2.0"

long_description = ""
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

# Get Python library path dynamically
import sys
import sysconfig

# Try to get the actual Python library directory
# Use sysconfig which is more reliable than sys.prefix for finding libraries
python_lib_dir = sysconfig.get_config_var('LIBDIR')
if not python_lib_dir or not os.path.exists(os.path.join(python_lib_dir, f"libpython{sys.version_info.major}.{sys.version_info.minor}.so")):
    # Fallback: resolve the real path of the Python executable and use its lib directory
    python_executable = os.path.realpath(sys.executable)
    python_base_dir = os.path.dirname(os.path.dirname(python_executable))
    python_lib_dir = os.path.join(python_base_dir, "lib")

python_lib_name = f"python{sys.version_info.major}.{sys.version_info.minor}"


# Detect CUDA availability
def has_cuda():
    """Check if CUDA is available."""
    # Check for nvcc
    if os.system("which nvcc > /dev/null 2>&1") != 0:
        return False
    # Check for CUDA headers
    if not os.path.exists("/usr/local/cuda/include/cuda.h"):
        return False
    return True


# Detect ROCm/HIP availability
def has_rocm():
    """Check if ROCm/HIP is available."""
    # Check for hipcc
    if os.system("which hipcc > /dev/null 2>&1") != 0:
        return False
    # Check for HIP headers in common ROCm locations
    hip_paths = [
        "/opt/rocm/include/hip/hip_runtime.h",
        "/opt/rocm/hip/include/hip/hip_runtime.h"
    ]
    if not any(os.path.exists(path) for path in hip_paths):
        return False
    return True


cuda_available = has_cuda()
rocm_available = has_rocm()

# Build extension based on GPU availability
if cuda_available:
    print("Building with CUDA support")
    ext_modules = [
        Extension(
            "scaling_elections",
            ["scaling_elections.cu"],
            include_dirs=[
                pybind11.get_include(),
                np.get_include(),
                get_python_inc(),
                "/usr/local/cuda/include/",
            ],
            library_dirs=[
                "/usr/local/cuda/lib64",
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
            "scaling_elections",
            ["scaling_elections.cu"],  # HIP can compile CUDA-style code
            include_dirs=[
                pybind11.get_include(),
                np.get_include(),
                get_python_inc(),
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
                "gomp",      # OpenMP
                python_lib_name,
            ],
            extra_link_args=[
                f"-Wl,-rpath,{python_lib_dir}",
                f"-Wl,-rpath,/opt/rocm/lib",
                "-fopenmp",
            ],
            language="c++",
        ),
    ]
else:
    print("Building CPU-only (OpenMP) - No GPU support (CUDA/ROCm) available")
    ext_modules = [
        Extension(
            "scaling_elections",
            ["scaling_elections.cu"],  # Will be compiled as C++ with GCC
            include_dirs=[
                pybind11.get_include(),
                np.get_include(),
                get_python_inc(),
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
    url="https://github.com/ashvardanian/ScalingElections",
    description="GPU-accelerated Schulze voting algorithm",
    long_description=long_description,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.9",
)
