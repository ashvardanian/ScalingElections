"""Build the native extension with the available CUDA, HIP, or host compiler.

Package metadata lives in `pyproject.toml`; this file only picks a toolkit and drives its compiler.
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum

import pybind11
from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension


class Toolkit(StrEnum):
    """The compiler driver the native sources are built with."""

    cuda = "cuda"
    hip = "hip"
    host = "host"


@dataclass(frozen=True)
class DeviceToolkit:
    """Where one accelerator toolkit lives, what it links, and which targets it emits by default."""

    driver: str
    home_variable: str
    default_home: str
    runtime_header: str
    runtime_library: str
    architectures_variable: str
    default_architectures: tuple[str, ...]


DEVICE_TOOLKITS = {
    # Ampere, Ada, Hopper, Blackwell datacenter, Blackwell consumer. Newest last: it supplies the PTX.
    Toolkit.cuda: DeviceToolkit(
        driver="nvcc",
        home_variable="CUDA_HOME",
        default_home="/usr/local/cuda",
        runtime_header="include/cuda_runtime.h",
        runtime_library="cudart",
        architectures_variable="SCALING_ELECTIONS_CUDA_ARCH",
        default_architectures=("80", "89", "90", "100", "120"),
    ),
    # MI50, MI100, MI200, MI300, MI355X.
    Toolkit.hip: DeviceToolkit(
        driver="hipcc",
        home_variable="ROCM_PATH",
        default_home="/opt/rocm",
        runtime_header="include/hip/hip_runtime.h",
        runtime_library="amdhip64",
        architectures_variable="SCALING_ELECTIONS_HIP_ARCH",
        default_architectures=("gfx906", "gfx908", "gfx90a", "gfx942", "gfx950"),
    ),
}

HEADERS = ["types.cuh", "ballots.cuh", "schulze.cuh", "kemeny.cuh"]
HOST_FLAGS = ["-std=c++20", "-fPIC", "-O3", "-march=native", "-ffast-math", "-funroll-loops"]


def toolkit_home(toolkit: DeviceToolkit) -> str:
    """The root above the driver on `PATH`, so headers and libraries match the compiler that runs."""
    driver = shutil.which(toolkit.driver)
    if driver:
        return os.path.dirname(os.path.dirname(os.path.realpath(driver)))
    return os.environ.get(toolkit.home_variable, toolkit.default_home)


def toolkit_library_directory(home: str) -> str:
    """CUDA installs into `lib64` on Linux, ROCm into `lib`."""
    for name in ("lib64", "lib"):
        if os.path.isdir(os.path.join(home, name)):
            return os.path.join(home, name)
    return os.path.join(home, "lib")


def toolkit_architectures(toolkit: DeviceToolkit) -> list[str]:
    """Targets from the toolkit's environment variable, else its defaults; `native` names the local device."""
    requested = [code.strip() for code in os.environ.get(toolkit.architectures_variable, "").split(",")]
    return [code for code in requested if code] or list(toolkit.default_architectures)


def detect_toolkit() -> Toolkit:
    """The first accelerator toolkit whose driver and runtime headers are both installed."""
    for name, toolkit in DEVICE_TOOLKITS.items():
        home = toolkit_home(toolkit)
        if shutil.which(toolkit.driver) and os.path.exists(os.path.join(home, toolkit.runtime_header)):
            return name
    return Toolkit.host


def device_flags(name: Toolkit, architectures: list[str]) -> list[str]:
    """Code-generation and host-compiler flags for one device driver."""
    match name:
        case Toolkit.cuda:
            # SASS for every target and PTX only for the newest, so newer drivers can JIT forward.
            gencodes = [f"-gencode=arch=compute_{code},code=sm_{code}" for code in architectures]
            gencodes.append(f"-gencode=arch=compute_{architectures[-1]},code=compute_{architectures[-1]}")
            host_compiler = os.environ.get("CUDAHOSTCXX", "g++")
            return ["-ccbin", host_compiler, *gencodes, "-Xcompiler", "-fPIC,-fopenmp,-march=native"]
        case Toolkit.hip:
            return [f"--offload-arch={','.join(architectures)}", "-fPIC", "-fopenmp", "-D__HIP_PLATFORM_AMD__"]
    raise ValueError(f"{name} has no device driver")


TOOLKIT = detect_toolkit()
SYSTEM = platform.system()


class BuildExt(build_ext):
    """Compile `scalingelections.cu` with the detected toolkit and link the extension."""

    def build_extensions(self) -> None:
        self.compiler.src_extensions.append(".cu")
        os.makedirs(self.build_temp, exist_ok=True)
        for extension in self.extensions:
            match TOOLKIT:
                case Toolkit.host:
                    objects = self.compiler.compile(
                        extension.sources,
                        output_dir=self.build_temp,
                        extra_preargs=["-x", "c++"],
                        extra_postargs=extension.extra_compile_args,
                        include_dirs=extension.include_dirs,
                    )
                case Toolkit.cuda | Toolkit.hip:
                    objects = [self.compile_device(extension)]
            self.compiler.link_shared_object(
                objects,
                self.get_ext_fullpath(extension.name),
                libraries=extension.libraries,
                library_dirs=extension.library_dirs,
                runtime_library_dirs=extension.runtime_library_dirs,
                extra_postargs=extension.extra_link_args,
                target_lang=extension.language,
            )

    def compile_device(self, extension: Extension) -> str:
        """Compile the extension's one source with the toolkit's driver, returning the object path."""
        toolkit = DEVICE_TOOLKITS[TOOLKIT]
        (source,) = extension.sources
        output = os.path.join(self.build_temp, "scalingelections.o")
        includes = [f"-I{directory}" for directory in self.compiler.include_dirs + extension.include_dirs]
        command = [toolkit.driver, "-c", source, "-o", output, "-std=c++20", "-O3", "-g", *includes]
        subprocess.run([*command, *extension.extra_compile_args], check=True)
        return output


# A Python extension leaves the interpreter's symbols unresolved until import, so it never links `libpython`.
include_dirs = [pybind11.get_include()]
match TOOLKIT:
    case Toolkit.cuda | Toolkit.hip:
        toolkit = DEVICE_TOOLKITS[TOOLKIT]
        home = toolkit_home(toolkit)
        library_directory = toolkit_library_directory(home)
        native_options = dict(
            include_dirs=[*include_dirs, os.path.join(home, "include")],
            extra_compile_args=device_flags(TOOLKIT, toolkit_architectures(toolkit)),
            library_dirs=[library_directory],
            runtime_library_dirs=[library_directory],
            libraries=[toolkit.runtime_library],
            extra_link_args=["-fopenmp"],
        )
    case Toolkit.host if SYSTEM == "Darwin":
        # Apple Clang ships without OpenMP, so the macOS host build is single-team.
        native_options = dict(
            include_dirs=include_dirs,
            extra_compile_args=HOST_FLAGS,
            extra_link_args=["-undefined", "dynamic_lookup"],
        )
    case Toolkit.host:
        native_options = dict(
            include_dirs=include_dirs,
            extra_compile_args=[*HOST_FLAGS, "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )

print(f"Building the {TOOLKIT} extension on {SYSTEM}")
setup(
    ext_modules=[
        Extension("scalingelections_cuda", ["scalingelections.cu"], depends=HEADERS, language="c++", **native_options)
    ],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
