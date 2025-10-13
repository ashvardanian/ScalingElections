# Scaling Elections with GPUs

![Scaling Elections Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/scaling-democracy.jpg?raw=true)

This repository implements tiled parallel adaptations of the Schulze voting algorithm with hardware acceleration across CPUs and GPUs in Mojo and CUDA C++ wrapped into Python.
That algorithm is often used by Pirate Parties and open-source foundations, and it's a good example of a combinatorial problem that can be parallelized by changing evaluation order.

- The Mojo implementation is packed into a single file `scaling_elections.mojo`, containing both CPU and GPU implementations, and the benchmarking code.
- The other implementation is a sandwich of `scaling_elections.cu` CUDA C++ code wrapped with PyBind11, and `scaling_elections.py` packing the Python benchmarking code and Numba reference kernels.

Not a single line of CMake is used in this repository!
The entire native library build is packed into `setup.py` for Python, and `pixi` takes care of the Mojo build.

## Usage

Both Python and Mojo implementations are included.
Both support the same CLI arguments:

- `--no-serial`: Skip serial baseline
- `--num-candidates N`: Number of candidates (default: 128)
- `--num-voters N`: Number of voters (default: 2000)
- `--run-cpu`: Run CPU implementations
- `--run-gpu`: Run GPU implementation
- `--cpu-tile-size N`: CPU tile size (default: 16)
- `--gpu-tile-size N`: GPU tile size (default: 32)
- `--help, -h`: Show help message

### Python

Pull:

```sh
git clone https://github.com/ashvardanian/ScalingElections.git
cd ScalingElections
```

Build the environment and run with `uv`:

```sh
uv venv -p python3.12               # Pick a recent Python version
uv sync --extra cpu --extra gpu     # Build locally and install dependencies
uv run scaling_elections.py         # Run the default problem size
uv run scaling_elections.py --num-candidates 4096 --num-voters 4096 --run-cpu --run-gpu
```

Alternatively, with your local environment:

```sh
pip install -e . --force-reinstall  # Build locally and install dependencies
python scaling_elections.py         # Run the default problem size
```

### Mojo

This repository also includes a pure Mojo implementation in `scaling_elections.mojo`.
To install and run it, use `pixi`:

```sh
pixi install
pixi run mojo scaling_elections.mojo --help
pixi run mojo scaling_elections.mojo --num-candidates 4096 --num-voters 4096 --run-cpu --run-gpu
```

Or compile and run as a standalone binary:

```sh
pixi run mojo build scaling_elections.mojo -o schulze
./schulze --num-candidates 4096 --run-cpu --run-gpu
```

## Links

- [Blogpost](https://ashvardanian.com/posts/scaling-democracy/)
- [Schulze voting method description](https://en.wikipedia.org/wiki/Schulze_method)
- [On traversal order for Floyd Warshall algorithm](https://moorejs.github.io/APSP-in-parallel/)
- [CUDA + Python project template](https://github.com/ashvardanian/cuda-python-starter-kit)

## Throughput

| Candidates | Numba, `384vCPU` | Mojo, `384vCPU` |    CUDA, `h100` |    Mojo, `h100` |
| :--------- | ---------------: | --------------: | --------------: | --------------: |
| 2'048      |   34.4 Gcells³/s |  48.5 Gcells³/s | 182.7 Gcells³/s | 153.4 Gcells³/s |
| 4'096      |   86.8 Gcells³/s |  50.3 Gcells³/s | 264.1 Gcells³/s | 232.6 Gcells³/s |
| 8'192      |   74.6 Gcells³/s |  71.6 Gcells³/s | 495.3 Gcells³/s | 408.0 Gcells³/s |
| 16'384     |   76.7 Gcells³/s |  78.5 Gcells³/s | 600.7 Gcells³/s | 635.3 Gcells³/s |
| 32'768     |  101.4 Gcells³/s |  82.3 Gcells³/s | 921.4 Gcells³/s | 893.7 Gcells³/s |

---

With NVIDIA Nsight Compute CLI we can dissect the kernels and see that there is more room for improvement:

```sh
ncu uv run scaling_elections.py --num-candidates 4096 --num-voters 4096 --gpu-tile-size 32 --run-gpu
>  void _cuda_independent<32>(unsigned int, unsigned int, unsigned int *) (128, 128, 1)x(32, 32, 1), Context 1, Stream 7, Device 0, CC 9.0
>    Section: GPU Speed Of Light Throughput
>    ----------------------- ----------- ------------
>    Metric Name             Metric Unit Metric Value
>    ----------------------- ----------- ------------
>    DRAM Frequency                  Ghz         3.20
>    SM Frequency                    Ghz         1.50
>    Elapsed Cycles                cycle       307595
>    Memory Throughput                 %        78.65
>    DRAM Throughput                   %        10.69
>    Duration                         us       205.22
>    L1/TEX Cache Throughput           %        79.90
>    L2 Cache Throughput               %        15.54
>    SM Active Cycles              cycle    302305.17
>    Compute (SM) Throughput           %        66.51
>    ----------------------- ----------- ------------
>
>    OPT   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L1 
>          bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes      
>          transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or        
>          whether there are values you can (re)compute.                                                                 
>
>    Section: Launch Statistics
>    -------------------------------- --------------- ---------------
>    Metric Name                          Metric Unit    Metric Value
>    -------------------------------- --------------- ---------------
>    Block Size                                                  1024
>    Cluster Scheduling Policy                           PolicySpread
>    Cluster Size                                                   0
>    Function Cache Configuration                     CachePreferNone
>    Grid Size                                                  16384
>    Registers Per Thread             register/thread              31
>    Shared Memory Configuration Size           Kbyte           32.77
>    Driver Shared Memory Per Block       Kbyte/block            1.02
>    Dynamic Shared Memory Per Block       byte/block               0
>    Static Shared Memory Per Block       Kbyte/block           12.29
>    # SMs                                         SM             132
>    Stack Size                                                  1024
>    Threads                                   thread        16777216
>    # TPCs                                                        66
>    Enabled TPC IDs                                              all
>    Uses Green Context                                             0
>    Waves Per SM                                               62.06
>    -------------------------------- --------------- ---------------
>
>    Section: Occupancy
>    ------------------------------- ----------- ------------
>    Metric Name                     Metric Unit Metric Value
>    ------------------------------- ----------- ------------
>    Max Active Clusters                 cluster            0
>    Max Cluster Size                      block            8
>    Overall GPU Occupancy                     %            0
>    Cluster Occupancy                         %            0
>    Block Limit Barriers                  block           32
>    Block Limit SM                        block           32
>    Block Limit Registers                 block            2
>    Block Limit Shared Mem                block            2
>    Block Limit Warps                     block            2
>    Theoretical Active Warps per SM        warp           64
>    Theoretical Occupancy                     %          100
>    Achieved Occupancy                        %        92.97
>    Achieved Active Warps Per SM           warp        59.50
>    ------------------------------- ----------- ------------
```

So feel free to fork and suggest improvements 🤗
