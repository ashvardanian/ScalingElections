# Scaling Elections with GPUs

![Scaling Elections Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/scaling-democracy.jpg?raw=true)

This repository implements the Schulze voting algorithm using CUDA for hardware acceleration.
That algorithm is often used by Pirate Parties and open-source foundations, and it's a good example of a combinatorial problem that can be parallelized efficiently on GPUs.
It's built as a single `scaling_elections.cu` CUDA file, wrapped with PyBind11, and compiled __without__ CMake directly from the `setup.py`.

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
uv sync --extra dev                 # Build locally and install dependencies
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

A typical benchmark output comparing serial Numba code to 16 Intel Ice Lake cores to SXM Nvidia H100 GPU would be:

```sh
> Generating 4,096 random voter rankings with 4,096 candidates
> Generated voter rankings, proceeding with 16 threads
> Numba: 11.2169 secs, 6,126,425,774.83 cells^3/sec
> CUDA: 0.3146 secs, 218,437,101,103.83 cells^3/sec
> CUDA with TMA: 0.2969 secs, 231,448,250,952.52 cells^3/sec
> OpenMP: 24.7729 secs, 2,773,975,923.94 cells^3/sec
> Serial: 58.8089 secs, 1,168,522,106.72 cells^3/sec
```

CUDA outperforms the baseline JIT-compiled parallel kernel by a factor of __37.78x__.

---

40 core CPU uses ~270 Watts, so 10 cores use ~67.5 Watts.
Our SXM Nvidia H100 has a ~700 Watt TDP, but consumes only 360 under such load, so 5x more power-hungry, meaning the CUDA implementation is up to 7x more power-efficient than Numba on that Intel CPU.
As the matrix grows, the GPU utilization improves and the experimentally observed throughput fits a sub-cubic curve.
Comparing to Arm-based CPUs and native SIMD-accelerated code would be more fair.
Repeating the experiment with 192-core AWS Graviton 4 chips, the timings with tile-size 32 are:

| Candidates | Numba on `c8g` | OpenMP on `c8g` | OpenMP + NEON on `c8g` | CUDA on `h100` | Mojo on `h100` |
| :--------- | -------------: | --------------: | ---------------------: | -------------: | -------------: |
| 2'048      |         1.14 s |          0.35 s |                 0.16 s |                |                |
| 4'096      |         1.84 s |          1.02 s |                 0.35 s |                |                |
| 8'192      |         7.49 s |          5.50 s |                 4.64 s |         1.98 s |                |
| 16'384     |        38.04 s |         24.67 s |                24.20 s |         9.53 s |                |
| 32'768     |       302.85 s |        246.85 s |               179.82 s |        42.90 s |                |

Comparing the numbers, we are still looking at a roughly 4x speedup of CUDA for the largest matrix size tested for a comparable power consumption and hardware rental cost.

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
