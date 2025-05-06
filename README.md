# Scaling Democracy with GPUs

![Scaling Democracy Thumbnail](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/scaling-democracy.jpg?raw=true)

This repository implements the Schulze voting algorithm using CUDA for hardware acceleration.
That algorithm is often used by Pirate Parties and open-source foundations, and it's a good example of a combinatorial problem that can be parallelized efficiently on GPUs.
It's built as a single `scaling_democracy.cu` CUDA file, wrapped with PyBind11, and compiled __without__ CMake directly from the `setup.py`.

## Usage

Pull:

```sh
git clone https://github.com/ashvardanian/scaling-democracy.git
cd scaling-democracy
```

Build the environment and run with `uv`:

```sh
uv venv -p python3.12 .   # Pick a recent Python version
uv sync --extra dev       # Build locally and install dependencies
uv run benchmark.py       # Run the default problem size
```

Alternatively, with your local environment:

```sh
pip install -e .         # Build locally and install dependencies
python benchmark.py      # Run the default problem size
```

Or with custom parameters:

```sh
uv run benchmark.py \
--num-candidates 1024 \
--num-voters 1024 \
--tile-size 16 \
--run-openmp --run-numba --run-serial --run-cuda
```

## Throughput

A typical benchmark output comparing serial Numba code to 20 Intel Ice Lake threads to SXM Nvidia H100 GPU would be:

```sh
Generating 128 random voter rankings with 8,192 candidates
Generated voter rankings, proceeding with 20 threads
Serial: 454.5124 seconds, 1,209,550,826.73 candidates^3/sec
Parallel: 68.4111 seconds, 8,036,063,293.27 candidates^3/sec
CUDA: 1.9802 seconds, 277,620,752,084.85 candidates^3/sec
```

CUDA outperforms the baseline JIT-compiled parallel kernel by a factor of __34.55x__.
40 core CPU uses ~270 Watts, so 10 cores use ~67.5 Watts.
Our SXM Nvidia H100 has a ~700 Watt TDP, but consumes only 360 under such load, so 5x more power-hungry, meaning the CUDA implementation is up to 7x more power-efficient than Numba on that Intel CPU.
As the matrix grows, the GPU utilization improves and the experimentally observed throughput fits a sub-cubic curve.
Comparing to Arm-based CPUs and native SIMD-accelerated code would be more fair.
Repeating the experiment with 192-core AWS Graviton 4 chips, the timings with tile-size 32 are:

| Candidates | Numba on `c8g` | OpenMP on `c8g` | OpenMP + NEON on `c8g` | CUDA on `h100` |
| :--------- | -------------: | --------------: | ---------------------: | -------------: |
| 2'048      |         1.14 s |          0.35 s |                 0.16 s |                |
| 4'096      |         1.84 s |          1.02 s |                 0.35 s |                |
| 8'192      |         7.49 s |          5.50 s |                 4.64 s |         1.98 s |
| 16'384     |        38.04 s |         24.67 s |                24.20 s |         9.53 s |
| 32'768     |       302.85 s |        246.85 s |               179.82 s |        42.90 s |

Comparing the numbers, we are still looking at a roughly 4x speedup of CUDA for the largest matrix size tested for a comparable power consumption and hardware rental cost.

## Links

- [Blogpost](https://ashvardanian.com/posts/scaling-democracy/)
- [Schulze voting method description](https://en.wikipedia.org/wiki/Schulze_method)
- [On traversal order for Floyd Warshall algorithm](https://moorejs.github.io/APSP-in-parallel/)
- [CUDA + Python project template](https://github.com/ashvardanian/cuda-python-starter-kit)
