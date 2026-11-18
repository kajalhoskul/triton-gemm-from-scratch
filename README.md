# Triton GEMM from scratch

**Author:** [Kajal Changappa](https://github.com/kajalhoskul)

[![CI](https://github.com/kajalhoskul/triton-gemm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kajalhoskul/triton-gemm-from-scratch/actions/workflows/ci.yml)

Writing a matrix-multiply (GEMM) kernel in [Triton](https://triton-lang.org), starting from the simplest correct version and optimizing it step by step, with every step measured against cuBLAS (`torch.matmul`) on a real NVIDIA GPU.

The goal is not just a fast kernel. It is a record of **why** each change helps, backed by numbers.

## Plan

| Step | Change | Status |
|---|---|---|
| 1 | Benchmark harness and cuBLAS reference | done |
| 2 | Naive Triton kernel, correct and tested | done |
| 3 | Tiling with data reuse, then autotuned block sizes | done |
| 4 | FP16 Tensor Cores, grouped program ordering, roofline analysis | done |
| 5 | Results, charts and write-up | planned |

## Step 2: the naive kernel

[`gemm/naive.py`](gemm/naive.py) launches one Triton program per element of C. Each program walks along one row of A and one column of B in chunks of `BLOCK_K`, multiplies them element by element, and sums in FP32.

It is correct for any shape and layout: masks handle sizes that are not multiples of `BLOCK_K`, and strides handle transposed inputs. But it is slow by design:

- **No reuse.** Each value of A is read from global memory N times, and each value of B M times. For a 4096 x 4096 GEMM the kernel asks for about 2,700 times more data than the minimum of reading A and B once and writing C once (the GPU caches absorb part of it).
- **No Tensor Cores.** It multiplies with plain vector arithmetic instead of `tl.dot`.

Step 3 fixes the reuse problem, and step 4 brings in Tensor Cores.

## Step 3: tiling and autotuning

[`gemm/tiled.py`](gemm/tiled.py) gives each program a `BLOCK_M x BLOCK_N` **tile** of C instead of a single value. For every step along K it loads one tile of A and one tile of B and multiplies them with `tl.dot`:

| Per K step | Naive (step 2) | Tiled, 64 x 64 x 32 (step 3) |
|---|---|---|
| Values loaded | 2 · 32 = 64 | 64·32 + 32·64 = 4,096 |
| Multiply-adds | 32 | 64 · 64 · 32 = 131,072 |
| **Multiply-adds per value loaded** | **0.5** | **32** |

That ratio, the *arithmetic intensity*, is the whole point. A GPU can do far more arithmetic per second than it can fetch values from memory, so a kernel that does 0.5 multiply-adds per value spends nearly all its time waiting for memory. Reusing each loaded value across a whole row or column of the tile raises the ratio 64-fold.

**Why autotune?** Bigger tiles mean more reuse, but they need more registers and shared memory, and they leave fewer tiles to spread across the GPU's processors. The best balance depends on the GPU and the matrix shape. `autotuned_matmul` lets Triton time [8 tile configurations](gemm/tiled.py) on the first call for each shape and keep the fastest. The configurations are sized for a T4's 64 KB of shared memory; any that don't fit on a given GPU are skipped.

## Step 4: Tensor Cores, grouped ordering and the roofline

[`gemm/fast.py`](gemm/fast.py) keeps the step 3 tiling and changes two things.

**1. FP16 inputs, FP32 accumulator.** When both tiles passed to `tl.dot` are FP16, Triton compiles it to Tensor Core instructions. On a T4 their peak is **65 TFLOPS**, against **8.1 TFLOPS** for ordinary FP32 maths. Sums are still kept in FP32, so error doesn't build up with K: a test adds 4,096 ones and checks the answer is exactly 4,096, where an FP16 running sum would get stuck at 2,048. FP16 also halves the bytes per value, so tiles cost half the memory traffic and half the shared memory.

**2. Grouped tile ordering.** GPUs start programs roughly in order of their id, and programs running together share the L2 cache. Here is which program computes which tile:

```
row-major (step 3)       grouped, GROUP_M = 2 (step 4)
 0  1  2  3                0  2  4  6
 4  5  6  7                1  3  5  7
 8  9 10 11                8 10 12 14
12 13 14 15                9 11 13 15
```

In row-major order, programs 0 to 3 need 1 row of A tiles and **4** columns of B tiles. Grouped, they need 2 rows and 2 columns, so fewer distinct tiles for the same work, and more cache hits. [`gemm/ordering.py`](gemm/ordering.py) models this mapping in plain Python, and its tests prove that every tile is computed exactly once for every grid shape and group size.

**3. Roofline analysis.** [`gemm/roofline.py`](gemm/roofline.py) answers the question "how good is this number?". A kernel is capped either by the GPU's peak FLOP rate or by memory bandwidth × arithmetic intensity, whichever is lower. For each benchmark result it reports the ceiling, whether the shape is compute-bound or memory-bound, and the percentage of the ceiling reached. A square 4096 FP16 GEMM, for example, does 1,365 FLOPs per byte at minimum traffic. That puts its memory ceiling far above 65 TFLOPS, so it's compute-bound, and any shortfall is in the kernel, not the memory system.

```bash
python -m gemm.roofline results/t4_fp16.csv
```

## How performance is measured

- **Throughput** is reported in TFLOPS: a GEMM of `M x K` by `K x N` does `2·M·N·K` floating-point operations.
- **Timing** uses CUDA events on the GPU, after warm-up runs, and reports the **median** of many repetitions so one slow run cannot skew the result.
- **Correctness** is checked against `torch.matmul` on every shape before any timing is reported, with tolerances that depend on the data type.
- **Hardware:** benchmarks run on an NVIDIA T4 (free Google Colab). The exact GPU and software versions are saved with every result.

## Results

Coming in step 5.

## Running it

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

Triton needs Linux and an NVIDIA GPU. Without one, the tests run Triton kernels on the CPU through its interpreter (`TRITON_INTERPRET=1`), which checks correctness but not speed. CI does exactly this.

To benchmark on a GPU, open the notebook in Colab with a T4 runtime and run all cells:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kajalhoskul/triton-gemm-from-scratch/blob/main/notebooks/benchmark_colab.ipynb)

Or from a shell on any machine with an NVIDIA GPU:

```bash
python -m gemm.bench --kernels torch --dtype float16 --sizes 512 1024 2048 4096
```

## Layout

```
gemm/reference.py   torch.matmul (cuBLAS) baseline
gemm/bench.py       correctness check, timing, TFLOPS, CSV output
gemm/naive.py       step 2: one Triton program per output element
gemm/tiled.py       step 3: one program per tile, tl.dot, autotuning
gemm/fast.py        step 4: FP16 Tensor Cores and grouped tile ordering
gemm/ordering.py    plain-Python model of the tile ordering
gemm/roofline.py    roofline model: attainable TFLOPS and % of it reached
tests/              harness, Triton smoke test, kernel correctness tests
notebooks/          Colab notebook for GPU benchmarks
results/            measured results, with hardware and versions
```

## Testing

Every test, what it checks and why it exists is listed in [docs/TEST_CASES.md](docs/TEST_CASES.md).

## Author

Built by **Kajal Changappa** ([@kajalhoskul](https://github.com/kajalhoskul)).

## License

MIT © 2026 Kajal Changappa
