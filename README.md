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
| 4 | FP16 Tensor Cores, grouped program ordering, profiling | planned |
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
