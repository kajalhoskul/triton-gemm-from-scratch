# Triton GEMM from scratch

[![CI](https://github.com/kajalhoskul/triton-gemm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kajalhoskul/triton-gemm-from-scratch/actions/workflows/ci.yml)

Writing a matrix-multiply (GEMM) kernel in [Triton](https://triton-lang.org), starting from the simplest correct version and optimizing it step by step, with every step measured against cuBLAS (`torch.matmul`) on a real NVIDIA GPU.

The goal is not just a fast kernel. It is a record of **why** each change helps, backed by numbers.

## Plan

| Step | Change | Status |
|---|---|---|
| 1 | Benchmark harness and cuBLAS reference | in progress |
| 2 | Naive Triton kernel, correct and tested | planned |
| 3 | Tiling with data reuse, then autotuned block sizes | planned |
| 4 | FP16 Tensor Cores, grouped program ordering, profiling | planned |
| 5 | Results, charts and write-up | planned |

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

To benchmark on a GPU, open [`notebooks/benchmark_colab.ipynb`](notebooks/benchmark_colab.ipynb) in Google Colab with a T4 runtime and run all cells.

## License

MIT
