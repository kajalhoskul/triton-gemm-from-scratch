# Test cases

**Project:** triton-gemm-from-scratch · **Author:** Kajal Changappa · **Covers:** steps 1 and 2

27 automated test cases in 3 files. Run them with:

```bash
pytest -v
```

## Where the tests run

| Environment | Harness tests | Triton tests | What it proves |
|---|---|---|---|
| **GitHub Actions CI** (Ubuntu, CPU only, `TRITON_INTERPRET=1`) | ✅ run | ✅ run, through Triton's CPU interpreter | Every kernel computes the right answer on every push |
| **Google Colab** (NVIDIA T4) | ✅ run | ✅ run, compiled for the GPU | The kernels also compile and run correctly on real hardware |
| **macOS** (Triton cannot install) | ✅ run | ⏭️ skipped automatically | The harness works on any machine |

## How correctness is judged

Every kernel output is compared with `torch.matmul` (cuBLAS on a GPU) using **relative error**:

```
max |output − reference|  /  max |reference|
```

| Data type | Allowed error | Why this limit |
|---|---|---|
| FP32 | 1e-2 | On Ampere and newer GPUs, Triton's FP32 `tl.dot` uses TF32 (10-bit mantissa) by default. This is a deliberate speed trade-off, not a bug. |
| FP16 | 1e-2 | FP16 stores outputs with a 10-bit mantissa. |
| BF16 | 3e-2 | BF16 has only a 7-bit mantissa. |

Dividing by the largest reference value keeps the limit meaningful for every matrix size. Raw differences grow as K grows, but the relative error doesn't.

All inputs come from `make_inputs`, which uses a fixed random seed. Every run uses identical data, so a failure can always be reproduced.

---

## 1. Benchmark harness: `tests/test_bench.py` (11 cases)

These protect the measuring stick. If the harness is wrong, every speed number in the project is wrong.

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| H-01 | `test_gemm_flops_counts_multiply_and_add` | M=2, N=3, K=4 | 48 FLOPs (= 2·2·3·4) | Each output needs K multiplies **and** K adds. Counting only one halves every TFLOPS figure. |
| H-02 | `test_tflops_conversion` | 1000×1000×1000 in 1 ms | 2.0 TFLOPS | Checks the unit conversion (ms to s, FLOP to TFLOP). |
| H-03 | `test_make_inputs_shapes_dtype_and_seed` | 4×6 and 6×5, FP16, seeds 0 and 1 | Correct shapes and dtype; same seed gives identical data; different seed gives different data | Reproducible inputs make failures repeatable and comparisons fair. |
| H-04 | `test_check_close_accepts_exact_output` | Output identical to the reference | Passes, with error exactly 0 | The checker must never reject a correct kernel. |
| H-05 | `test_check_close_rejects_wrong_output` | One element off by 10× the largest value | Fails, with error > 1 | A single wrong element must be caught, not averaged away. |
| H-06 | `test_check_close_rejects_wrong_shape` | 4×4 output against a 4×5 reference | Fails | A kernel returning the wrong shape is wrong, whatever the values. |
| H-07 | `test_check_close_allows_small_rounding_error` | Output scaled by (1 + 1e-4) | Passes | Normal floating-point rounding must not count as a failure. |
| H-08 | `test_time_ms_runs_warmup_and_reps` | 3 warm-up runs and 7 timed runs | Function called exactly 10 times; 7 timings recorded; min ≤ median ≤ max | Warm-up runs (compilation, cache filling) must happen but must not be timed. |
| H-09 | `test_benchmark_reference_is_correct_and_timed` | `torch.matmul`, 32×48×16 | Marked correct (error < 1e-5); time > 0; TFLOPS > 0 | End-to-end check of the full benchmark path. |
| H-10 | `test_benchmark_does_not_time_a_wrong_kernel` | A kernel that adds 1.0 to every output | Marked incorrect; time and TFLOPS left empty | **A wrong kernel can never report a speed.** This rule keeps the results table honest. |
| H-11 | `test_cli_writes_csv_with_environment` | CLI run: sizes 16 and 32, FP32, CPU | CSV contains the PyTorch version in its header comments, one row per size, all marked correct | Every saved result records the software it was measured with. |

## 2. Triton environment: `tests/test_triton_smoke.py` (1 case)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| S-01 | `test_vector_add_matches_torch` | Two vectors of length 1000, block size 256 | Equal to `x + y` | Proves Triton runs in this environment before any matrix kernel is tested. 1000 isn't a multiple of 256, so the last block also checks **masking**. |

## 3. Naive GEMM kernel: `tests/test_naive.py` (15 cases)

These check that `naive_matmul` (step 2) gives the right answer for awkward inputs, not just for convenient ones.

### 3.1 Shapes, FP32 (7 cases)

`test_matches_torch_fp32` runs once per shape. The kernel reads K in chunks of `BLOCK_K = 64`.

| ID | Shape (M × N × K) | Edge case tested | Expected result |
|---|---|---|---|
| N-01 | 1 × 1 × 1 | Smallest possible matrix: a single program, a single value | Output shape 1×1, FP32, matches `torch.matmul` |
| N-02 | 16 × 16 × 16 | K is **smaller than one chunk** (16 < 64): most lanes are masked off | Matches `torch.matmul` |
| N-03 | 8 × 12 × 64 | K is **exactly one chunk**: the boundary with no partial chunk | Matches `torch.matmul` |
| N-04 | 5 × 7 × 130 | K spans **three chunks, the last one partial** (64 + 64 + 2) | Matches `torch.matmul` |
| N-05 | 33 × 17 × 45 | **No dimension is a power of two**, which catches off-by-one indexing | Matches `torch.matmul` |
| N-06 | 1 × 24 × 100 | A is a **single row** | Matches `torch.matmul` |
| N-07 | 24 × 1 × 100 | B is a **single column** | Matches `torch.matmul` |

### 3.2 Data types (2 cases)

| ID | Test | Shape | Expected result | Why it matters |
|---|---|---|---|---|
| N-08 | `test_matches_torch_fp16` | 16 × 16 × 16 | Output dtype FP16; matches `torch.matmul` | FP16 is the type later steps use for Tensor Cores. The kernel adds up in FP32 and converts back only at the end. |
| N-09 | `test_matches_torch_fp16` | 33 × 17 × 45 | Same as above | Same check with uneven sizes. |

### 3.3 Behaviour and layout (3 cases)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| N-10 | `test_small_block_k_gives_same_answer` | 9×11×50 with `block_k=8` and with `block_k=64` | Both results equal | Chunk size is a tuning knob. Changing it must never change the answer. |
| N-11 | `test_transposed_inputs_use_strides` | A (13×9) and B (9×20) created as transposed **views** | Inputs confirmed non-contiguous; output matches `torch.matmul` | `.t()` doesn't copy data, it only swaps strides. A kernel that assumes row-major layout would silently return wrong numbers. |
| N-12 | `test_zero_k_gives_zeros` | A is 4×0, B is 0×5 | A 4×5 matrix of zeros | An empty sum is 0. The loop must handle running zero times. |

### 3.4 Input validation (2 cases)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| N-13 | `test_rejects_mismatched_inner_dimension` | A is 3×4, B is 5×6 | `ValueError` mentioning "inner dimensions" | Without this check, the kernel would read past the end of memory. |
| N-14 | `test_rejects_mixed_dtypes` | FP32 A with FP16 B | `ValueError` mentioning "dtypes" | Mixed types would be loaded as the wrong bit patterns. |

### 3.5 Integration (1 case)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| N-15 | `test_registered_in_benchmark` | Benchmark run of `naive`, 12×10×70 | Registered under the name `naive`; marked correct; TFLOPS > 0 | Confirms the kernel can be benchmarked by name, the way the Colab notebook does it. |

---

## Coverage summary

| Area | Cases | Covered |
|---|---|---|
| Benchmark harness | 11 | FLOP counting, unit conversion, reproducibility, error checking, timing loop, CSV output |
| Triton environment | 1 | Triton install, interpreter mode, masking |
| Naive kernel: shapes | 7 | Tiny, one chunk, partial chunk, multi-chunk, odd sizes, single row, single column |
| Naive kernel: types | 2 | FP16 with FP32 accumulation |
| Naive kernel: behaviour | 3 | Chunk-size independence, strided inputs, empty K |
| Naive kernel: validation | 2 | Shape and dtype errors |
| Integration | 1 | Registered and benchmarked by name |
| **Total** | **27** | |

### Not yet covered (planned)

- **Large shapes on a real GPU** (512 to 4096). These run in the Colab notebook rather than CI, because the CPU interpreter is too slow for them.
- **BF16** inputs: in step 4, alongside Tensor Cores.
- **Tiled and autotuned kernels:** in step 3, which will reuse every case above.
