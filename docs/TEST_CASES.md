# Test cases

**Project:** triton-gemm-from-scratch · **Author:** Kajal Changappa · **Covers:** steps 1 to 3

57 automated test cases in 4 files. Run them with:

```bash
pytest -v
```

## Where the tests run

| Environment | Harness tests | Triton tests | What it proves |
|---|---|---|---|
| **GitHub Actions CI** (Ubuntu, CPU only, `TRITON_INTERPRET=1`) | ✅ run | ✅ run, through Triton's CPU interpreter | Every kernel computes the right answer on every push |
| **Google Colab** (NVIDIA T4) | ✅ run | ✅ run, compiled for the GPU | The kernels also compile and run correctly on real hardware, and the autotuner (T-21) runs |
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

## 4. Tiled and autotuned GEMM: `tests/test_tiled.py` (30 cases)

Tiling adds failure modes the naive kernel didn't have: tiles that hang over the edge of the matrix, matrices smaller than one tile, and block sizes that must never change the answer. The default tile is **64 × 64**, with **BLOCK_K = 32**.

### 4.1 Tile edges, FP32 (8 cases)

`test_tiled_matches_torch_fp32` runs once per shape.

| ID | Shape (M × N × K) | Edge case tested | Expected result |
|---|---|---|---|
| T-01 | 1 × 1 × 1 | Far smaller than one tile in every dimension: almost everything is masked | Output shape 1×1, FP32, matches `torch.matmul` |
| T-02 | 64 × 64 × 32 | Exactly one tile and exactly one K step, with no masking at all | Matches |
| T-03 | 128 × 64 × 96 | Whole tiles only: 2 × 1 tiles, 3 K steps | Matches |
| T-04 | 65 × 64 × 32 | **M one past a tile edge**: a second tile row with only 1 valid row | Matches |
| T-05 | 64 × 65 × 32 | **N one past a tile edge** | Matches |
| T-06 | 64 × 64 × 33 | **K one past a step**: the last K step loads just 1 column of A | Matches |
| T-07 | 100 × 70 × 50 | **Partial tiles in M, N and K at the same time** | Matches |
| T-08 | 3 × 130 × 7 | Very wide and very short: 3 tiles side by side, each mostly empty | Matches |

### 4.2 Data types (2 cases)

| ID | Test | Shape | Expected result | Why it matters |
|---|---|---|---|---|
| T-09 | `test_tiled_matches_torch_fp16` | 64 × 64 × 32 | Output FP16; matches | FP16 tiles go through `tl.dot` with an FP32 accumulator, which is the path to Tensor Cores in step 4. |
| T-10 | `test_tiled_matches_torch_fp16` | 100 × 70 × 50 | Same | FP16 with partial tiles. |

### 4.3 Block sizes (3 cases)

`test_block_sizes_do_not_change_the_answer`, on a 70 × 90 × 110 problem.

| ID | BLOCK_M × BLOCK_N × BLOCK_K | Expected result | Why it matters |
|---|---|---|---|
| T-11 | 16 × 16 × 16 | Same answer as the default tile | The smallest tile `tl.dot` allows. |
| T-12 | 32 × 64 × 16 | Same | A non-square tile. |
| T-13 | 128 × 32 × 64 | Same | A tall tile with a large K step. |

### 4.4 Behaviour and layout (3 cases)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| T-14 | `test_tiled_agrees_with_naive` | 40 × 30 × 50 | Tiled and naive outputs agree | Each step must give the same answer as the step before it. |
| T-15 | `test_transposed_inputs_use_strides` | Transposed views, 70 × 40 and 40 × 80 | Inputs confirmed non-contiguous; output matches | 2-D pointer grids must follow the strides, not assume row-major layout. |
| T-16 | `test_zero_k_gives_zeros` | A is 5×0, B is 0×6 | A 5×6 matrix of zeros | Handled before launch, so no tile loop runs on an empty K. |

### 4.5 Input validation (3 cases)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| T-17 | `test_rejects_invalid_block_size` | `block_m = 8` | `ValueError` mentioning "power of two" | `tl.dot` needs every tile dimension to be at least 16. |
| T-18 | `test_rejects_invalid_block_size` | `block_m = 48` | `ValueError` mentioning "power of two" | Triton block shapes must be powers of two. |
| T-19 | `test_rejects_mismatched_inner_dimension` | A is 3×4, B is 5×6 | `ValueError` mentioning "inner dimensions" | Same guard as the naive kernel. |

### 4.6 Autotuning (10 cases)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| T-20a to T-20h | `test_every_autotune_config_is_correct` (one case per config: 128×128×32, 128×128×16, 128×64×32, 64×128×32, 64×64×32, 64×64×64, 32×64×32, 32×32×32) | 72 × 40 × 56 with that config's block sizes | Matches `torch.matmul` | The autotuner may pick **any** of these, so each must be correct on its own. Runs in CI. |
| T-21 | `test_autotuned_matches_torch_and_records_choice` | 72 × 40 × 56 through `autotuned_matmul` | No config cached before the call; output matches; afterwards a config is cached, and it's one of the 8 candidates | Checks the full autotune path end to end. **GPU only:** the autotuner times configs on a GPU, and the CPU interpreter has no GPU driver. Runs on Colab, skipped in CI. |
| T-22 | `test_autotune_configs_are_valid` | All 8 configs | Every block size is a power of two ≥ 16 | Catches a bad config before it can fail at compile time. |

### 4.7 Integration (1 case)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| T-23 | `test_registered_in_benchmark` | Benchmark run of `tiled`, 48 × 40 × 72 | `tiled` and `autotuned` registered by name; marked correct; TFLOPS > 0 | Confirms the Colab notebook can benchmark both by name. |

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
| Naive kernel: integration | 1 | Registered and benchmarked by name |
| Tiled kernel: tile edges | 8 | Smaller than a tile, exact tiles, one past each edge in M, N and K, all three at once |
| Tiled kernel: types | 2 | FP16 through `tl.dot` with an FP32 accumulator |
| Tiled kernel: block sizes | 3 | Smallest, non-square and tall tiles give the same answer |
| Tiled kernel: behaviour | 3 | Agrees with naive, strided inputs, empty K |
| Tiled kernel: validation | 3 | Invalid block sizes, shape errors |
| Autotuning | 10 | Every candidate config is correct, full autotune path (GPU), config sanity |
| Tiled kernel: integration | 1 | Both kernels registered and benchmarked by name |
| **Total** | **57** | 56 run in CI; T-21 runs on GPU only |

### Not yet covered (planned)

- **Large shapes on a real GPU** (512 to 4096). These run in the Colab notebook rather than CI, because the CPU interpreter is too slow for them.
- **BF16** inputs: in step 4, alongside Tensor Cores.
- **Grouped program ordering:** in step 4, which will reuse the tile-edge cases above.
