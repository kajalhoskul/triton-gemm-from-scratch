# Test cases

**Project:** triton-gemm-from-scratch · **Author:** Kajal Changappa · **Covers:** steps 1 to 4

138 automated test cases in 7 files. Run them with:

```bash
pytest -v
```

## Where the tests run

| Environment | Harness tests | Triton tests | What it proves |
|---|---|---|---|
| **GitHub Actions CI** (Ubuntu, CPU only, `TRITON_INTERPRET=1`) | ✅ run | ✅ run, through Triton's CPU interpreter | Every kernel computes the right answer on every push |
| **Google Colab** (NVIDIA T4) | ✅ run | ✅ run, compiled for the GPU | The kernels also compile and run correctly on real hardware, and the autotuners (T-21, F-21) run |
| **macOS** (Triton cannot install) | ✅ run, plus the ordering and roofline tests | ⏭️ skipped automatically | The harness and the performance models work on any machine |

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

## 5. FP16 Tensor Core GEMM: `tests/test_fast.py` (29 cases)

Grouped ordering changes **which program computes which tile**. A wrong mapping would leave some tiles of C unwritten, or write them twice, so these tests use tile grids where the last group is only partly full, and check every output value.

### 5.1 Group sizes (5 cases)

`test_every_group_size_fills_every_tile`: an 80 × 48 × 40 FP16 problem with 16 × 16 tiles, which is a **5 × 3 tile grid**.

| ID | GROUP_M | What happens to the 5 tile rows | Expected result |
|---|---|---|---|
| F-01 | 1 | 5 groups of 1, the same as row-major order | Matches `torch.matmul` |
| F-02 | 2 | Groups of 2, 2 and a **partial group of 1** | Matches |
| F-03 | 3 | A group of 3 and a **partial group of 2** | Matches |
| F-04 | 5 | Exactly one full group | Matches |
| F-05 | 8 | One group **larger than the grid**, clipped to 5 rows | Matches |

### 5.2 Shapes, FP16, default 64 × 64 × 32 tiles with GROUP_M = 8 (5 cases)

| ID | Shape (M × N × K) | Edge case tested | Expected result |
|---|---|---|---|
| F-06 | 1 × 1 × 1 | One tile, almost entirely masked | Shape 1×1, FP16, matches |
| F-07 | 64 × 64 × 32 | Exactly one tile, no masking | Matches |
| F-08 | 200 × 72 × 40 | 4 × 2 tiles: a single **partial group** (4 rows < 8) | Matches |
| F-09 | 520 × 130 × 70 | 9 × 3 tiles: **one full group of 8 plus a group of 1**, with partial edges in M, N and K | Matches |
| F-10 | 33 × 300 × 129 | Wide, with partial tiles everywhere | Matches |

### 5.3 Behaviour (5 cases)

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| F-11 | `test_grouped_works_in_fp32` | 100 × 70 × 50, FP32 | Matches | The kernel isn't limited to FP16. |
| F-12 | `test_grouped_agrees_with_tiled` | 150 × 90 × 60 FP16, the same 32 × 32 × 32 tiles in both kernels | Grouped output equals step 3 tiled output | Only the order changes, never the result. |
| F-13 | `test_fp16_accumulates_in_fp32` | All-ones 16 × 4096 times 4096 × 16, FP16 | Every output is **exactly 4096** | An FP16 running sum would get stuck at 2,048, where the gap between FP16 numbers is 2. This proves the accumulator is FP32. |
| F-14 | `test_transposed_inputs_use_strides` | Transposed FP16 views | Inputs confirmed non-contiguous; output matches | Strides are still followed under the new ordering. |
| F-15 | `test_zero_k_gives_zeros` | A is 5×0, B is 0×6, FP16 | Zeros | Empty K is handled before launch. |

### 5.4 Input validation (3 cases)

| ID | Input | Expected result |
|---|---|---|
| F-16 | `block_k = 8` | `ValueError` mentioning "power of two" (`tl.dot` needs ≥ 16) |
| F-17 | `block_n = 96` | `ValueError` mentioning "power of two" |
| F-18 | `group_m = 0` | `ValueError` mentioning "group_m" (a group of 0 would divide by zero) |

### 5.5 Autotuning (11 cases)

| ID | Test | Expected result | Why it matters |
|---|---|---|---|
| F-19a to F-19h | `test_every_fast_config_is_correct` (one case per config: 128×128×32 with 4 warps and 4 stages, 128×128×32 w8 s3, 128×128×64 w8 s2, 128×64×32 w4 s4, 64×128×32 w4 s4, 64×64×64 w4 s3, 64×64×32 w4 s4, 128×32×32 w4 s4) on 150 × 140 × 100 FP16 | Matches `torch.matmul` | The autotuner may pick any of them, so each must be correct. Runs in CI. |
| F-20 | `test_fast_configs_fit_t4_shared_memory` | Every config needs ≤ 64 KB: (BLOCK_M·BLOCK_K + BLOCK_K·BLOCK_N) × 2 bytes × stages | A config that can't fit on the target GPU is wasted autotuning time. |
| F-21 | `test_fast_matmul_autotunes_and_matches_torch` | On 256 × 192 × 128 FP16: output matches; the chosen config is one of the 8 | The full autotune path. **GPU only** (the autotuner needs a GPU driver), so it runs on Colab and is skipped in CI. |

### 5.6 Integration (1 case)

| ID | Test | Expected result |
|---|---|---|
| F-22 | `test_registered_in_benchmark` | `grouped` and `fast` registered by name; a benchmark run is marked correct with TFLOPS > 0 |

## 6. Tile ordering model: `tests/test_ordering.py` (41 cases)

Pure Python, so these run everywhere, including the Mac. [`gemm/ordering.py`](../gemm/ordering.py) is the plain-Python model of the mapping in the step 4 kernel.

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| O-01 to O-32 | `test_every_tile_visited_exactly_once` | **8 tile grids** (1×1, 4×4, 5×3, 7×2, 3×9, 16×16, 1×10, 10×1) × **4 group sizes** (1, 2, 3, 8) | The program-to-tile mapping is a **one-to-one cover**: M·N programs, and every tile appears exactly once | The core property. A tile visited twice wastes work, and a tile never visited leaves garbage in C. |
| O-33 | `test_matches_the_documented_4x4_example` | 4 × 4 grid, GROUP_M = 2 | Matches the picture in the docstring and the README | The documentation and the code agree. |
| O-34 | `test_group_of_one_is_row_major` | 5 × 7 grid, GROUP_M = 1 | The same as step 3's order for every program | Grouping generalises row-major order rather than replacing it. |
| O-35 | `test_last_group_can_be_partial` | 5 × 3 grid, GROUP_M = 2 | Programs 12 to 14 map to tile row 4, columns 0 to 2 | The partial final group is handled. |
| O-36 | `test_grouping_needs_fewer_distinct_tiles` | First 16 programs on a 16 × 16 grid | Row-major needs 1 row + 16 columns = **17 tiles**; grouped (GROUP_M = 4) needs 4 + 4 = **8 tiles** | Shows **why** grouping helps L2 cache reuse, with numbers. |
| O-37 to O-41 | `test_cdiv` | (0,4), (1,4), (4,4), (5,4), (130,64) | 0, 1, 1, 2, 3 | Rounding up tile counts, the most common off-by-one source. |

## 7. Roofline model: `tests/test_roofline.py` (11 cases)

Pure Python. T4 peaks from NVIDIA's specifications: 8.1 TFLOPS FP32, 65 TFLOPS FP16 Tensor Core, 320 GB/s memory.

| ID | Test | Input | Expected result | Why it matters |
|---|---|---|---|---|
| R-01 | `test_min_bytes_reads_a_and_b_and_writes_c_once` | 2 × 3 × 4, FP16 | (8 + 12 + 6) × 2 = 52 bytes | The minimum traffic every GEMM must pay. |
| R-02 | `test_intensity_of_square_gemm` | 4096³ in FP16 and FP32 | 2N / (3 × bytes): 1,365 and 683 FLOPs per byte | Matches the closed-form formula. |
| R-03 | `test_fp16_doubles_intensity_over_fp32` | 512³ | FP16 intensity = 2 × FP32 intensity | Half the bytes for the same FLOPs. |
| R-04 | `test_peak_depends_on_dtype` | T4 | FP16 gives 65.0, FP32 gives 8.1 | Each dtype is judged against the right ceiling. |
| R-05 | `test_low_intensity_is_memory_bound` | 10 FLOP/B | 10 × 320 GB/s = 3.2 TFLOPS | The memory ceiling. |
| R-06 | `test_high_intensity_is_compute_bound` | 10,000 FLOP/B | Capped at 65 TFLOPS | The compute ceiling. |
| R-07 | `test_skinny_gemm_is_memory_bound_on_t4` | 4096 × 4096 × 16, FP16 | "memory", with a ceiling below 65 TFLOPS | Small K means little reuse, so it's memory-bound. |
| R-08 | `test_large_square_fp16_gemm_is_compute_bound_on_t4` | 4096³ FP16 measured at 32.5 TFLOPS | "compute"; ceiling 65; **50.0 %** of ceiling | The percentage that will go in the results. |
| R-09 | `test_analyze_file_reads_bench_csv` | A CSV with one correct and one wrong result | Only the correct row is analysed, with the right ceiling | A wrong kernel never gets a performance score. |
| R-10 | `test_analyze_file_rejects_unknown_gpu` | A CSV from "cpu" | `ValueError` "no GPU spec", unless a custom spec is passed | No made-up peaks for unknown hardware. |
| R-11 | `test_cli_prints_table` | `python -m gemm.roofline file.csv` | The table shows the kernel, the size and "compute" | The command the Colab notebook runs. |

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
| Grouped kernel: group sizes | 5 | Full, partial and oversized groups |
| Grouped kernel: shapes | 5 | One tile, partial groups, full plus partial groups, edges everywhere |
| Grouped kernel: behaviour | 5 | FP32 support, agrees with tiled, FP32 accumulation proof, strides, empty K |
| Grouped kernel: validation | 3 | Block sizes, group size |
| Fast kernel: autotuning | 11 | Every config correct, every config fits 64 KB, full autotune path (GPU) |
| Grouped kernel: integration | 1 | Registered and benchmarked by name |
| Tile ordering model | 41 | One-to-one cover over 32 grid and group combinations, docs example, row-major special case, partial group, cache benefit, `cdiv` |
| Roofline model | 11 | Minimum bytes, intensity, both ceilings, bound classification, CSV analysis, CLI |
| **Total** | **138** | 136 run in CI; T-21 and F-21 run on GPU only |

### Not yet covered (planned)

- **Large shapes on a real GPU** (512 to 4096). These run in the Colab notebook rather than CI, because the CPU interpreter is too slow for them.
- **BF16** inputs: the T4 (Turing) has no BF16 Tensor Cores, so BF16 is left out rather than tested on hardware that can't run it fast.
- **Measured L2 hit rates:** grouped ordering's cache benefit is shown by the model (O-36) and by benchmark speed, not by hardware counters.
