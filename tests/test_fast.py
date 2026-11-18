"""Correctness tests for the step 4 grouped and Tensor Core GEMMs.

Author: Kajal Changappa

Grouped ordering changes which program computes which tile. If the mapping
were wrong, some tiles of C would never be written, or would be written
twice, so these tests use tile grids where the last group is only partly
full and check every output value.
"""
import os

import pytest
import torch

pytest.importorskip("triton")

from gemm import bench  # noqa: E402
from gemm.fast import FAST_CONFIGS, best_config, fast_matmul, grouped_matmul  # noqa: E402
from gemm.tiled import tiled_matmul  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="no GPU and TRITON_INTERPRET is not set",
)


def assert_matches_torch(a, b, out):
    ok, err = bench.check_close(out, torch.matmul(a, b))
    assert ok, f"max relative error {err:.2e}"


# With 16 x 16 tiles, (80, 48) is a 5 x 3 tile grid. GROUP_M = 2 then gives
# two full groups and one final group of a single tile row.
@pytest.mark.parametrize("group_m", [1, 2, 3, 5, 8])
def test_every_group_size_fills_every_tile(group_m):
    a, b = bench.make_inputs(80, 48, 40, dtype=torch.float16, device=DEVICE)
    out = grouped_matmul(a, b, block_m=16, block_n=16, block_k=16, group_m=group_m)
    assert_matches_torch(a, b, out)


@pytest.mark.parametrize("m,n,k", [
    (1, 1, 1),        # one tile, almost entirely masked
    (64, 64, 32),     # exactly one default tile
    (200, 72, 40),    # 4 x 2 tiles: one partial group with GROUP_M = 8
    (520, 130, 70),   # 9 x 3 tiles: one full group of 8 rows plus 1 row, all edges partial
    (33, 300, 129),   # wide, with partial tiles everywhere
])
def test_grouped_matches_torch_fp16(m, n, k):
    a, b = bench.make_inputs(m, n, k, dtype=torch.float16, device=DEVICE)
    out = grouped_matmul(a, b)
    assert out.shape == (m, n) and out.dtype == torch.float16
    assert_matches_torch(a, b, out)


def test_grouped_works_in_fp32():
    a, b = bench.make_inputs(100, 70, 50, dtype=torch.float32, device=DEVICE)
    assert_matches_torch(a, b, grouped_matmul(a, b))


def test_grouped_agrees_with_tiled():
    # Same tiles, different order: the result must be identical in value.
    a, b = bench.make_inputs(150, 90, 60, dtype=torch.float16, device=DEVICE)
    same_tile = dict(block_m=32, block_n=32, block_k=32)
    torch.testing.assert_close(grouped_matmul(a, b, group_m=4, **same_tile),
                               tiled_matmul(a, b, **same_tile))


def test_fp16_accumulates_in_fp32():
    # 4096 additions of 1.0 * 1.0: an FP16 running sum would stop growing at
    # 2048 (the spacing between FP16 numbers there is 2). FP32 accumulation
    # gets the exact answer, which FP16 can represent.
    a = torch.ones(16, 4096, dtype=torch.float16, device=DEVICE)
    b = torch.ones(4096, 16, dtype=torch.float16, device=DEVICE)
    out = grouped_matmul(a, b, block_m=16, block_n=16, block_k=64)
    assert torch.all(out == 4096)


def test_transposed_inputs_use_strides():
    at, bt = bench.make_inputs(90, 60, 50, dtype=torch.float16, device=DEVICE)
    a, b = bt.t(), at.t()  # a: 60x50, b: 50x90, both non-contiguous
    assert not a.is_contiguous() and not b.is_contiguous()
    assert_matches_torch(a, b, grouped_matmul(a, b))


def test_zero_k_gives_zeros():
    a = torch.empty(5, 0, dtype=torch.float16, device=DEVICE)
    b = torch.empty(0, 6, dtype=torch.float16, device=DEVICE)
    assert torch.equal(grouped_matmul(a, b), torch.zeros(5, 6, dtype=torch.float16, device=DEVICE))


@pytest.mark.parametrize("kwargs,message", [
    ({"block_k": 8}, "power of two"),
    ({"block_n": 96}, "power of two"),
    ({"group_m": 0}, "group_m"),
])
def test_rejects_invalid_parameters(kwargs, message):
    a, b = bench.make_inputs(4, 4, 4, dtype=torch.float16, device=DEVICE)
    with pytest.raises(ValueError, match=message):
        grouped_matmul(a, b, **kwargs)


@pytest.mark.parametrize(
    "config", FAST_CONFIGS,
    ids=lambda c: "{BLOCK_M}x{BLOCK_N}x{BLOCK_K}-w{w}-s{s}".format(w=c.num_warps, s=c.num_stages, **c.kwargs),
)
def test_every_fast_config_is_correct(config):
    kw = config.kwargs
    a, b = bench.make_inputs(150, 140, 100, dtype=torch.float16, device=DEVICE)
    out = grouped_matmul(a, b, block_m=kw["BLOCK_M"], block_n=kw["BLOCK_N"],
                         block_k=kw["BLOCK_K"], group_m=kw["GROUP_M"])
    assert_matches_torch(a, b, out)


def test_fast_configs_fit_t4_shared_memory():
    # FP16: (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 2 bytes * stages <= 64 KB.
    for c in FAST_CONFIGS:
        kw = c.kwargs
        smem = (kw["BLOCK_M"] * kw["BLOCK_K"] + kw["BLOCK_K"] * kw["BLOCK_N"]) * 2 * c.num_stages
        assert smem <= 64 * 1024, f"{kw} with {c.num_stages} stages needs {smem} bytes"


@pytest.mark.skipif(DEVICE == "cpu", reason="the autotuner times configs on a real GPU; the interpreter has no GPU driver")
def test_fast_matmul_autotunes_and_matches_torch():
    m, n, k = 256, 192, 128
    a, b = bench.make_inputs(m, n, k, dtype=torch.float16, device=DEVICE)
    assert_matches_torch(a, b, fast_matmul(a, b))
    chosen = best_config(m, n, k)
    assert chosen is not None and chosen.kwargs in [c.kwargs for c in FAST_CONFIGS]


def test_registered_in_benchmark():
    assert bench.KERNELS["grouped"] is grouped_matmul
    assert bench.KERNELS["fast"] is fast_matmul
    (r,) = bench.run_benchmark({"grouped": grouped_matmul}, [(80, 64, 48)], dtype=torch.float16,
                               device=DEVICE, warmup=1, reps=2)
    assert r.correct and r.tflops > 0
