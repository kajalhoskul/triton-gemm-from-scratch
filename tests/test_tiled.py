"""Correctness tests for the step 3 tiled and autotuned Triton GEMMs.

Author: Kajal Changappa

Tiling adds new ways to go wrong that the naive kernel did not have: tiles
that hang over the edge of the matrix in M, N or K, matrices smaller than a
single tile, and block sizes that must not change the answer. These tests
cover each one.
"""
import os

import pytest
import torch

pytest.importorskip("triton")

from gemm import bench  # noqa: E402
from gemm.tiled import (  # noqa: E402
    AUTOTUNE_CONFIGS,
    autotuned_matmul,
    best_config,
    tiled_matmul,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="no GPU and TRITON_INTERPRET is not set",
)

# Default tile is 64 x 64 with BLOCK_K = 32.
SHAPES = [
    (1, 1, 1),        # far smaller than one tile in every dimension
    (64, 64, 32),     # exactly one tile, exactly one K step
    (128, 64, 96),    # whole tiles only: 2 x 1 tiles, 3 K steps
    (65, 64, 32),     # M one past a tile edge: a second, almost empty tile row
    (64, 65, 32),     # N one past a tile edge
    (64, 64, 33),     # K one past a step: the last K step loads 1 column
    (100, 70, 50),    # partial tiles in M, N and K at once
    (3, 130, 7),      # very wide, very short
]


def assert_matches_torch(a, b, out):
    ok, err = bench.check_close(out, torch.matmul(a, b))
    assert ok, f"max relative error {err:.2e}"


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_tiled_matches_torch_fp32(m, n, k):
    a, b = bench.make_inputs(m, n, k, dtype=torch.float32, device=DEVICE)
    out = tiled_matmul(a, b)
    assert out.shape == (m, n) and out.dtype == torch.float32
    assert_matches_torch(a, b, out)


@pytest.mark.parametrize("m,n,k", [(64, 64, 32), (100, 70, 50)])
def test_tiled_matches_torch_fp16(m, n, k):
    a, b = bench.make_inputs(m, n, k, dtype=torch.float16, device=DEVICE)
    out = tiled_matmul(a, b)
    assert out.dtype == torch.float16
    assert_matches_torch(a, b, out)


@pytest.mark.parametrize("bm,bn,bk", [(16, 16, 16), (32, 64, 16), (128, 32, 64)])
def test_block_sizes_do_not_change_the_answer(bm, bn, bk):
    a, b = bench.make_inputs(70, 90, 110, device=DEVICE)
    ref = tiled_matmul(a, b)  # default 64 x 64 x 32
    out = tiled_matmul(a, b, block_m=bm, block_n=bn, block_k=bk)
    assert bench.check_close(out, ref)[0]


def test_tiled_agrees_with_naive():
    from gemm.naive import naive_matmul

    a, b = bench.make_inputs(40, 30, 50, device=DEVICE)
    assert bench.check_close(tiled_matmul(a, b), naive_matmul(a, b))[0]


def test_transposed_inputs_use_strides():
    at, bt = bench.make_inputs(80, 70, 40, device=DEVICE)  # at: 80x40, bt: 40x70
    a, b = bt.t(), at.t()  # a: 70x40, b: 40x80, both non-contiguous
    assert not a.is_contiguous() and not b.is_contiguous()
    assert_matches_torch(a, b, tiled_matmul(a, b))


def test_zero_k_gives_zeros():
    a = torch.empty(5, 0, device=DEVICE)
    b = torch.empty(0, 6, device=DEVICE)
    assert torch.equal(tiled_matmul(a, b), torch.zeros(5, 6, device=DEVICE))


@pytest.mark.parametrize("bad", [8, 48])
def test_rejects_invalid_block_size(bad):
    a, b = bench.make_inputs(4, 4, 4, device=DEVICE)
    with pytest.raises(ValueError, match="power of two"):
        tiled_matmul(a, b, block_m=bad)


def test_rejects_mismatched_inner_dimension():
    with pytest.raises(ValueError, match="inner dimensions"):
        tiled_matmul(torch.zeros(3, 4, device=DEVICE), torch.zeros(5, 6, device=DEVICE))


@pytest.mark.parametrize(
    "config", AUTOTUNE_CONFIGS,
    ids=lambda c: "{BLOCK_M}x{BLOCK_N}x{BLOCK_K}".format(**c.kwargs),
)
def test_every_autotune_config_is_correct(config):
    # The autotuner may pick any of these, so each one must be correct on its
    # own. Runs everywhere, including CI without a GPU.
    m, n, k = 72, 40, 56
    a, b = bench.make_inputs(m, n, k, device=DEVICE)
    kw = config.kwargs
    out = tiled_matmul(a, b, block_m=kw["BLOCK_M"], block_n=kw["BLOCK_N"], block_k=kw["BLOCK_K"])
    assert_matches_torch(a, b, out)


@pytest.mark.skipif(DEVICE == "cpu", reason="the autotuner times configs on a real GPU; the interpreter has no GPU driver")
def test_autotuned_matches_torch_and_records_choice():
    m, n, k = 72, 40, 56
    a, b = bench.make_inputs(m, n, k, device=DEVICE)
    assert best_config(m, n, k) is None  # nothing tuned for this shape yet
    assert_matches_torch(a, b, autotuned_matmul(a, b))
    chosen = best_config(m, n, k)
    assert chosen is not None
    assert chosen.kwargs in [c.kwargs for c in AUTOTUNE_CONFIGS]


def test_autotune_configs_are_valid():
    for config in AUTOTUNE_CONFIGS:
        for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K"):
            size = config.kwargs[name]
            assert size >= 16 and size & (size - 1) == 0, f"{name}={size}"


def test_registered_in_benchmark():
    assert bench.KERNELS["tiled"] is tiled_matmul
    assert bench.KERNELS["autotuned"] is autotuned_matmul
    (r,) = bench.run_benchmark({"tiled": tiled_matmul}, [(48, 40, 72)], device=DEVICE, warmup=1, reps=2)
    assert r.correct and r.tflops > 0
