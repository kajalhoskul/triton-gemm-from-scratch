"""Correctness tests for the step 2 naive Triton GEMM.

Without a GPU these run through Triton's CPU interpreter (TRITON_INTERPRET=1),
which is slow, so the shapes are small. What matters is that they hit the
edge cases: sizes that are not multiples of BLOCK_K, single rows and columns,
K smaller than one block, and non-contiguous inputs.
"""
import os

import pytest
import torch

pytest.importorskip("triton")

from gemm import bench  # noqa: E402
from gemm.naive import naive_matmul  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="no GPU and TRITON_INTERPRET is not set",
)

SHAPES = [
    (1, 1, 1),      # smallest possible
    (16, 16, 16),   # K smaller than one BLOCK_K chunk
    (8, 12, 64),    # K exactly one chunk
    (5, 7, 130),    # K spans three chunks, the last one partial
    (33, 17, 45),   # nothing is a power of two
    (1, 24, 100),   # a single row of A
    (24, 1, 100),   # a single column of B
]


def assert_matches_torch(a, b, out):
    ok, err = bench.check_close(out, torch.matmul(a, b))
    assert ok, f"max relative error {err:.2e}"


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_matches_torch_fp32(m, n, k):
    a, b = bench.make_inputs(m, n, k, dtype=torch.float32, device=DEVICE)
    out = naive_matmul(a, b)
    assert out.shape == (m, n) and out.dtype == torch.float32
    assert_matches_torch(a, b, out)


@pytest.mark.parametrize("m,n,k", [(16, 16, 16), (33, 17, 45)])
def test_matches_torch_fp16(m, n, k):
    a, b = bench.make_inputs(m, n, k, dtype=torch.float16, device=DEVICE)
    out = naive_matmul(a, b)
    assert out.dtype == torch.float16
    assert_matches_torch(a, b, out)


def test_small_block_k_gives_same_answer():
    # Splitting K into more, smaller chunks must not change the result.
    a, b = bench.make_inputs(9, 11, 50, device=DEVICE)
    torch.testing.assert_close(naive_matmul(a, b, block_k=8), naive_matmul(a, b, block_k=64))


def test_transposed_inputs_use_strides():
    # .t() returns a view with swapped strides, not a copy. The kernel must
    # follow the strides rather than assume a row-major layout.
    at, bt = bench.make_inputs(20, 13, 9, device=DEVICE)  # at: 20x9, bt: 9x13
    a, b = bt.t(), at.t()  # a: 13x9, b: 9x20, both non-contiguous
    assert not a.is_contiguous() and not b.is_contiguous()
    assert_matches_torch(a, b, naive_matmul(a, b))


def test_zero_k_gives_zeros():
    a = torch.empty(4, 0, device=DEVICE)
    b = torch.empty(0, 5, device=DEVICE)
    assert torch.equal(naive_matmul(a, b), torch.zeros(4, 5, device=DEVICE))


def test_rejects_mismatched_inner_dimension():
    with pytest.raises(ValueError, match="inner dimensions"):
        naive_matmul(torch.zeros(3, 4, device=DEVICE), torch.zeros(5, 6, device=DEVICE))


def test_rejects_mixed_dtypes():
    with pytest.raises(ValueError, match="dtypes"):
        naive_matmul(torch.zeros(3, 4, device=DEVICE),
                     torch.zeros(4, 6, device=DEVICE, dtype=torch.float16))


def test_registered_in_benchmark():
    assert bench.KERNELS["naive"] is naive_matmul
    (r,) = bench.run_benchmark({"naive": naive_matmul}, [(12, 10, 70)], device=DEVICE, warmup=1, reps=2)
    assert r.correct and r.tflops > 0
