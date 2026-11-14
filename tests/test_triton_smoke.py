"""Checks that Triton itself works in this environment.

Triton needs Linux. With no NVIDIA GPU, it can still run kernels on the CPU
through its interpreter when TRITON_INTERPRET=1 is set before import; CI
does this. On other machines (for example macOS) the test is skipped.
"""
import os

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="no GPU and TRITON_INTERPRET is not set",
)


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n  # the last block may run past the end of the data
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def test_vector_add_matches_torch():
    n = 1000  # deliberately not a multiple of BLOCK, to exercise masking
    x = torch.randn(n, device=DEVICE)
    y = torch.randn(n, device=DEVICE)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    add_kernel[grid](x, y, out, n, BLOCK=256)
    torch.testing.assert_close(out, x + y)
