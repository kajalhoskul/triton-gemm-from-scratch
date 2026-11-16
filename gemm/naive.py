"""Step 2: the simplest correct GEMM in Triton.

One Triton program computes ONE element of C:

    C[m, n] = sum over k of A[m, k] * B[k, n]

It walks along row m of A and column n of B in chunks of BLOCK_K, multiplies
them element by element, and sums. There are M*N programs in total.

This is deliberately naive. Every program re-reads a full row of A and a full
column of B from global memory, so each value of A is loaded N times and each
value of B is loaded M times, with no reuse between programs. It also never
uses tl.dot, so it cannot use Tensor Cores. Steps 3 and 4 fix both problems;
this version is the baseline they are measured against.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def naive_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_K: tl.constexpr,
):
    # Which output element this program owns.
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N

    # Partial sums, one per lane, kept in FP32 even for FP16 inputs so that
    # rounding error does not grow with K.
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask = offs_k < K  # the last chunk can run past the end of K
        # Strides let A and B be any 2-D layout, including transposed views.
        a = tl.load(a_ptr + m * stride_am + offs_k * stride_ak, mask=mask, other=0.0)
        b = tl.load(b_ptr + offs_k * stride_bk + n * stride_bn, mask=mask, other=0.0)
        acc += a.to(tl.float32) * b.to(tl.float32)

    c = tl.sum(acc, axis=0)
    tl.store(c_ptr + m * stride_cm + n * stride_cn, c.to(c_ptr.dtype.element_ty))


def naive_matmul(a: torch.Tensor, b: torch.Tensor, block_k: int = 64) -> torch.Tensor:
    """C = A @ B with one Triton program per output element."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D matrices, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions differ: A is {tuple(a.shape)}, B is {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtypes differ: {a.dtype} and {b.dtype}")
    if a.device != b.device:
        raise ValueError(f"devices differ: {a.device} and {b.device}")

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c

    grid = (M * N,)
    naive_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_K=block_k,
    )
    return c
