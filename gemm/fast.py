"""Step 4: FP16 Tensor Cores and grouped tile ordering.

Author: Kajal Changappa

Two changes on top of the step 3 tiled kernel:

1. FP16 inputs with an FP32 accumulator. When both tiles passed to tl.dot
   are FP16, Triton compiles it to Tensor Core instructions (mma), which on a
   T4 have a peak of 65 TFLOPS against 8.1 TFLOPS for ordinary FP32 maths.
   Accumulating in FP32 keeps the rounding error from growing with K.
   FP16 also halves the bytes per value, so each tile costs half the memory
   traffic and half the shared memory, which allows bigger tiles.

2. Grouped tile ordering (see gemm/ordering.py). Programs that run at the
   same time are given tiles from a few rows AND a few columns, instead of
   one whole row, so they share more of the A and B tiles they load through
   the L2 cache.

Everything else (masks, strides, the K loop) is the same as step 3.
"""
import torch
import triton
import triton.language as tl

from gemm.tiled import _check_inputs


@triton.jit
def grouped_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped ordering: the same arithmetic as gemm.ordering.grouped_tile.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # FP32 accumulator even for FP16 inputs: Tensor Cores multiply in FP16
    # and add in FP32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# Tile configs for FP16 on a T4 (64 KB shared memory). With 2-byte values,
# shared memory per config is (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * 2 * stages,
# so 128 x 128 x 32 with 4 stages uses exactly 64 KB. More stages let the
# loads for the next K steps run while Tensor Cores work on the current one.
FAST_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=4),
]

fast_matmul_kernel = triton.autotune(configs=FAST_CONFIGS, key=["M", "N", "K"])(grouped_matmul_kernel)


def _launch(kernel, a, b, **meta):
    _check_inputs(a, b)
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c
    if K == 0:
        return c.zero_()

    def grid(m):
        return (triton.cdiv(M, m["BLOCK_M"]) * triton.cdiv(N, m["BLOCK_N"]),)

    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **meta,
    )
    return c


def grouped_matmul(a: torch.Tensor, b: torch.Tensor, block_m: int = 64, block_n: int = 64,
                   block_k: int = 32, group_m: int = 8) -> torch.Tensor:
    """C = A @ B with fixed tile sizes and grouped tile ordering."""
    for name, size in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if size < 16 or size & (size - 1):
            raise ValueError(f"{name} must be a power of two >= 16, got {size}")
    if group_m < 1:
        raise ValueError(f"group_m must be at least 1, got {group_m}")
    return _launch(grouped_matmul_kernel, a, b,
                   BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m)


def fast_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B: grouped ordering plus autotuned tiles. Use FP16 inputs for Tensor Cores."""
    return _launch(fast_matmul_kernel, a, b)


def best_config(m: int, n: int, k: int):
    """The config the autotuner picked for this shape, or None if not tuned yet."""
    for key, config in fast_matmul_kernel.cache.items():
        if tuple(key[:3]) == (m, n, k):
            return config
    return None
