"""Step 3: a tiled GEMM that reuses data, plus an autotuned version.

Author: Kajal Changappa

The naive kernel (step 2) gives each program ONE output element, so every
program re-reads a whole row of A and column of B. Here each program owns a
BLOCK_M x BLOCK_N TILE of C instead. For every step along K it loads one
BLOCK_M x BLOCK_K tile of A and one BLOCK_K x BLOCK_N tile of B into fast
on-chip memory and multiplies them with tl.dot:

    loads per K step:   BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N   values
    work  per K step:   2 * BLOCK_M * BLOCK_N * BLOCK_K     FLOPs

Each loaded value of A is now reused BLOCK_N times and each value of B
BLOCK_M times. That ratio of work to memory traffic (arithmetic intensity)
is what makes this fast: with 64 x 64 tiles, every value fetched from
global memory feeds 64 multiply-adds instead of 1.

Bigger tiles mean more reuse but need more registers and shared memory, and
leave fewer programs to keep the GPU busy. The best size depends on the GPU
and the matrix shape, so `autotuned_matmul` lets Triton try several
configurations and keep the fastest one for each shape.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def tiled_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Which tile of C this program owns, in plain row-major order.
    # (Step 4 changes this order to make better use of the L2 cache.)
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 2-D grids of pointers to the first A tile (BLOCK_M x BLOCK_K) and the
    # first B tile (BLOCK_K x BLOCK_N). Broadcasting a column of row offsets
    # against a row of column offsets builds every address at once.
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # Masks zero-fill anything outside the matrix, so edge tiles work.
        k_remaining = K - k0
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        # Slide both tiles one step along K.
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# Candidate tile shapes for the autotuner. Shared memory per stage is
# (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * bytes per value, so for FP32 the
# 128x128x32 tile with 2 stages needs 64 KB, the most a T4 (Turing) allows.
# Configurations that do not fit on a given GPU fail to compile and the
# autotuner simply skips them.
AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
]

# The same kernel, wrapped so that the first call for each (M, N, K) times
# every config and caches the winner.
autotuned_matmul_kernel = triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])(tiled_matmul_kernel)


def _check_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"expected 2-D matrices, got {a.ndim}-D and {b.ndim}-D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"inner dimensions differ: A is {tuple(a.shape)}, B is {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtypes differ: {a.dtype} and {b.dtype}")
    if a.device != b.device:
        raise ValueError(f"devices differ: {a.device} and {b.device}")


def _launch(kernel, a, b, **block_sizes):
    _check_inputs(a, b)
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c
    if K == 0:
        return c.zero_()

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **block_sizes,
    )
    return c


def tiled_matmul(a: torch.Tensor, b: torch.Tensor,
                 block_m: int = 64, block_n: int = 64, block_k: int = 32) -> torch.Tensor:
    """C = A @ B with one Triton program per BLOCK_M x BLOCK_N tile of C.

    Block sizes must be powers of two and at least 16 (a tl.dot requirement).
    """
    for name, size in (("block_m", block_m), ("block_n", block_n), ("block_k", block_k)):
        if size < 16 or size & (size - 1):
            raise ValueError(f"{name} must be a power of two >= 16, got {size}")
    return _launch(tiled_matmul_kernel, a, b, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)


def autotuned_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B, with block sizes chosen by Triton's autotuner per shape."""
    return _launch(autotuned_matmul_kernel, a, b)


def best_config(m: int, n: int, k: int):
    """The config the autotuner picked for this shape, or None if not tuned yet."""
    for key, config in autotuned_matmul_kernel.cache.items():
        if tuple(key[:3]) == (m, n, k):
            return config
    return None
