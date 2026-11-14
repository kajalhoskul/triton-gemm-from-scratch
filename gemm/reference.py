"""The baseline every kernel is checked and measured against.

On an NVIDIA GPU, `torch.matmul` dispatches to cuBLAS, NVIDIA's hand-tuned
GEMM library. That makes it both the correctness oracle and the speed bar.
"""
import torch


def torch_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B using PyTorch (cuBLAS on GPU)."""
    return torch.matmul(a, b)
