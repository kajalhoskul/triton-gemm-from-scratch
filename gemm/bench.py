"""Benchmark harness: check a GEMM kernel is correct, then time it fairly.

Every kernel is a function `fn(a, b) -> c` computing C = A @ B, where A is
M x K and B is K x N. For each shape the harness:

1. checks the output against `torch.matmul` (no timing for a wrong kernel),
2. runs warm-up calls so compilation and caches do not count,
3. times many repetitions and reports the median,
4. converts the time to TFLOPS using 2*M*N*K floating-point operations.

Usage (on a machine with an NVIDIA GPU):

    python -m gemm.bench --kernels torch --dtype float16 --sizes 512 1024 2048 4096
"""
from __future__ import annotations

import argparse
import csv
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from gemm.reference import torch_matmul

Kernel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# Kernels the CLI can run, by name. Later steps register their Triton kernels here.
KERNELS: Dict[str, Kernel] = {
    "torch": torch_matmul,
}

try:  # Triton only installs on Linux; elsewhere only the reference is available.
    from gemm.naive import naive_matmul
    from gemm.tiled import autotuned_matmul, tiled_matmul

    KERNELS["naive"] = naive_matmul
    KERNELS["tiled"] = tiled_matmul
    KERNELS["autotuned"] = autotuned_matmul

    from gemm.fast import fast_matmul, grouped_matmul

    KERNELS["grouped"] = grouped_matmul
    KERNELS["fast"] = fast_matmul
except ImportError:
    pass

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Allowed error, measured as max|out - ref| / max|ref|.
# FP32 allows 1e-2 because on Ampere and newer GPUs Triton's tl.dot uses
# TF32 (10-bit mantissa) by default, which is a deliberate speed/precision
# trade-off, not a bug. FP16/BF16 outputs are rounded to 10/7 mantissa bits.
TOLERANCES = {
    torch.float32: 1e-2,
    torch.float16: 1e-2,
    torch.bfloat16: 3e-2,
}


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float
    reps: int


@dataclass
class Result:
    kernel: str
    m: int
    n: int
    k: int
    dtype: str
    correct: bool
    max_rel_err: float
    median_ms: Optional[float]
    tflops: Optional[float]


def gemm_flops(m: int, n: int, k: int) -> int:
    """Each of the M*N outputs needs K multiplies and K adds."""
    return 2 * m * n * k


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return gemm_flops(m, n, k) / (ms * 1e-3) / 1e12


def make_inputs(m: int, n: int, k: int, dtype=torch.float32, device="cpu", seed=0):
    """Random A (M x K) and B (K x N), reproducible for a given seed."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(m, k, generator=gen).to(device=device, dtype=dtype)
    b = torch.randn(k, n, generator=gen).to(device=device, dtype=dtype)
    return a, b


def max_rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest absolute error, scaled by the largest reference value."""
    out, ref = out.float(), ref.float()
    scale = ref.abs().max().item()
    err = (out - ref).abs().max().item()
    return err / scale if scale > 0 else err


def check_close(out: torch.Tensor, ref: torch.Tensor) -> Tuple[bool, float]:
    if out.shape != ref.shape:
        return False, float("inf")
    err = max_rel_err(out, ref)
    return err <= TOLERANCES.get(ref.dtype, 1e-2), err


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Callable[[], object], device="cpu", warmup: int = 10, reps: int = 50) -> Timing:
    """Median wall time of `fn` in milliseconds.

    On the GPU, kernel launches are asynchronous, so a Python timer would only
    measure the launch. CUDA events are recorded on the GPU's own stream and
    measure when the work actually starts and finishes.
    """
    device = torch.device(device)
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    if device.type == "cuda":
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    else:
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)

    return Timing(statistics.median(times), min(times), max(times), reps)


def run_benchmark(
    kernels: Dict[str, Kernel],
    shapes: Iterable[Tuple[int, int, int]],
    dtype=torch.float32,
    device="cpu",
    warmup: int = 10,
    reps: int = 50,
) -> List[Result]:
    results = []
    dtype_name = str(dtype).replace("torch.", "")
    for m, n, k in shapes:
        a, b = make_inputs(m, n, k, dtype=dtype, device=device)
        ref = torch_matmul(a, b)
        for name, fn in kernels.items():
            correct, err = check_close(fn(a, b), ref)
            ms = flops = None
            if correct:
                ms = time_ms(lambda: fn(a, b), device=device, warmup=warmup, reps=reps).median_ms
                flops = tflops(m, n, k, ms)
            results.append(Result(name, m, n, k, dtype_name, correct, err, ms, flops))
    return results


def environment_info(device="cpu") -> Dict[str, str]:
    """Hardware and library versions, saved next to every result."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
    }
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = "not installed"
    if torch.device(device).type == "cuda":
        info["device"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda or "unknown"
    return info


def write_csv(results: List[Result], path: Path, env: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        for key, value in env.items():
            f.write(f"# {key}: {value}\n")
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def format_table(results: List[Result]) -> str:
    lines = [f"{'kernel':<12}{'M':>6}{'N':>6}{'K':>6}  {'dtype':<9}{'ms':>9}{'TFLOPS':>9}  {'rel err':>8}"]
    for r in results:
        ms = f"{r.median_ms:.3f}" if r.median_ms is not None else "-"
        tf = f"{r.tflops:.2f}" if r.tflops is not None else "WRONG"
        lines.append(f"{r.kernel:<12}{r.m:>6}{r.n:>6}{r.k:>6}  {r.dtype:<9}{ms:>9}{tf:>9}  {r.max_rel_err:>8.1e}")
    return "\n".join(lines)


def main(argv=None) -> List[Result]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernels", nargs="+", default=["torch"], choices=sorted(KERNELS))
    parser.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096],
                        help="square sizes: each N runs an N x N x N GEMM")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--out", type=Path, help="CSV file to write results to")
    args = parser.parse_args(argv)

    shapes = [(s, s, s) for s in args.sizes]
    kernels = {name: KERNELS[name] for name in args.kernels}
    results = run_benchmark(kernels, shapes, DTYPES[args.dtype], args.device, args.warmup, args.reps)
    env = environment_info(args.device)

    print(", ".join(f"{k}={v}" for k, v in env.items()))
    print(format_table(results))
    if args.out:
        write_csv(results, args.out, env)
        print(f"Wrote {args.out}")
    return results


if __name__ == "__main__":
    main()
