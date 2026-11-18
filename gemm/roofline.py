"""Roofline analysis: how close is a kernel to what the GPU can do at best?

Author: Kajal Changappa

A kernel is limited by one of two ceilings:

* compute:  the GPU's peak FLOP rate, and
* memory:   peak memory bandwidth x arithmetic intensity, where arithmetic
            intensity = FLOPs / bytes that must move to or from memory.

The attainable speed is the lower of the two. Even a perfect GEMM has to read
A and B once and write C once, so its minimum traffic is
(M*K + K*N + M*N) * bytes per value. Comparing a measured TFLOPS figure with
the attainable one says how much is left to gain, and which ceiling is in
the way.

Usage, on a CSV written by `python -m gemm.bench --out ...`:

    python -m gemm.roofline results/t4_fp16.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GpuSpec:
    name: str
    fp32_tflops: float          # ordinary FP32 (CUDA cores)
    fp16_tensor_tflops: float   # FP16 inputs on Tensor Cores
    mem_gbps: float             # peak memory bandwidth, GB/s

    def peak_tflops(self, dtype: str) -> float:
        return self.fp16_tensor_tflops if dtype in ("float16", "bfloat16") else self.fp32_tflops


# Peak figures from NVIDIA's published specifications.
GPUS: Dict[str, GpuSpec] = {
    "Tesla T4": GpuSpec("Tesla T4", fp32_tflops=8.1, fp16_tensor_tflops=65.0, mem_gbps=320.0),
}

BYTES_PER_VALUE = {"float32": 4, "float16": 2, "bfloat16": 2}


@dataclass
class Analysis:
    kernel: str
    m: int
    n: int
    k: int
    dtype: str
    measured_tflops: float
    arithmetic_intensity: float  # FLOPs per byte, at minimum traffic
    attainable_tflops: float
    bound: str                   # "compute" or "memory"
    pct_of_attainable: float


def min_bytes(m: int, n: int, k: int, dtype: str) -> int:
    """Bytes a perfect GEMM must move: read A and B once, write C once."""
    return (m * k + k * n + m * n) * BYTES_PER_VALUE[dtype]


def arithmetic_intensity(m: int, n: int, k: int, dtype: str) -> float:
    return 2 * m * n * k / min_bytes(m, n, k, dtype)


def attainable_tflops(intensity: float, peak_tflops: float, mem_gbps: float) -> float:
    """The roofline: min(compute ceiling, memory ceiling)."""
    memory_ceiling = intensity * mem_gbps / 1000.0  # GB/s x FLOP/B = GFLOP/s -> TFLOP/s
    return min(peak_tflops, memory_ceiling)


def analyze(kernel: str, m: int, n: int, k: int, dtype: str, measured_tflops: float, gpu: GpuSpec) -> Analysis:
    ai = arithmetic_intensity(m, n, k, dtype)
    peak = gpu.peak_tflops(dtype)
    attainable = attainable_tflops(ai, peak, gpu.mem_gbps)
    bound = "compute" if attainable >= peak else "memory"
    return Analysis(kernel, m, n, k, dtype, measured_tflops, ai, attainable, bound,
                    100.0 * measured_tflops / attainable)


def read_results(path: Path):
    """Rows and header comments (# key: value) from a gemm.bench CSV."""
    env, lines = {}, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            env[key.strip()] = value.strip()
        else:
            lines.append(line)
    return env, list(csv.DictReader(lines))


def analyze_file(path: Path, gpu: Optional[GpuSpec] = None) -> List[Analysis]:
    env, rows = read_results(path)
    if gpu is None:
        device = env.get("device", "")
        gpu = next((spec for name, spec in GPUS.items() if name in device), None)
        if gpu is None:
            raise ValueError(f"no GPU spec for device {device!r}; known: {sorted(GPUS)}")
    out = []
    for r in rows:
        if r["correct"] != "True" or not r["tflops"]:
            continue  # wrong kernels have no speed to analyze
        out.append(analyze(r["kernel"], int(r["m"]), int(r["n"]), int(r["k"]), r["dtype"],
                           float(r["tflops"]), gpu))
    return out


def format_table(results: List[Analysis]) -> str:
    lines = [f"{'kernel':<12}{'size':>16}  {'dtype':<9}{'FLOP/B':>8}{'measured':>10}{'ceiling':>9}  {'bound':<8}{'% of ceiling':>13}"]
    for a in results:
        size = f"{a.m}x{a.n}x{a.k}"
        lines.append(f"{a.kernel:<12}{size:>16}  {a.dtype:<9}{a.arithmetic_intensity:>8.0f}"
                     f"{a.measured_tflops:>10.2f}{a.attainable_tflops:>9.1f}  {a.bound:<8}{a.pct_of_attainable:>12.1f}%")
    return "\n".join(lines)


def main(argv=None) -> List[Analysis]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="+", help="result files from gemm.bench --out")
    args = parser.parse_args(argv)
    results = []
    for path in args.csv:
        results += analyze_file(path)
    print(format_table(results))
    return results


if __name__ == "__main__":
    main()
