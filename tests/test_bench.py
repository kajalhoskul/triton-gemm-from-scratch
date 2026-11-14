"""Tests for the benchmark harness. They run on CPU, so no GPU is needed."""
import csv

import pytest
import torch

from gemm import bench


def test_gemm_flops_counts_multiply_and_add():
    # 2 x 3 output, each needing 4 multiplies and 4 adds.
    assert bench.gemm_flops(2, 3, 4) == 2 * 2 * 3 * 4


def test_tflops_conversion():
    # A 1000^3 GEMM is 2e9 FLOPs; in 1 ms that is 2e12 FLOP/s = 2 TFLOPS.
    assert bench.tflops(1000, 1000, 1000, ms=1.0) == pytest.approx(2.0)


def test_make_inputs_shapes_dtype_and_seed():
    a, b = bench.make_inputs(4, 5, 6, dtype=torch.float16)
    assert a.shape == (4, 6) and b.shape == (6, 5)
    assert a.dtype == b.dtype == torch.float16
    a2, _ = bench.make_inputs(4, 5, 6, dtype=torch.float16)
    assert torch.equal(a, a2)  # same seed, same data
    a3, _ = bench.make_inputs(4, 5, 6, dtype=torch.float16, seed=1)
    assert not torch.equal(a, a3)


def test_check_close_accepts_exact_output():
    ref = torch.randn(8, 8)
    ok, err = bench.check_close(ref.clone(), ref)
    assert ok and err == 0


def test_check_close_rejects_wrong_output():
    ref = torch.randn(8, 8)
    out = ref.clone()
    out[3, 3] += 10 * ref.abs().max()
    ok, err = bench.check_close(out, ref)
    assert not ok and err > 1


def test_check_close_rejects_wrong_shape():
    ok, _ = bench.check_close(torch.zeros(4, 4), torch.zeros(4, 5))
    assert not ok


def test_check_close_allows_small_rounding_error():
    ref = torch.randn(64, 64)
    out = ref * (1 + 1e-4)
    assert bench.check_close(out, ref)[0]


def test_time_ms_runs_warmup_and_reps():
    calls = []
    timing = bench.time_ms(lambda: calls.append(1), warmup=3, reps=7)
    assert len(calls) == 10
    assert timing.reps == 7
    assert 0 <= timing.min_ms <= timing.median_ms <= timing.max_ms


def test_benchmark_reference_is_correct_and_timed():
    results = bench.run_benchmark({"torch": torch.matmul}, [(32, 48, 16)], warmup=1, reps=3)
    (r,) = results
    assert r.correct and r.max_rel_err < 1e-5
    assert r.median_ms > 0 and r.tflops > 0


def test_benchmark_does_not_time_a_wrong_kernel():
    def broken(a, b):
        return torch.matmul(a, b) + 1.0

    (r,) = bench.run_benchmark({"broken": broken}, [(16, 16, 16)], warmup=1, reps=3)
    assert not r.correct
    assert r.median_ms is None and r.tflops is None


def test_cli_writes_csv_with_environment(tmp_path):
    out = tmp_path / "results.csv"
    bench.main(["--kernels", "torch", "--dtype", "float32", "--sizes", "16", "32",
                "--device", "cpu", "--warmup", "1", "--reps", "2", "--out", str(out)])
    lines = out.read_text().splitlines()
    comments = [l for l in lines if l.startswith("#")]
    assert any(l.startswith("# torch:") for l in comments)
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    assert [int(r["m"]) for r in rows] == [16, 32]
    assert all(r["correct"] == "True" for r in rows)
