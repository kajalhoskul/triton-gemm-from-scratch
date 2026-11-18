"""Tests for the roofline model in gemm/roofline.py. Pure Python, runs everywhere.

Author: Kajal Changappa
"""
import pytest

from gemm import bench, roofline
from gemm.roofline import GPUS, GpuSpec

T4 = GPUS["Tesla T4"]


def test_min_bytes_reads_a_and_b_and_writes_c_once():
    # FP16: (2*4 + 4*3 + 2*3) values * 2 bytes.
    assert roofline.min_bytes(2, 3, 4, "float16") == (8 + 12 + 6) * 2


def test_intensity_of_square_gemm():
    # N x N x N: 2N^3 FLOPs over 3N^2 values, so N*2/(3*bytes).
    n = 4096
    assert roofline.arithmetic_intensity(n, n, n, "float16") == pytest.approx(2 * n / (3 * 2))
    assert roofline.arithmetic_intensity(n, n, n, "float32") == pytest.approx(2 * n / (3 * 4))


def test_fp16_doubles_intensity_over_fp32():
    assert roofline.arithmetic_intensity(512, 512, 512, "float16") == pytest.approx(
        2 * roofline.arithmetic_intensity(512, 512, 512, "float32"))


def test_peak_depends_on_dtype():
    assert T4.peak_tflops("float16") == 65.0
    assert T4.peak_tflops("float32") == 8.1


def test_low_intensity_is_memory_bound():
    # 10 FLOP/B x 320 GB/s = 3.2 TFLOPS, far below the FP16 peak.
    assert roofline.attainable_tflops(10, 65.0, 320.0) == pytest.approx(3.2)


def test_high_intensity_is_compute_bound():
    assert roofline.attainable_tflops(10_000, 65.0, 320.0) == 65.0


def test_skinny_gemm_is_memory_bound_on_t4():
    a = roofline.analyze("x", 4096, 4096, 16, "float16", measured_tflops=1.0, gpu=T4)
    assert a.bound == "memory"
    assert a.attainable_tflops < T4.peak_tflops("float16")


def test_large_square_fp16_gemm_is_compute_bound_on_t4():
    a = roofline.analyze("x", 4096, 4096, 4096, "float16", measured_tflops=32.5, gpu=T4)
    assert a.bound == "compute"
    assert a.attainable_tflops == 65.0
    assert a.pct_of_attainable == pytest.approx(50.0)


def test_analyze_file_reads_bench_csv(tmp_path):
    results = [
        bench.Result("fast", 1024, 1024, 1024, "float16", True, 1e-4, 0.1, 21.47),
        bench.Result("broken", 1024, 1024, 1024, "float16", False, 9.9, None, None),
    ]
    path = tmp_path / "r.csv"
    bench.write_csv(results, path, {"device": "Tesla T4", "torch": "x"})
    (a,) = roofline.analyze_file(path)  # the wrong kernel is skipped
    assert a.kernel == "fast" and a.measured_tflops == pytest.approx(21.47)
    assert a.attainable_tflops == pytest.approx(min(65.0, roofline.arithmetic_intensity(1024, 1024, 1024, "float16") * 0.32))


def test_analyze_file_rejects_unknown_gpu(tmp_path):
    path = tmp_path / "r.csv"
    bench.write_csv([bench.Result("t", 8, 8, 8, "float32", True, 0, 1.0, 1.0)], path, {"device": "cpu"})
    with pytest.raises(ValueError, match="no GPU spec"):
        roofline.analyze_file(path)
    custom = GpuSpec("test", fp32_tflops=1.0, fp16_tensor_tflops=2.0, mem_gbps=100.0)
    assert len(roofline.analyze_file(path, gpu=custom)) == 1


def test_cli_prints_table(tmp_path, capsys):
    path = tmp_path / "r.csv"
    bench.write_csv([bench.Result("fast", 2048, 2048, 2048, "float16", True, 1e-4, 1.0, 17.2)], path,
                    {"device": "Tesla T4"})
    roofline.main([str(path)])
    out = capsys.readouterr().out
    assert "fast" in out and "2048x2048x2048" in out and "compute" in out
