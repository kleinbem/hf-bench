"""Unit tests for parsers, KV-cache math, corpus generation, and card/Parquet exports."""

import json
from pathlib import Path
import tempfile
import unittest

from hf_bench.corpus import write_corpora
from hf_bench.export import export_all, render_model_card
from hf_bench.models import MODELS
from hf_bench.parsers import parse_llama_bench_json, parse_perplexity_output
from hf_bench.pipeline import ORIN_RUNNER_PY

SAMPLE_PPL_LOG = """
Final estimate: PPL = 6.8421 +/- 0.04123

====== KL divergence statistics ======
Mean    KLD:   0.018420 ±   0.000512
Maximum KLD:   1.942100
99.9%   KLD:   0.482100
99.0%   KLD:   0.194200
Median  KLD:   0.006310
Minimum KLD:  -0.000012

====== Token probability statistics ======
RMS Δp    :  3.412 ± 0.045 %
Same top p: 95.840 ± 0.182 %
"""

SAMPLE_BENCH_JSON = """
[
  {
    "cpu_info": "13th Gen Intel(R) Core(TM) i3-1315U",
    "gpu_info": "",
    "backends": "CPU",
    "model_size": 2498123776,
    "n_params": 4022468096,
    "n_prompt": 512,
    "n_gen": 0,
    "avg_ts": 84.52,
    "stddev_ts": 1.14
  },
  {
    "cpu_info": "13th Gen Intel(R) Core(TM) i3-1315U",
    "gpu_info": "",
    "backends": "CPU",
    "model_size": 2498123776,
    "n_params": 4022468096,
    "n_prompt": 0,
    "n_gen": 128,
    "avg_ts": 16.84,
    "stddev_ts": 0.21
  }
]
"""


class HfBenchTests(unittest.TestCase):
    def test_qwen3_kv_cache_math(self) -> None:
        qwen3 = MODELS["qwen3-4b"]
        # 2 * 36 * 8 * 128 * 2 bytes = 147,456 bytes/token = 144 KiB/token -> 1152 MiB at 8192 ctx.
        self.assertAlmostEqual(qwen3.kv_bytes_per_token(16.0), 147456.0)
        self.assertAlmostEqual(qwen3.kv_mib(8192, 16.0), 1152.0)
        self.assertAlmostEqual(qwen3.kv_mib(8192, 8.0), 576.0)

    def test_parse_perplexity_and_kld(self) -> None:
        m = parse_perplexity_output(SAMPLE_PPL_LOG)
        self.assertAlmostEqual(m.ppl, 6.8421)
        self.assertAlmostEqual(m.mean_kld or 0.0, 0.01842)
        self.assertAlmostEqual(m.p99_kld or 0.0, 0.1942)
        self.assertAlmostEqual(m.rms_delta_p_pct or 0.0, 3.412)
        self.assertAlmostEqual(m.same_top1_pct or 0.0, 95.84)

    def test_parse_llama_bench_json(self) -> None:
        tp = parse_llama_bench_json("noise before\n" + SAMPLE_BENCH_JSON)
        self.assertAlmostEqual(tp.pp512_tok_s, 84.52)
        self.assertAlmostEqual(tp.tg128_tok_s, 16.84)
        self.assertEqual(tp.model_size_bytes, 2498123776)

    def test_corpora_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib, ev = write_corpora(root / "corpora")
            self.assertGreater(calib.stat().st_size, 20000)
            self.assertGreater(ev.stat().st_size, 8000)

            bench_dir = root / "cache" / "models" / "qwen3-4b" / "bench"
            bench_dir.mkdir(parents=True)
            rec = {
                "record_id": "qwen3-4b__orin-nano__Q4_K_M",
                "timestamp_utc": "2026-11-13T19:00:00+00:00",
                "model_slug": "qwen3-4b",
                "model_name": "Qwen3-4B",
                "hf_source": "Qwen/Qwen3-4B",
                "hf_target_repo": "kleinbem/Qwen3-4B-Edge-GGUF",
                "params_b": 4.02,
                "quant": "Q4_K_M",
                "imatrix_used": True,
                "bpw": 4.97,
                "weight_bytes": 2498123776,
                "weight_gib": 2.327,
                "hardware_id": "orin-nano",
                "hardware_label": "Jetson Orin Nano",
                "backend": "CUDA 12.6",
                "memory_budget_gib": 7.6,
                "status": "ok",
                "oom_reason": None,
                "pp512_tok_s": 245.1,
                "pp512_std": 2.0,
                "tg128_tok_s": 28.4,
                "tg128_std": 0.3,
                "ppl": 6.84,
                "ppl_base_fp16": 6.71,
                "mean_kld": 0.01842,
                "p99_kld": 0.1942,
                "median_kld": 0.00631,
                "max_kld": 1.94,
                "rms_delta_p_pct": 3.41,
                "same_top1_pct": 95.84,
                "avg_watts": 11.2,
                "joules_per_token": 0.3944,
                "peak_mem_mib": 2920.0,
            }
            (bench_dir / "orin-nano-Q4_K_M.json").write_text(json.dumps(rec), encoding="utf-8")
            repo_dir = root / "repo"
            export_dir = export_all(root / "cache", repo_dir)
            self.assertTrue((export_dir / "dataset" / "data" / "benchmarks.parquet").exists())
            self.assertTrue((repo_dir / "static" / "data" / "benchmarks.json").exists())
            card = render_model_card(MODELS["qwen3-4b"], [rec])
            self.assertIn("28.4 tok/s", card)
            self.assertIn("0.394 J/tok", card)

    def test_orin_runner_script_compiles(self) -> None:
        code = compile(ORIN_RUNNER_PY, "<orin_runner>", "exec")
        self.assertIsNotNone(code)
        self.assertIn("ina3221", ORIN_RUNNER_PY)


if __name__ == "__main__":
    unittest.main()
