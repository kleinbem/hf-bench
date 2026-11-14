"""Parsers for llama-perplexity KL-divergence output and llama-bench JSON."""

from dataclasses import dataclass
import json
import re
from typing import Any

PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)")
MEAN_PPL_Q_RE = re.compile(r"Mean\s+PPL\(Q\)\s*:\s*([0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)")
MEAN_KLD_RE = re.compile(r"Mean\s+KLD:\s*([-0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)")
MAX_KLD_RE = re.compile(r"Maximum\s+KLD:\s*([-0-9.]+)")
P99_KLD_RE = re.compile(r"99\.0%\s+KLD:\s*([-0-9.]+)")
MEDIAN_KLD_RE = re.compile(r"Median\s+KLD:\s*([-0-9.]+)")
RMS_DP_RE = re.compile(r"RMS\s+Δp\s*:\s*([-0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)\s*%")
SAME_TOP_RE = re.compile(r"Same top p:\s*([0-9.]+)\s*(?:\+/-|±)\s*([0-9.]+)\s*%")


@dataclass(frozen=True)
class PerplexityMetrics:
    ppl: float
    ppl_std: float
    mean_kld: float | None = None
    mean_kld_std: float | None = None
    p99_kld: float | None = None
    median_kld: float | None = None
    max_kld: float | None = None
    rms_delta_p_pct: float | None = None
    same_top1_pct: float | None = None


@dataclass(frozen=True)
class BenchThroughput:
    pp512_tok_s: float
    pp512_std: float
    tg128_tok_s: float
    tg128_std: float
    model_size_bytes: int
    n_params: int
    backend_desc: str


def _match_float(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    m = pattern.search(text)
    return float(m.group(group)) if m else None


def parse_perplexity_output(log_text: str) -> PerplexityMetrics:
    ppl_match = PPL_RE.search(log_text) or MEAN_PPL_Q_RE.search(log_text)
    if not ppl_match:
        raise ValueError("Could not find 'Final estimate: PPL =' or 'Mean PPL(Q) :' in llama-perplexity output")
    return PerplexityMetrics(
        ppl=float(ppl_match.group(1)),
        ppl_std=float(ppl_match.group(2)),
        mean_kld=_match_float(MEAN_KLD_RE, log_text, 1),
        mean_kld_std=_match_float(MEAN_KLD_RE, log_text, 2),
        p99_kld=_match_float(P99_KLD_RE, log_text, 1),
        median_kld=_match_float(MEDIAN_KLD_RE, log_text, 1),
        max_kld=_match_float(MAX_KLD_RE, log_text, 1),
        rms_delta_p_pct=_match_float(RMS_DP_RE, log_text, 1),
        same_top1_pct=_match_float(SAME_TOP_RE, log_text, 1),
    )


def parse_llama_bench_json(raw_json: str) -> BenchThroughput:
    # llama-bench may emit backend loader lines to stderr; extract the JSON array cleanly.
    start = raw_json.find("[")
    end = raw_json.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in llama-bench stdout")
    entries: list[dict[str, Any]] = json.loads(raw_json[start : end + 1])
    if not entries:
        raise ValueError("Empty JSON array from llama-bench")

    pp_ts = 0.0
    pp_std = 0.0
    tg_ts = 0.0
    tg_std = 0.0
    size_bytes = 0
    n_params = 0
    backend_desc = ""

    for row in entries:
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        size_bytes = int(row.get("model_size", size_bytes))
        n_params = int(row.get("n_params", n_params))
        backend_desc = str(row.get("gpu_info") or row.get("cpu_info") or row.get("backends") or backend_desc)
        if n_prompt > 0 and n_gen == 0:
            pp_ts = float(row.get("avg_ts", 0.0))
            pp_std = float(row.get("stddev_ts", 0.0))
        elif n_gen > 0 and n_prompt == 0:
            tg_ts = float(row.get("avg_ts", 0.0))
            tg_std = float(row.get("stddev_ts", 0.0))

    return BenchThroughput(
        pp512_tok_s=pp_ts,
        pp512_std=pp_std,
        tg128_tok_s=tg_ts,
        tg128_std=tg_std,
        model_size_bytes=size_bytes,
        n_params=n_params,
        backend_desc=backend_desc,
    )
