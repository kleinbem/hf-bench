"""Export Parquet datasets, Space static JSON, and Hugging Face Model/Dataset Cards."""

from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from hf_bench.models import MODELS, QUANT_MATRIX, ModelSpec


def collect_records(cache_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models_root = cache_dir / "models"
    if not models_root.exists():
        return rows
    for bench_file in sorted(models_root.glob("*/bench/*.json")):
        rows.append(json.loads(bench_file.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: (str(r["model_slug"]), str(r["hardware_id"]), -float(r["bpw"])))
    return rows


def render_model_card(spec: ModelSpec, records: list[dict[str, Any]]) -> str:
    model_rows = [r for r in records if r["model_slug"] == spec.slug]
    by_hw_quant = {(r["hardware_id"], r["quant"]): r for r in model_rows}

    table_lines = [
        "| Quant | `imatrix` | BPW | Size (GiB) | Mean KLD (nats) | Top-1 Match | PPL | Orin CUDA `tg128` | Orin Power / Energy | x86 AVX2 `tg128` |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for qspec in QUANT_MATRIX:
        orin = by_hw_quant.get(("orin-nano", qspec.name))
        x86 = by_hw_quant.get(("nixos-nvme", qspec.name))
        ref = orin or x86
        if not ref:
            continue
        imat = "Yes" if qspec.use_imatrix else "No *(control)*"
        kld = f"{ref['mean_kld']:.5f}" if ref.get("mean_kld") is not None else "—"
        top1 = f"{ref['same_top1_pct']:.2f}%" if ref.get("same_top1_pct") is not None else "—"
        ppl = f"{ref['ppl']:.3f}" if ref.get("ppl") is not None else "—"

        if orin is None:
            orin_ts = "—"
            orin_pwr = "—"
        elif orin["status"] == "oom":
            orin_ts = "**OOM (>8 GiB)**"
            orin_pwr = "—"
        else:
            orin_ts = f"{orin['tg128_tok_s']:.1f} tok/s"
            if orin.get("avg_watts") and orin.get("joules_per_token"):
                orin_pwr = f"{orin['avg_watts']:.1f} W ({orin['joules_per_token']:.3f} J/tok)"
            else:
                orin_pwr = "—"

        x86_ts = f"{x86['tg128_tok_s']:.1f} tok/s" if (x86 and x86["status"] == "ok") else "—"
        table_lines.append(
            f"| `{qspec.name}` | {imat} | {ref['bpw']:.2f} | {ref['weight_gib']:.2f} | "
            f"{kld} | {top1} | {ppl} | {orin_ts} | {orin_pwr} | {x86_ts} |"
        )

    ablation_note = ""
    q4_im = by_hw_quant.get(("nixos-nvme", "Q4_K_M")) or by_hw_quant.get(("orin-nano", "Q4_K_M"))
    q4_std = by_hw_quant.get(("nixos-nvme", "Q4_K_M-std")) or by_hw_quant.get(("orin-nano", "Q4_K_M-std"))
    if q4_im and q4_std and q4_im.get("mean_kld") is not None and q4_std.get("mean_kld"):
        reduction = (1.0 - (float(q4_im["mean_kld"]) / float(q4_std["mean_kld"]))) * 100.0
        ablation_note = (
            f"\n> **`imatrix` Ablation (`Q4_K_M` vs `Q4_K_M-std`)**: Calibrating second-moment channel "
            f"activations on agentic tool-calling + systems code changed Mean KL-Divergence from "
            f"`{q4_std['mean_kld']:.5f}` to `{q4_im['mean_kld']:.5f}` (**{reduction:+.1f}% KLD reduction** at "
            f"identical byte size).\n"
        )

    table_md = "\n".join(table_lines)
    return f"""---
license: {spec.license_id}
base_model: {spec.hf_source}
base_model_relation: quantized
library_name: gguf
quantized_by: kleinbem
pipeline_tag: text-generation
tags:
  - gguf
  - llama-cpp
  - imatrix
  - edge-ai
  - jetson-orin-nano
  - nixos
  - quantization
---

# {spec.display_name} — Edge-Calibrated `imatrix` GGUF Suite

Importance-matrix (`llama-imatrix`) GGUF quantizations of [`{spec.hf_source}`](https://huggingface.co/{spec.hf_source}), calibrated specifically for **agentic tool-calling, code review, and declarative infrastructure reasoning** and benchmarked across **8 GiB NVIDIA Jetson Orin Nano (CUDA 12.6 `sm_87`)** and **x86_64 AVX2**.

- **Interactive Pareto & 8 GiB Edge KV-Cache Explorer**: [`spaces/kleinbem/edge-llm-pareto`](https://huggingface.co/spaces/kleinbem/edge-llm-pareto)
- **Full Parquet Telemetry Dataset**: [`datasets/kleinbem/edge-llm-benchmarks`](https://huggingface.co/datasets/kleinbem/edge-llm-benchmarks)
- **Reproducible Nix Flake & Pipeline Source**: [`github.com/kleinbem/hf-bench`](https://github.com/kleinbem/hf-bench)

## Quality (FP16 Logit KL-Divergence) & Hardware Telemetry

Unlike perplexity-only tables, every quantization below was scored directly against the exact **FP16 base model probability distribution** (`llama-perplexity --kl-divergence`), alongside real hardware throughput and INA3221/RAPL power telemetry:

{table_md}
{ablation_note}
## GQA KV-Cache Sizing Formula ({spec.display_name})

`{spec.display_name}` uses Grouped-Query Attention with `{spec.num_layers}` layers, `{spec.num_kv_heads}` KV heads, and `head_dim = {spec.head_dim}` (`{spec.kv_bytes_per_token(16.0) / 1024.0:.1f} KiB/token` in FP16 KV cache, or `{spec.kv_mib(8192, 16.0):.0f} MiB` at `8,192` context). Use the interactive [Edge Unified-Memory Sizer](https://huggingface.co/spaces/kleinbem/edge-llm-pareto) to verify exact headroom for your context length and KV quantization (`FP16` / `Q8_0` / `Q4_0`).

## Reproducing with Nix

```bash
nix run github:kleinbem/hf-bench#hf-bench -- run-local --model {spec.slug}
nix run github:kleinbem/hf-bench#hf-bench -- bench-orin --model {spec.slug} --host orin-nano
```
"""


def render_dataset_card(records: list[dict[str, Any]]) -> str:
    models_seen = sorted({str(r["hf_source"]) for r in records})
    models_yaml = "\n".join(f"  - {m}" for m in models_seen) if models_seen else "  - Qwen/Qwen3-4B"
    return f"""---
license: mit
task_categories:
  - text-generation
tags:
  - benchmark
  - quantization
  - kl-divergence
  - gguf
  - llama-cpp
  - jetson-orin-nano
  - energy-efficiency
  - edge-ai
pretty_name: Edge & Local GGUF Quantization, KL-Divergence & Power Telemetry
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/benchmarks.parquet
---

# Edge & Local GGUF Quantization, KL-Divergence & Power Telemetry

Hardware throughput (`pp512`, `tg128`), exact **FP16 logit Kullback-Leibler divergence (`mean_kld`, `p99_kld`, `same_top1_pct`)**, unified-memory OOM boundaries, and **INA3221 / RAPL energy per token (`Joules/token`)** collected reproducibly via [`github.com/kleinbem/hf-bench`](https://github.com/kleinbem/hf-bench).

- **Interactive Explorer Space**: [`spaces/kleinbem/edge-llm-pareto`](https://huggingface.co/spaces/kleinbem/edge-llm-pareto)

## Evaluated Base Models

```yaml
base_models:
{models_yaml}
```

## Hardware Targets

1. **`orin-nano`**: NVIDIA Jetson Orin Nano (8 GiB unified LPDDR5, Ampere 1024-core Tegra GPU `sm_87`, NixOS + JetPack 6 CUDA 12.6, INA3221 hardware power rail sampling).
2. **`nixos-nvme`**: 13th Gen Intel Core i3-1315U (`x86_64` Raptor Lake AVX2, 62 GiB DDR5, NixOS idle scheduler priority, Intel RAPL package energy sampling).

## Schema Highlights

| Column | Type | Description |
|---|---|---|
| `model_name` | `string` | Base model architecture name |
| `quant` | `string` | `llama.cpp` quantization (`Q8_0` .. `IQ3_M`, plus `Q4_K_M-std` uncalibrated control) |
| `imatrix_used` | `bool` | Whether agentic + code importance matrix calibration was applied |
| `bpw` | `float64` | Effective bits per weight (`weight_bytes * 8 / params`) |
| `status` | `string` | `ok` or `oom` when exceeding the device's unified memory cliff |
| `mean_kld` | `float64` | Mean Kullback-Leibler divergence (nats) vs exact FP16 base logits |
| `same_top1_pct` | `float64` | Percentage of evaluation tokens sharing the exact FP16 argmax prediction |
| `pp512_tok_s` | `float64` | 512-token prompt prefill speed (`tokens/sec`) |
| `tg128_tok_s` | `float64` | 128-token autoregressive decode speed (`tokens/sec`) |
| `avg_watts` | `float64` | Mean board/package power draw during inference (`Watts`) |
| `joules_per_token` | `float64` | Energy consumed per generated token (`avg_watts / tg128_tok_s`) |
"""


def export_all(cache_dir: Path, repo_root: Path) -> Path:
    records = collect_records(cache_dir)
    export_dir = cache_dir / "exports"
    (export_dir / "dataset" / "data").mkdir(parents=True, exist_ok=True)
    (export_dir / "model_cards").mkdir(parents=True, exist_ok=True)

    if records:
        table = pa.Table.from_pylist(records)
        parquet_path = export_dir / "dataset" / "data" / "benchmarks.parquet"
        pq.write_table(table, parquet_path, compression="zstd")

        space_payload = {
            "generatedAt": datetime.now(UTC).isoformat(),
            "sourceRepo": "https://github.com/kleinbem/hf-bench",
            "datasetRepo": "https://huggingface.co/datasets/kleinbem/edge-llm-benchmarks",
            "models": [asdict(m) for m in MODELS.values()],
            "records": records,
        }
        space_json = repo_root / "static" / "data" / "benchmarks.json"
        space_json.parent.mkdir(parents=True, exist_ok=True)
        space_json.write_text(json.dumps(space_payload, indent=2) + "\n", encoding="utf-8")

    ds_readme = export_dir / "dataset" / "README.md"
    ds_readme.write_text(render_dataset_card(records), encoding="utf-8")

    for slug, spec in MODELS.items():
        if any(r["model_slug"] == slug for r in records):
            card_path = export_dir / "model_cards" / f"{slug}-README.md"
            card_path.write_text(render_model_card(spec, records), encoding="utf-8")

    return export_dir
