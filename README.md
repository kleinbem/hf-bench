# hf-bench

Reproducible `llama.cpp` importance-matrix (`imatrix`) quantization, exact FP16 logit KL-divergence evaluation, hardware power/throughput telemetry, and interactive Pareto + edge unified-memory sizing explorer for [`huggingface.co/kleinbem`](https://huggingface.co/kleinbem).

## What It Produces

1. **Calibrated Edge GGUF Suites (`kleinbem/*-Edge-GGUF`)**: Quantized from BF16/FP16 safetensors (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, `IQ4_XS`, `IQ3_M`) using a deterministic agentic tool-calling + systems code calibration corpus, including an uncalibrated `Q4_K_M` control to quantify `imatrix` KL-divergence reduction.
2. **Hardware & Quality Telemetry Dataset (`kleinbem/edge-llm-benchmarks`)**: Parquet + JSON records combining FP16 logit KL-divergence (`mean_kld`, `max_kld`, `ppl`, `top1_agreement`) with hardware decode/prefill throughput, unified memory footprint, and power efficiency (`Watts`, `Joules/token`) across NVIDIA Jetson Orin Nano (`sm_87` CUDA 13, 8 GiB unified RAM) and Intel Raptor Lake (`x86_64` AVX2).
3. **Interactive Static Space (`spaces/kleinbem/edge-llm-pareto`)**: Svelte 5 + Tailwind CSS 4 explorer built via Nix (`buildNpmPackage`), featuring a real-time Pareto frontier plot and an 8 GiB Edge GQA KV-cache calculator.

## Quickstart

```bash
# Build and verify both the Static Space and CLI via Nix
nix flake check -L

# Run the full local quantization + FP16 KL-divergence + x86_64 benchmark for Qwen3-4B
nix run .#hf-bench -- run-local --model qwen3-4b

# Run the CUDA sm_87 + INA3221 power benchmark on Jetson Orin Nano
nix run .#hf-bench -- bench-orin --model qwen3-4b --host orin-nano

# Export Parquet dataset, Space JSON, and Model/Dataset Cards
nix run .#hf-bench -- export --repo-root .

# Publish Model(s), Dataset, and compiled Nix Static Space to Hugging Face Hub
nix run .#hf-bench -- publish --repo-root . --space-build "$(nix build --no-link --print-out-paths .#space)"
```

## License

MIT
