# hf-bench

Edge and local LLM `imatrix` quantization, FP16 logit KL-divergence evaluation, hardware power/throughput benchmarking (`orin-nano` CUDA `sm_87` and `nixos-nvme` `x86_64`), and interactive SvelteKit Space source for `huggingface.co/kleinbem`.

## Layout

| Path | Purpose |
|---|---|
| `flake.nix` | Builds `.#space` (static SvelteKit Space) and `.#hf-bench` (orchestrator CLI). |
| `nix/space.nix` | `buildNpmPackage` for the SvelteKit Space, injecting HF `sdk: static` metadata. |
| `nix/cli.nix` | Wraps Python + `llama-cpp` + `convert_hf_to_gguf.py` into `hf-bench`. |
| `python/hf_bench/` | Calibration corpus, GGUF quantization, KL-divergence, SSH Orin bench, Parquet/HF publisher. |
| `python/tests/` | Unit tests for the Python pipeline, exporters, and memory sizing math. |
| `src/` | Svelte 5 + Tailwind CSS 4 Space (`@kleinbem/web-config`) for Pareto and 8 GiB KV-cache sizing. |
| `static/data/` | Committed `benchmarks.json` powering the Space UI; Parquet exports live in `~/.cache/hf-bench`. |

## Operational Rules

- **Storage**: Never place GGUF weights, logits, or safetensors in `/tmp` (8 GiB tmpfs). All artifacts live under `~/.cache/hf-bench` locally and `/persist/cache/hf-bench` on `orin-nano`.
- **Priority on `nixos-nvme`**: Local `llama-imatrix`, `llama-quantize`, `llama-perplexity`, and `llama-bench` run under `chrt --idle 0 ionice -c3` with `-t 6`.
- **`orin-nano` GPU window**: Touch `/run/mesh-llm-build-window/start` before benchmarking so `mesh-llm.service` yields unified memory, and always touch `/run/mesh-llm-build-window/done` on exit. Build `llama-cpp` on Orin with `cudaPackages_12_6` and `cudaCapabilities = [ "8.7" ]`.
- **Browser support**: Modern evergreen browsers; follow `@kleinbem/web-config` strict Svelte 5 runes mode.
