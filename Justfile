set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Run full Nix flake check (unit tests, Svelte check, static Space build + Caddy smoke test)
check:
    nix flake check -L

# Start Vite dev server for the SvelteKit Space
dev:
    npm run dev

# Quantize, compute KL-divergence, and benchmark locally on nixos-nvme
run-local model="qwen3-4b":
    nix run .#hf-bench -- run-local --model "{{model}}"

# Benchmark quantized GGUFs on Jetson Orin Nano (CUDA 13 sm_87 + INA3221 power)
bench-orin model="qwen3-4b" host="orin-nano":
    nix run .#hf-bench -- bench-orin --model "{{model}}" --host "{{host}}"

# Export Parquet dataset, Space static JSON, and HF cards
export:
    nix run .#hf-bench -- export --repo-root .

# Publish Model(s), Parquet Dataset, and Static Space to Hugging Face
publish:
    #!/usr/bin/env bash
    set -euo pipefail
    space_out=$(nix build --no-link --print-out-paths .#space)
    nix run .#hf-bench -- publish --repo-root . --space-build "$space_out"
