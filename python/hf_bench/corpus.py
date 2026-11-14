"""Deterministic agentic, code-review, and systems calibration corpus."""

from pathlib import Path

SECTIONS: tuple[str, ...] = (
    """# Systems & Declarative Infrastructure Reasoning
In an immutable Linux deployment where `/` is a tmpfs wiped on every reboot, persistent state must
be explicitly bound into `/persist` or `/data` via systemd mount units. When packaging CUDA-enabled
inference runtimes for embedded Tegra devices (NVIDIA Jetson Orin Nano, Ampere sm_87, 8 GiB unified
LPDDR5), system memory is shared between the kernel, page cache, CUDA context (~420 MiB), model
weights, KV cache, and compute buffers. Unlike discrete PCIe GPUs with isolated GDDR6, unified
memory contention triggers kernel reclaim or OOM kills unless background inference daemons yield
their allocations during build or benchmark windows.
""",
    """<|im_start|>system
You are an autonomous infrastructure agent with access to structured MCP tools. Always inspect
blast radius before mutating declarative configuration, and validate every shell or kernel boundary.
<|im_end|>
<|im_start|>user
Inspect why `mesh-llm.service` failed to load the quantized worker on `orin-nano` after a nixpkgs
bump, and propose the minimal Nix expression fix.
<|im_end|>
<|im_start|>assistant
<think>
Let's check the CUDA runtime driver compatibility first. JetPack 6 ships a CUDA 12 kernel driver;
if `nixpkgs` switched its default `cudaPackages` alias to CUDA 13, `cuInit` returns
`CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (error 803). Let's call `inspect_build_drv` to verify.
</think>
<tool_call>
{"name": "inspect_build_drv", "arguments": {"host": "orin-nano", "unit": "mesh-llm.service"}}
</tool_call>
<|im_end|>
<|im_start|>user
<tool_response>
{"cuda_version": "13.0", "driver_version": "540.4.0", "sm_targets": ["8.7"], "exit_code": 1}
</tool_response>
<|im_end|>
<|im_start|>assistant
The derivation linked against CUDA 13.0 while Tegra driver 540.4.0 supports up to CUDA 12.x. Pin
`cudaPackages = pkgs.cudaPackages_12_6` and restrict `cudaCapabilities = [ "8.7" ]`:

```nix
{ pkgs, ... }:
let
  orinPkgs = import pkgs.path {
    inherit (pkgs.stdenv.hostPlatform) system;
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "8.7" ];
    };
  };
in
orinPkgs.llama-cpp.override {
  cudaPackages = orinPkgs.cudaPackages_12_6;
}
```
<|im_end|>
""",
    """# Automated Pull Request Code Review Analysis
```diff
@@ -42,8 +42,14 @@ pub async fn run_hook(script_path: &Path, payload: &[u8]) -> Result<()> {
-    let header = std::fs::read_to_string(script_path)?;
-    if header.lines().next().unwrap_or_default().contains('\\n') {
-        bail!("invalid shebang");
-    }
+    let raw = std::fs::read(script_path)?;
+    let first_line = raw.split(|&b| b == b'\\n').next().unwrap_or_default();
+    if !first_line.starts_with(b"#!") || first_line.len() > 255 {
+        bail!("shebang missing or exceeds Linux BINPRM_BUF_SIZE (255 bytes)");
+    }
+    let interp = std::str::from_utf8(&first_line[2..])?.trim_ascii_start();
+    if interp.is_empty() || interp.contains([' ', '\\t', '\\r']) {
+        bail!("interpreter path cannot contain whitespace");
+    }
```
Review verdict: **Approve**. Linux `fs/binfmt_script.c` truncates at `BINPRM_BUF_SIZE` (256 bytes
including trailing NUL) and splits the interpreter path at the first space or tab. Checking byte
length before UTF-8 decoding prevents multi-byte characters from bypassing the kernel limit, and
rejecting `\r` prevents CRLF scripts from looking up `/bin/sh\r` (`ENOENT`).
""",
    """# Numerical Quantization & Information Theory
Post-training quantization maps floating-point tensor blocks $W \\in \\mathbb{R}^{B}$ onto compact
codebooks $\\hat{W}$ while minimizing expected output activation distortion:
$$\\mathcal{L}(W, \\hat{W}) = \\mathbb{E}_{x \\sim \\mathcal{D}} \\left\\| (W - \\hat{W}) x \\right\\|_2^2 = \\operatorname{Tr}\\left((W - \\hat{W}) H (W - \\hat{W})^\\top\\right)$$
where $H = \\mathbb{E}[x x^\\top]$ is the empirical second-moment Hessian approximated diagonally by
`llama-imatrix` ($I_j = \\sum_t x_{t,j}^2$). Without an importance matrix, uniform K-quants treat
outlier input channels identically to dormant channels, inflating Kullback-Leibler divergence
$D_{\\mathrm{KL}}(P_{\\mathrm{FP16}} \\parallel P_{Q}) = \\sum_v P_{\\mathrm{FP16}}(v) \\ln \\frac{P_{\\mathrm{FP16}}(v)}{P_Q(v)}$
particularly on structured syntax tokens (`{`, `}`, `"`, indentation, type annotations) where
logit margins are sharp. Non-linear lookup grids (`IQ4_XS`, `IQ3_M`) pair an E8 lattice or
learned codebook with super-block scales weighted by $I_j$, preserving high-salience channels down
to 3.4–4.25 bits per weight.
""",
    """// Modern Svelte 5 Runes Pareto Frontier Computation
export interface BenchmarkPoint {
  id: string;
  quant: string;
  meanKld: number;
  decodeTokPerSec: number;
  joulesPerToken: number | null;
  weightGib: number;
  oom: boolean;
}

export function computeParetoFrontier(points: readonly BenchmarkPoint[]): Set<string> {
  const viable = points.filter((p) => !p.oom && p.decodeTokPerSec > 0);
  const sorted = [...viable].sort((a, b) => a.meanKld - b.meanKld);
  const frontier = new Set<string>();
  let bestSpeed = -Infinity;
  for (const pt of sorted) {
    if (pt.decodeTokPerSec > bestSpeed) {
      frontier.add(pt.id);
      bestSpeed = pt.decodeTokPerSec;
    }
  }
  return frontier;
}
""",
    """# Python PyArrow & Telemetry Pipeline Example
from dataclasses import asdict, dataclass
import json
from pathlib import Path

@dataclass(slots=True)
class PowerSample:
    timestamp_ns: int
    milliwatts: float

def integrate_joules(samples: list[PowerSample]) -> float:
    if len(samples) < 2:
        return 0.0
    total_joules = 0.0
    for prev, curr in zip(samples, samples[1:], strict=False):
        dt_s = max(0.0, (curr.timestamp_ns - prev.timestamp_ns) * 1e-9)
        avg_w = ((prev.milliwatts + curr.milliwatts) * 0.5) * 1e-3
        total_joules += avg_w * dt_s
    return total_joules
""",
)


def write_corpora(out_dir: Path) -> tuple[Path, Path]:
    """Write deterministic calibration and held-out evaluation corpora."""
    out_dir.mkdir(parents=True, exist_ok=True)
    calib_path = out_dir / "calibration_corpus.txt"
    eval_path = out_dir / "eval_corpus.txt"

    # Repeat varied permutations with distinct framing so llama-imatrix gets >= 16 chunks of 512 tokens.
    calib_blocks: list[str] = []
    for round_idx in range(8):
        for sec_idx in range(len(SECTIONS)):
            shifted = SECTIONS[(sec_idx + round_idx) % len(SECTIONS)]
            calib_blocks.append(f"--- Session {round_idx + 1}.{sec_idx + 1} ---\n{shifted.strip()}\n")
    calib_path.write_text("\n\n".join(calib_blocks) + "\n", encoding="utf-8")

    # Held-out ordering & structured prompt evaluation text (~4-6 chunks of 512 tokens for fast KLD).
    eval_blocks = [
        SECTIONS[1].strip(),
        SECTIONS[2].strip(),
        SECTIONS[3].strip(),
        SECTIONS[4].strip(),
        SECTIONS[0].strip(),
        SECTIONS[5].strip(),
    ] * 3
    eval_path.write_text("\n\n".join(eval_blocks) + "\n", encoding="utf-8")
    return calib_path, eval_path
