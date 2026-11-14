"""End-to-end quantization, KL-divergence, local x86_64, and SSH Orin Nano benchmark pipeline."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from huggingface_hub import snapshot_download

from hf_bench.corpus import write_corpora
from hf_bench.models import MODELS, QUANT_MATRIX, ModelSpec, QuantSpec
from hf_bench.parsers import parse_llama_bench_json, parse_perplexity_output


@dataclass
class BenchmarkRecord:
    record_id: str
    timestamp_utc: str
    model_slug: str
    model_name: str
    hf_source: str
    hf_target_repo: str
    params_b: float
    quant: str
    imatrix_used: bool
    bpw: float
    weight_bytes: int
    weight_gib: float
    hardware_id: str
    hardware_label: str
    backend: str
    memory_budget_gib: float
    status: str
    oom_reason: str | None
    pp512_tok_s: float | None
    pp512_std: float | None
    tg128_tok_s: float | None
    tg128_std: float | None
    ppl: float | None
    ppl_base_fp16: float | None
    mean_kld: float | None
    p99_kld: float | None
    median_kld: float | None
    max_kld: float | None
    rms_delta_p_pct: float | None
    same_top1_pct: float | None
    avg_watts: float | None
    joules_per_token: float | None
    peak_mem_mib: float | None


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "hf-bench"


def _idle_prefix() -> list[str]:
    chrt = shutil.which("chrt")
    ionice = shutil.which("ionice")
    prefix: list[str] = []
    if chrt:
        prefix.extend([chrt, "--idle", "0"])
    if ionice:
        prefix.extend([ionice, "-c3"])
    return prefix


def _run_logged(cmd: list[str], *, idle: bool = True) -> str:
    full_cmd = (_idle_prefix() + cmd) if idle else cmd
    print(f"+ {' '.join(full_cmd)}", flush=True)
    proc = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
    return f"{proc.stdout}\n{proc.stderr}"


def _read_rapl_uj() -> int | None:
    rapl = Path("/sys/class/powercap/intel-rapl:0/energy_uj")
    try:
        return int(rapl.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_mem_available_mib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kib = int(line.split()[1])
                return kib / 1024.0
    except (OSError, ValueError, IndexError):
        return None
    return None


def model_work_dir(cache_dir: Path, spec: ModelSpec) -> Path:
    d = cache_dir / "models" / spec.slug
    for sub in ("gguf", "quality", "bench"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def prepare_f16(spec: ModelSpec, cache_dir: Path) -> Path:
    work = model_work_dir(cache_dir, spec)
    f16_path = work / "gguf" / f"{spec.display_name}-F16.gguf"
    if f16_path.exists() and f16_path.stat().st_size > 1024 * 1024:
        print(f"Reusing existing F16 GGUF: {f16_path}")
        return f16_path

    hf_cache = cache_dir / "hf-hub"
    hf_cache.mkdir(parents=True, exist_ok=True)
    print(f"Downloading safetensors for {spec.hf_source} into {hf_cache}...")
    snap_dir = snapshot_download(
        repo_id=spec.hf_source,
        cache_dir=str(hf_cache),
        ignore_patterns=["*.pth", "*.pt", "*.bin", "*.msgpack", "*.h5"],
    )

    convert_bin = os.environ.get("CONVERT_HF_TO_GGUF") or shutil.which("convert-hf-to-gguf")
    if not convert_bin:
        raise RuntimeError("convert-hf-to-gguf not found on PATH or CONVERT_HF_TO_GGUF")

    _run_logged([convert_bin, "--outtype", "f16", "--outfile", str(f16_path), snap_dir], idle=True)
    return f16_path


def build_imatrix(spec: ModelSpec, cache_dir: Path, f16_path: Path, threads: int = 6) -> Path:
    work = model_work_dir(cache_dir, spec)
    imatrix_path = work / "gguf" / "imatrix.gguf"
    if imatrix_path.exists() and imatrix_path.stat().st_size > 4096:
        print(f"Reusing existing imatrix: {imatrix_path}")
        return imatrix_path

    calib_path, _ = write_corpora(cache_dir / "corpora")
    _run_logged(
        [
            "llama-imatrix",
            "-m",
            str(f16_path),
            "-f",
            str(calib_path),
            "-c",
            "512",
            "--chunks",
            "12",
            "--parse-special",
            "--process-output",
            "-t",
            str(threads),
            "-o",
            str(imatrix_path),
        ],
        idle=True,
    )
    return imatrix_path


def quant_gguf_path(work: Path, spec: ModelSpec, qspec: QuantSpec) -> Path:
    return work / "gguf" / f"{spec.display_name}-{qspec.name}.gguf"


def quantize_all(
    spec: ModelSpec,
    cache_dir: Path,
    f16_path: Path,
    imatrix_path: Path,
    threads: int = 6,
) -> dict[str, Path]:
    work = model_work_dir(cache_dir, spec)
    outputs: dict[str, Path] = {}
    for qspec in QUANT_MATRIX:
        out_path = quant_gguf_path(work, spec, qspec)
        if out_path.exists() and out_path.stat().st_size > 1024 * 1024:
            print(f"Reusing {qspec.name}: {out_path}")
            outputs[qspec.name] = out_path
            continue
        cmd = ["llama-quantize"]
        if qspec.use_imatrix:
            cmd.extend(["--imatrix", str(imatrix_path)])
        cmd.extend([str(f16_path), str(out_path), qspec.gguf_type, str(threads)])
        _run_logged(cmd, idle=True)
        outputs[qspec.name] = out_path
    return outputs


def eval_quality_all(
    spec: ModelSpec,
    cache_dir: Path,
    f16_path: Path,
    quant_paths: dict[str, Path],
    threads: int = 6,
    chunks: int = 6,
) -> None:
    work = model_work_dir(cache_dir, spec)
    _, eval_path = write_corpora(cache_dir / "corpora")
    base_logits = work / "quality" / "base-f16.logits"
    f16_json = work / "quality" / "F16.json"

    need_logits = any(not (work / "quality" / f"{qname}.json").exists() for qname in quant_paths)
    if not f16_json.exists() or (need_logits and not base_logits.exists()):
        out = _run_logged(
            [
                "llama-perplexity",
                "-m",
                str(f16_path),
                "-f",
                str(eval_path),
                "-c",
                "512",
                "--chunks",
                str(chunks),
                "-t",
                str(threads),
                "--save-all-logits",
                str(base_logits),
            ],
            idle=True,
        )
        metrics = parse_perplexity_output(out)
        f16_json.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")

    for qname, qpath in quant_paths.items():
        qjson = work / "quality" / f"{qname}.json"
        if qjson.exists():
            print(f"Reusing quality metrics for {qname}")
            continue
        out = _run_logged(
            [
                "llama-perplexity",
                "-m",
                str(qpath),
                "-f",
                str(eval_path),
                "-c",
                "512",
                "--chunks",
                str(chunks),
                "-t",
                str(threads),
                "--kl-divergence-base",
                str(base_logits),
                "--kl-divergence",
            ],
            idle=True,
        )
        metrics = parse_perplexity_output(out)
        qjson.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")

    # Remove multi-hundred-MiB temporary logits file once every quant is scored.
    if base_logits.exists():
        base_logits.unlink()


def _load_quality(work: Path, qname: str) -> tuple[dict[str, float | None], float | None]:
    f16_file = work / "quality" / "F16.json"
    base_ppl: float | None = None
    if f16_file.exists():
        base_ppl = json.loads(f16_file.read_text(encoding="utf-8")).get("ppl")
    qfile = work / "quality" / f"{qname}.json"
    if not qfile.exists():
        return {}, base_ppl
    return json.loads(qfile.read_text(encoding="utf-8")), base_ppl


def bench_local_all(
    spec: ModelSpec,
    cache_dir: Path,
    quant_paths: dict[str, Path],
    threads: int = 6,
) -> list[BenchmarkRecord]:
    work = model_work_dir(cache_dir, spec)
    records: list[BenchmarkRecord] = []
    qspec_by_name = {q.name: q for q in QUANT_MATRIX}

    for qname, qpath in quant_paths.items():
        out_json = work / "bench" / f"nixos-nvme-{qname}.json"
        if out_json.exists():
            print(f"Reusing local benchmark for {qname}")
            records.append(BenchmarkRecord(**json.loads(out_json.read_text(encoding="utf-8"))))
            continue

        qspec = qspec_by_name[qname]
        weight_bytes = qpath.stat().st_size
        weight_gib = round(weight_bytes / (1024.0**3), 3)
        bpw = round((weight_bytes * 8.0) / (spec.params_b * 1e9), 2)

        mem_before = _read_mem_available_mib()
        e_before = _read_rapl_uj()
        t0 = time.monotonic()
        proc = subprocess.run(
            [
                *_idle_prefix(),
                "llama-bench",
                "-m",
                str(qpath),
                "-p",
                "512",
                "-n",
                "128",
                "-r",
                "3",
                "-t",
                str(threads),
                "-o",
                "json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        dt = max(0.001, time.monotonic() - t0)
        e_after = _read_rapl_uj()
        mem_after = _read_mem_available_mib()

        tp = parse_llama_bench_json(proc.stdout)
        avg_watts: float | None = None
        j_per_tok: float | None = None
        if e_before is not None and e_after is not None and e_after >= e_before:
            joules = (e_after - e_before) * 1e-6
            avg_watts = round(joules / dt, 2)
            if tp.tg128_tok_s > 0:
                j_per_tok = round(avg_watts / tp.tg128_tok_s, 4)

        weight_mib = weight_bytes / (1024.0 * 1024.0)
        delta_mib = (mem_before - mem_after) if (mem_before and mem_after and mem_before > mem_after) else 0.0
        peak_mem_mib = round(max(weight_mib + 180.0, delta_mib), 1)

        qual, base_ppl = _load_quality(work, qname)
        rec = BenchmarkRecord(
            record_id=f"{spec.slug}__nixos-nvme__{qname}",
            timestamp_utc=datetime.now(UTC).isoformat(),
            model_slug=spec.slug,
            model_name=spec.display_name,
            hf_source=spec.hf_source,
            hf_target_repo=spec.hf_target_repo,
            params_b=spec.params_b,
            quant=qname,
            imatrix_used=qspec.use_imatrix,
            bpw=bpw,
            weight_bytes=weight_bytes,
            weight_gib=weight_gib,
            hardware_id="nixos-nvme",
            hardware_label="Intel Core i3-1315U (Raptor Lake AVX2, 62 GiB)",
            backend="CPU (AVX2 / AlderLake ggml)",
            memory_budget_gib=62.0,
            status="ok",
            oom_reason=None,
            pp512_tok_s=round(tp.pp512_tok_s, 2),
            pp512_std=round(tp.pp512_std, 2),
            tg128_tok_s=round(tp.tg128_tok_s, 2),
            tg128_std=round(tp.tg128_std, 2),
            ppl=qual.get("ppl"),
            ppl_base_fp16=base_ppl,
            mean_kld=qual.get("mean_kld"),
            p99_kld=qual.get("p99_kld"),
            median_kld=qual.get("median_kld"),
            max_kld=qual.get("max_kld"),
            rms_delta_p_pct=qual.get("rms_delta_p_pct"),
            same_top1_pct=qual.get("same_top1_pct"),
            avg_watts=avg_watts,
            joules_per_token=j_per_tok,
            peak_mem_mib=peak_mem_mib,
        )
        out_json.write_text(json.dumps(asdict(rec), indent=2) + "\n", encoding="utf-8")
        records.append(rec)
    return records


ORIN_NIX_EXPR = """
let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
      cudaCapabilities = [ "8.7" ];
    };
  };
in
pkgs.llama-cpp.override {
  cudaPackages = pkgs.cudaPackages_12_6;
}
""".strip()

ORIN_RUNNER_PY = """
import glob, json, os, subprocess, sys, threading, time

llama_bench = sys.argv[1]
model_path = sys.argv[2]

def read_orin_mw():
    # Tegra Orin Nano INA3221 hwmon exposes VDD_IN power or voltage*current.
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            with open(os.path.join(hw, "name"), encoding="utf-8") as f:
                if f.read().strip() != "ina3221":
                    continue
        except OSError:
            continue
        p_path = os.path.join(hw, "power1_input")
        if os.path.exists(p_path):
            try:
                with open(p_path, encoding="utf-8") as f:
                    return float(f.read().strip()) / 1000.0
            except (OSError, ValueError):
                pass
        try:
            with open(os.path.join(hw, "in1_input"), encoding="utf-8") as fv:
                mv = float(fv.read().strip())
            with open(os.path.join(hw, "curr1_input"), encoding="utf-8") as fc:
                ma = float(fc.read().strip())
            return (mv * ma) / 1000.0
        except (OSError, ValueError):
            continue
    return None

def mem_avail_mib():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return 0.0

samples = []
min_avail = mem_avail_mib()
baseline_avail = min_avail
stop = False

def sampler():
    global min_avail
    while not stop:
        mw = read_orin_mw()
        if mw is not None:
            samples.append(mw)
        ma = mem_avail_mib()
        if ma < min_avail:
            min_avail = ma
        time.sleep(0.2)

t = threading.Thread(target=sampler, daemon=True)
t.start()
proc = subprocess.run(
    [llama_bench, "-m", model_path, "-ngl", "99", "-p", "512", "-n", "128", "-r", "3", "-o", "json"],
    capture_output=True,
    text=True,
)
stop = True
t.join(timeout=1.0)

avg_w = (sum(samples) / len(samples) / 1000.0) if samples else None
used_delta_mib = max(0.0, baseline_avail - min_avail)
print("HF_BENCH_RESULT_JSON:" + json.dumps({
    "returncode": proc.returncode,
    "stdout": proc.stdout,
    "stderr": proc.stderr[-2000:],
    "avg_watts": avg_w,
    "used_delta_mib": used_delta_mib,
    "baseline_avail_mib": baseline_avail,
}))
""".strip()


def _ssh_opts_and_env() -> tuple[list[str], dict[str, str]]:
    env = {**os.environ, "SSH_ASKPASS": "/bin/false", "SSH_ASKPASS_REQUIRE": "force"}
    opts = ["-o", "BatchMode=yes"]
    notouch_key = Path.home() / ".ssh" / "id_ed25519_sk_rk_GitHubNoTouchV2"
    if notouch_key.exists():
        opts.extend(["-i", str(notouch_key)])
        existing = env.get("NIX_SSHOPTS", "").strip()
        key_flag = f"-i {shlex.quote(str(notouch_key))}"
        env["NIX_SSHOPTS"] = f"{existing} {key_flag}".strip() if existing else key_flag
    return opts, env


def _ssh_cmd(host: str, remote_args: list[str]) -> subprocess.CompletedProcess[str]:
    opts, env = _ssh_opts_and_env()
    return subprocess.run(
        ["ssh", *opts, host, *remote_args],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def bench_orin_all(
    spec: ModelSpec,
    cache_dir: Path,
    host: str = "orin-nano",
) -> list[BenchmarkRecord]:
    work = model_work_dir(cache_dir, spec)
    records: list[BenchmarkRecord] = []

    pending: list[tuple[QuantSpec, Path]] = []
    for qspec in QUANT_MATRIX:
        out_json = work / "bench" / f"orin-nano-{qspec.name}.json"
        qpath = quant_gguf_path(work, spec, qspec)
        if out_json.exists():
            print(f"Reusing Orin benchmark for {qspec.name}")
            records.append(BenchmarkRecord(**json.loads(out_json.read_text(encoding="utf-8"))))
        elif qpath.exists():
            pending.append((qspec, qpath))

    if not pending:
        return records

    # Open the Orin build/bench window inside try/finally so done is always touched on setup failure.
    prep_script = (
        "set -euo pipefail; "
        "touch /run/mesh-llm-build-window/start || true; "
        "for _ in $(seq 45); do systemctl is-active --quiet mesh-llm.service || break; sleep 2; done; "
        "py=$(command -v python3 || ls -1 /nix/store/*-python3-3.*/bin/python3 2>/dev/null | head -n1); "
        "if [ -z \"$py\" ]; then echo 'python3 not found on Orin' >&2; exit 1; fi; "
        "mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo); "
        "mem_avail_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); "
        'echo "REMOTE_INFO:$py:$mem_total_kib:$mem_avail_kib"'
    )
    try:
        prep = _ssh_cmd(host, ["bash", "-c", shlex.quote(prep_script)])
        if prep.returncode != 0:
            raise RuntimeError(f"SSH setup on {host} failed: {prep.stderr.strip()}")

        python_bin = "python3"
        mem_total_gib = 7.6
        mem_avail_gib = 6.6
        for line in prep.stdout.splitlines():
            if line.startswith("REMOTE_INFO:"):
                parts = line.split(":")
                python_bin = parts[1]
                mem_total_gib = round(int(parts[2]) / (1024.0 * 1024.0), 2)
                mem_avail_gib = round(int(parts[3]) / (1024.0 * 1024.0), 2)

        # Reuse an existing CUDA sm_87 llama-bench in /nix/store before falling back to nix-build.
        locate_or_build = (
            "set -euo pipefail; "
            "for b in $(command -v llama-bench 2>/dev/null || true) /nix/store/*-llama-cpp-*/bin/llama-bench; do "
            '  if [ -x "$b" ] && "$b" --list-devices 2>&1 | grep -q CUDA0; then echo "$b"; exit 0; fi; '
            "done; "
            f"out=$(systemd-run --user --wait --pipe --collect nix-build --no-out-link -E '{ORIN_NIX_EXPR}'); "
            'echo "$out/bin/llama-bench"'
        )
        build_res = _ssh_cmd(host, ["bash", "-c", shlex.quote(locate_or_build)])
        if build_res.returncode != 0:
            raise RuntimeError(f"Building/locating CUDA llama-bench on {host} failed: {build_res.stderr}")
        llama_bench_bin = build_res.stdout.strip().splitlines()[-1]
        print(f"Using Orin CUDA llama-bench: {llama_bench_bin}")

        for qspec, qpath in pending:
            out_json = work / "bench" / f"orin-nano-{qspec.name}.json"
            weight_bytes = qpath.stat().st_size
            weight_gib = round(weight_bytes / (1024.0**3), 3)
            bpw = round((weight_bytes * 8.0) / (spec.params_b * 1e9), 2)
            qual, base_ppl = _load_quality(work, qspec.name)

            # Tegra CUDA context + KV/compute buffers require ~0.45 GiB above weight tensors.
            if weight_gib + 0.45 > mem_avail_gib:
                rec = BenchmarkRecord(
                    record_id=f"{spec.slug}__orin-nano__{qspec.name}",
                    timestamp_utc=datetime.now(UTC).isoformat(),
                    model_slug=spec.slug,
                    model_name=spec.display_name,
                    hf_source=spec.hf_source,
                    hf_target_repo=spec.hf_target_repo,
                    params_b=spec.params_b,
                    quant=qspec.name,
                    imatrix_used=qspec.use_imatrix,
                    bpw=bpw,
                    weight_bytes=weight_bytes,
                    weight_gib=weight_gib,
                    hardware_id="orin-nano",
                    hardware_label="NVIDIA Jetson Orin Nano (Ampere sm_87 CUDA 12.6, 8 GiB Unified)",
                    backend="CUDA 12.6 (Tegra sm_87)",
                    memory_budget_gib=mem_total_gib,
                    status="oom",
                    oom_reason=(
                        f"Weights ({weight_gib:.2f} GiB) + CUDA/KV context exceed "
                        f"{mem_avail_gib:.2f} GiB available unified RAM"
                    ),
                    pp512_tok_s=None,
                    pp512_std=None,
                    tg128_tok_s=None,
                    tg128_std=None,
                    ppl=qual.get("ppl"),
                    ppl_base_fp16=base_ppl,
                    mean_kld=qual.get("mean_kld"),
                    p99_kld=qual.get("p99_kld"),
                    median_kld=qual.get("median_kld"),
                    max_kld=qual.get("max_kld"),
                    rms_delta_p_pct=qual.get("rms_delta_p_pct"),
                    same_top1_pct=qual.get("same_top1_pct"),
                    avg_watts=None,
                    joules_per_token=None,
                    peak_mem_mib=None,
                )
                out_json.write_text(json.dumps(asdict(rec), indent=2) + "\n", encoding="utf-8")
                records.append(rec)
                continue

            # Stream GGUF directly into Orin's NVMe-backed /nix/store (Orin / and /tmp are 4 GiB tmpfs).
            print(f"Staging {qpath.name} ({weight_gib} GiB) to ssh-ng://{host} /nix/store...")
            ssh_opts, ssh_env = _ssh_opts_and_env()
            add_proc = subprocess.run(
                ["nix", "--store", f"ssh-ng://{host}", "store", "add-file", str(qpath)],
                env=ssh_env,
                check=True,
                capture_output=True,
                text=True,
            )
            store_gguf = add_proc.stdout.strip()
            try:
                run_res = subprocess.run(
                    [
                        "ssh",
                        *ssh_opts,
                        host,
                        shlex.quote(python_bin),
                        "-",
                        shlex.quote(llama_bench_bin),
                        shlex.quote(store_gguf),
                    ],
                    env=ssh_env,
                    input=ORIN_RUNNER_PY,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                payload = None
                for line in run_res.stdout.splitlines():
                    if line.startswith("HF_BENCH_RESULT_JSON:"):
                        payload = json.loads(line.removeprefix("HF_BENCH_RESULT_JSON:"))
                if not payload:
                    raise RuntimeError(f"Missing JSON result on Orin for {qspec.name}: {run_res.stderr}")

                if int(payload["returncode"]) != 0:
                    rec = BenchmarkRecord(
                        record_id=f"{spec.slug}__orin-nano__{qspec.name}",
                        timestamp_utc=datetime.now(UTC).isoformat(),
                        model_slug=spec.slug,
                        model_name=spec.display_name,
                        hf_source=spec.hf_source,
                        hf_target_repo=spec.hf_target_repo,
                        params_b=spec.params_b,
                        quant=qspec.name,
                        imatrix_used=qspec.use_imatrix,
                        bpw=bpw,
                        weight_bytes=weight_bytes,
                        weight_gib=weight_gib,
                        hardware_id="orin-nano",
                        hardware_label="NVIDIA Jetson Orin Nano (Ampere sm_87 CUDA 12.6, 8 GiB Unified)",
                        backend="CUDA 12.6 (Tegra sm_87)",
                        memory_budget_gib=mem_total_gib,
                        status="oom",
                        oom_reason=f"CUDA allocation failed on 8 GiB unified RAM ({payload['stderr'][-160:].strip()})",
                        pp512_tok_s=None,
                        pp512_std=None,
                        tg128_tok_s=None,
                        tg128_std=None,
                        ppl=qual.get("ppl"),
                        ppl_base_fp16=base_ppl,
                        mean_kld=qual.get("mean_kld"),
                        p99_kld=qual.get("p99_kld"),
                        median_kld=qual.get("median_kld"),
                        max_kld=qual.get("max_kld"),
                        rms_delta_p_pct=qual.get("rms_delta_p_pct"),
                        same_top1_pct=qual.get("same_top1_pct"),
                        avg_watts=None,
                        joules_per_token=None,
                        peak_mem_mib=None,
                    )
                else:
                    tp = parse_llama_bench_json(payload["stdout"])
                    avg_w = round(float(payload["avg_watts"]), 2) if payload["avg_watts"] is not None else None
                    j_tok = round(avg_w / tp.tg128_tok_s, 4) if (avg_w and tp.tg128_tok_s > 0) else None
                    weight_mib = weight_bytes / (1024.0 * 1024.0)
                    peak_mib = round(max(weight_mib + 420.0, float(payload["used_delta_mib"])), 1)
                    rec = BenchmarkRecord(
                        record_id=f"{spec.slug}__orin-nano__{qspec.name}",
                        timestamp_utc=datetime.now(UTC).isoformat(),
                        model_slug=spec.slug,
                        model_name=spec.display_name,
                        hf_source=spec.hf_source,
                        hf_target_repo=spec.hf_target_repo,
                        params_b=spec.params_b,
                        quant=qspec.name,
                        imatrix_used=qspec.use_imatrix,
                        bpw=bpw,
                        weight_bytes=weight_bytes,
                        weight_gib=weight_gib,
                        hardware_id="orin-nano",
                        hardware_label="NVIDIA Jetson Orin Nano (Ampere sm_87 CUDA 12.6, 8 GiB Unified)",
                        backend="CUDA 12.6 (Tegra sm_87)",
                        memory_budget_gib=mem_total_gib,
                        status="ok",
                        oom_reason=None,
                        pp512_tok_s=round(tp.pp512_tok_s, 2),
                        pp512_std=round(tp.pp512_std, 2),
                        tg128_tok_s=round(tp.tg128_tok_s, 2),
                        tg128_std=round(tp.tg128_std, 2),
                        ppl=qual.get("ppl"),
                        ppl_base_fp16=base_ppl,
                        mean_kld=qual.get("mean_kld"),
                        p99_kld=qual.get("p99_kld"),
                        median_kld=qual.get("median_kld"),
                        max_kld=qual.get("max_kld"),
                        rms_delta_p_pct=qual.get("rms_delta_p_pct"),
                        same_top1_pct=qual.get("same_top1_pct"),
                        avg_watts=avg_w,
                        joules_per_token=j_tok,
                        peak_mem_mib=peak_mib,
                    )
                out_json.write_text(json.dumps(asdict(rec), indent=2) + "\n", encoding="utf-8")
                records.append(rec)
            finally:
                del_res = _ssh_cmd(host, [f"exec nix-store --delete {shlex.quote(store_gguf)}"])
                if del_res.returncode != 0:
                    err_msg = del_res.stderr.strip() or f"exit {del_res.returncode}"
                    print(
                        f"Warning: failed to delete staged GGUF {store_gguf} on {host}: {err_msg}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        print(f"Releasing Orin build window on {host}...")
        _ssh_cmd(host, ["bash", "-c", shlex.quote("touch /run/mesh-llm-build-window/done || true")])

    return records


def run_local_pipeline(model_slug: str, cache_dir: Path, threads: int = 6, drop_f16: bool = True) -> None:
    spec = MODELS[model_slug]
    f16_path = prepare_f16(spec, cache_dir)
    imatrix_path = build_imatrix(spec, cache_dir, f16_path, threads=threads)
    quant_paths = quantize_all(spec, cache_dir, f16_path, imatrix_path, threads=threads)
    eval_quality_all(spec, cache_dir, f16_path, quant_paths, threads=threads)
    bench_local_all(spec, cache_dir, quant_paths, threads=threads)
    if drop_f16 and f16_path.exists():
        print(f"Removing intermediate F16 GGUF ({f16_path}) to conserve ~/.cache space")
        f16_path.unlink()
