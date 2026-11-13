"""Command-line entry point for hf-bench."""

import argparse
from pathlib import Path

from hf_bench.export import export_all
from hf_bench.models import MODELS
from hf_bench.pipeline import bench_orin_all, default_cache_dir, run_local_pipeline
from hf_bench.publish import publish_to_hub


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hf-bench", description="Edge GGUF quantization & benchmark suite")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_local = sub.add_parser("run-local", help="Quantize, compute KL-divergence, and benchmark locally")
    p_local.add_argument("--model", choices=sorted(MODELS.keys()), required=True)
    p_local.add_argument("--threads", type=int, default=6)
    p_local.add_argument("--keep-f16", action="store_true")

    p_orin = sub.add_parser("bench-orin", help="Run CUDA sm_87 + INA3221 power benchmark on Orin Nano")
    p_orin.add_argument("--model", choices=sorted(MODELS.keys()), required=True)
    p_orin.add_argument("--host", default="orin-nano")

    p_exp = sub.add_parser("export", help="Export Parquet dataset, Space JSON, and HF cards")
    p_exp.add_argument("--repo-root", type=Path, default=Path("."))

    p_pub = sub.add_parser("publish", help="Publish Model(s), Dataset, and Static Space to Hugging Face")
    p_pub.add_argument("--repo-root", type=Path, default=Path("."))
    p_pub.add_argument("--space-build", type=Path, default=None)
    p_pub.add_argument("--skip-models", action="store_true")
    p_pub.add_argument("--skip-dataset", action="store_true")
    p_pub.add_argument("--skip-space", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run-local":
        run_local_pipeline(
            model_slug=args.model,
            cache_dir=args.cache_dir,
            threads=args.threads,
            drop_f16=not args.keep_f16,
        )
    elif args.cmd == "bench-orin":
        bench_orin_all(spec=MODELS[args.model], cache_dir=args.cache_dir, host=args.host)
    elif args.cmd == "export":
        out = export_all(cache_dir=args.cache_dir, repo_root=args.repo_root)
        print(f"Exported cards and Parquet dataset under {out}")
    elif args.cmd == "publish":
        publish_to_hub(
            cache_dir=args.cache_dir,
            repo_root=args.repo_root,
            space_build=args.space_build,
            models=not args.skip_models,
            dataset=not args.skip_dataset,
            space=not args.skip_space and args.space_build is not None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
