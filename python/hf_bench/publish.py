"""Publish quantized GGUF suites, Parquet dataset, and SvelteKit Static Space to Hugging Face."""

from pathlib import Path

from huggingface_hub import HfApi

from hf_bench.export import export_all
from hf_bench.models import MODELS, QUANT_MATRIX
from hf_bench.pipeline import model_work_dir, quant_gguf_path

DATASET_REPO = "kleinbem/edge-llm-benchmarks"
SPACE_REPO = "kleinbem/edge-llm-pareto"


def publish_to_hub(
    cache_dir: Path,
    repo_root: Path,
    space_build: Path | None = None,
    *,
    models: bool = True,
    dataset: bool = True,
    space: bool = True,
) -> None:
    export_dir = export_all(cache_dir, repo_root)
    api = HfApi()

    if dataset:
        ds_dir = export_dir / "dataset"
        print(f"Publishing Dataset -> https://huggingface.co/datasets/{DATASET_REPO}")
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            folder_path=str(ds_dir),
            commit_message="chore(data): update edge GGUF quality & power benchmark dataset",
        )

    if models:
        for slug, spec in MODELS.items():
            card_file = export_dir / "model_cards" / f"{slug}-README.md"
            if not card_file.exists():
                continue
            work = model_work_dir(cache_dir, spec)
            print(f"Publishing Model -> https://huggingface.co/{spec.hf_target_repo}")
            api.create_repo(repo_id=spec.hf_target_repo, repo_type="model", private=False, exist_ok=True)
            api.upload_file(
                path_or_fileobj=str(card_file),
                path_in_repo="README.md",
                repo_id=spec.hf_target_repo,
                repo_type="model",
                commit_message=f"docs(card): update {spec.display_name} KL-divergence & Orin telemetry table",
            )
            imatrix_path = work / "gguf" / "imatrix.gguf"
            if imatrix_path.exists():
                api.upload_file(
                    path_or_fileobj=str(imatrix_path),
                    path_in_repo="imatrix.gguf",
                    repo_id=spec.hf_target_repo,
                    repo_type="model",
                    commit_message="feat(imatrix): upload agentic + code importance matrix",
                )
            for qspec in QUANT_MATRIX:
                if not qspec.publish_gguf:
                    continue
                qpath = quant_gguf_path(work, spec, qspec)
                if qpath.exists():
                    print(f"  Uploading {qpath.name}...")
                    api.upload_file(
                        path_or_fileobj=str(qpath),
                        path_in_repo=qpath.name,
                        repo_id=spec.hf_target_repo,
                        repo_type="model",
                        commit_message=f"feat(gguf): add {qspec.name} calibrated quantization",
                    )

    if space and space_build is not None:
        print(f"Publishing Static Space -> https://huggingface.co/spaces/{SPACE_REPO}")
        api.create_repo(
            repo_id=SPACE_REPO,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=SPACE_REPO,
            repo_type="space",
            folder_path=str(space_build),
            commit_message="feat(space): deploy SvelteKit Pareto & 8 GiB Edge KV-cache explorer",
        )
