from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "kaggle_fable_gate_0704"
SOURCE_NOTEBOOK = (
    SOURCE_DIR / "name-that-llm-0704-fable-gate.ipynb"
)
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
INFER_SCRIPT = ROOT / "scripts" / "predict_pnasf_kernel.py"

OUTPUT_DIR = ROOT / "kaggle_pnasf_0704"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "name-that-llm-0704-pnasf.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"

KERNEL_ID = "yyyyhhhhwww/ntl-0704-pnasf"
KERNEL_TITLE = "NTL 0704 PNASF"
PNASF_WEIGHT = 0.10
DATASET_ID = "yyyyhhhhwww/ntl-pnasf-run3"


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    notebook = json.loads(
        SOURCE_NOTEBOOK.read_text(encoding="utf-8")
    )
    metadata = json.loads(
        SOURCE_METADATA.read_text(encoding="utf-8")
    )
    runtime = INFER_SCRIPT.read_text(encoding="utf-8")
    run_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "NTL_PIPE_WORKERS='1'" in "".join(cell.get("source", []))
    )
    runtime_cell = code_cell(
        "%%writefile pnasf_kernel_infer.py\n" + runtime
    )
    run_cell = code_cell(
        f'''# PNASF_0704: prompt-invariant source field reader.
PNASF_WEIGHT = {PNASF_WEIGHT}
PNASF_REPORT = {{"status": "skipped", "weight": PNASF_WEIGHT}}
if not FAST_COMMIT:
    try:
        pnasf_root = next(
            path
            for path in Path("/kaggle/input").rglob(
                "pnasf_encoder.pt"
            )
        ).parent
        pnasf_model = asset_path(
            "/kaggle/input/models/qwen-lm/qwen-3/"
            "transformers/4b-base/1"
        )
        pnasf_out = "/kaggle/working/pnasf_test.npz"
        subprocess.run(
            [
                sys.executable,
                "pnasf_kernel_infer.py",
                "--model",
                pnasf_model,
                "--input-csv",
                V1_TEST_CSV,
                "--encoder",
                str(pnasf_root / "pnasf_encoder.pt"),
                "--classifier",
                str(pnasf_root / "pnasf_classifier.joblib"),
                "--output",
                pnasf_out,
                "--batch-size",
                "2",
                "--max-length",
                "384",
            ],
            check=True,
        )
        with np.load(pnasf_out, allow_pickle=False) as payload:
            pnasf_ids = payload["ids"].astype(str).tolist()
            pnasf_logp = payload["logits"].astype(np.float64)
        with np.load(FINAL_LOGITS, allow_pickle=False) as saved:
            ids = saved["ids"].astype(str).tolist()
            labels = saved["labels"].astype(str).tolist()
            base_logits = saved["logits"].astype(np.float64)
        assert pnasf_ids == ids
        assert pnasf_logp.shape == base_logits.shape
        assert np.isfinite(pnasf_logp).all()
        np.savez(
            FINAL_LOGITS,
            ids=np.asarray(ids),
            labels=np.asarray(labels),
            logits=base_logits + PNASF_WEIGHT * pnasf_logp,
        )
        PNASF_REPORT = {{
            "status": "applied",
            "weight": PNASF_WEIGHT,
            "rows": len(ids),
        }}
        print("PNASF_READY", PNASF_REPORT, flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        PNASF_REPORT = {{
            "status": "failed",
            "weight": PNASF_WEIGHT,
            "error": repr(error),
        }}
        print(
            "PNASF failed; preserving the original 0.704 logits",
            PNASF_REPORT,
            flush=True,
        )
else:
    print("PNASF skipped on FAST_COMMIT", flush=True)
print("PNASF_REPORT", PNASF_REPORT, flush=True)
'''
    )
    notebook["cells"].insert(run_index + 1, runtime_cell)
    notebook["cells"].insert(run_index + 2, run_cell)
    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": (
                "# 0.704 plus PNASF source field\n\n"
                "This private ablation preserves the learned Fable/GLM gate "
                "and adds the PNASF prompt-invariant Qwen3-4B reader at log "
                "weight 0.10. The reader was trained on 20k external "
                "same-prompt passages with supervised contrastive, "
                "prompt-centroid contrastive, and prompt-adversarial "
                "objectives. A failed PNASF stage leaves the original "
                "0.704 logits unchanged."
            ),
        },
    )
    metadata["id"] = KERNEL_ID
    metadata["title"] = KERNEL_TITLE
    metadata["code_file"] = OUTPUT_NOTEBOOK.name
    metadata["is_private"] = True
    metadata.pop("id_no", None)
    sources = list(metadata.get("dataset_sources", []))
    if DATASET_ID not in sources:
        sources.append(DATASET_ID)
    metadata["dataset_sources"] = sources

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_NOTEBOOK}")
    print(f"wrote {OUTPUT_METADATA}")


if __name__ == "__main__":
    main()
