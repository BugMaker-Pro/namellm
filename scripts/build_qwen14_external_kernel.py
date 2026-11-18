from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts" / "kaggle_0701_pairwise"
SOURCE_NOTEBOOK = SOURCE_DIR / "name-that-llm-0701-glm-pairwise.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
RUNTIME_SCRIPT = ROOT / "scripts" / "predict_external_qwen14.py"

OUTPUT_DIR = ROOT / "kaggle_qwen14_external_0702"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "name-that-llm-qwen14-external-0702.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"

KERNEL_ID = "yyyyhhhhwww/name-that-llm-qwen14-external-0702"
KERNEL_TITLE = "Name That LLM Qwen14 External 0702"
DATASET_SLUG = "yyyyhhhhwww/ntl-qwen14-external-e6-lr8e5"
QWEN14_WEIGHT = 0.20


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(
            f"Expected exactly one match for patch target, got "
            f"{source.count(old)}"
        )
    return source.replace(old, new)


def main() -> None:
    notebook = json.loads(
        SOURCE_NOTEBOOK.read_text(encoding="utf-8")
    )
    metadata = json.loads(
        SOURCE_METADATA.read_text(encoding="utf-8")
    )
    runtime = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    if "QWEN14_EXTERNAL_0702" in json.dumps(notebook):
        raise ValueError("Source notebook already contains this experiment")

    inference_source = f"""%%writefile qwen14_external_infer.py
{runtime}"""
    run_source = f"""# QWEN14_EXTERNAL_0702: external fold-0 adapter as a low-weight reader.
QWEN14_EXTERNAL_OUT = '/kaggle/working/qwen14_external.npz'
QWEN14_EXTERNAL_REPORT = {{'status': 'skipped', 'weight': {QWEN14_WEIGHT}}}
if not FAST_COMMIT:
    try:
        qwen14_checkpoint = os.path.dirname(
            first('/kaggle/input/**/qwen14_external/COMPLETE.json')
        )
        qwen14_cmd = [
            sys.executable,
            'qwen14_external_infer.py',
            '--model-dir', B14,
            '--checkpoint', qwen14_checkpoint,
            '--data', DATA,
            '--output', QWEN14_EXTERNAL_OUT,
            '--selected-layers', '2,4,6,8,12,16,20,24',
            '--num-layers', '24',
            '--batch-size', '2',
            '--max-length', '384',
        ]
        print('QWEN14_EXTERNAL start', qwen14_checkpoint, flush=True)
        qwen14_code = subprocess.call(
            qwen14_cmd,
            env=dict(
                os.environ,
                HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false',
            ),
        )
        if qwen14_code != 0:
            raise RuntimeError(
                f'Qwen14 external inference exited {{qwen14_code}}'
            )
        with np.load(QWEN14_EXTERNAL_OUT, allow_pickle=False) as payload:
            qwen14_ids = payload['ids'].astype(str).tolist()
            qwen14_labels = payload['labels'].astype(str).tolist()
        test_ids = pd.read_csv(
            os.path.join(DATA, 'test.csv')
        )['id'].astype(str).tolist()
        assert qwen14_ids == test_ids
        assert qwen14_labels == sorted(qwen14_labels)
        QWEN14_EXTERNAL_REPORT = {{
            'status': 'applied',
            'weight': {QWEN14_WEIGHT},
            'rows': len(qwen14_ids),
        }}
        print('QWEN14_EXTERNAL_READY', QWEN14_EXTERNAL_REPORT, flush=True)
    except Exception as error:
        QWEN14_EXTERNAL_REPORT = {{
            'status': 'failed',
            'weight': {QWEN14_WEIGHT},
            'error': repr(error),
        }}
        print(
            'QWEN14_EXTERNAL failed; preserving the original 0.702 path',
            QWEN14_EXTERNAL_REPORT,
            flush=True,
        )
print('QWEN14_EXTERNAL_REPORT', QWEN14_EXTERNAL_REPORT, flush=True)
"""

    inserted = False
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell.get("source", []))
        if "GLM_PAIRWISE_GATE" not in source:
            continue
        if cell.get("cell_type") != "code":
            raise ValueError("GLM pairwise marker is not a code cell")
        original = "        base_logp = _log_softmax(base_logits)\n        base_pred = base_logp.argmax(axis=1)"
        replacement = f"""        base_logp = _log_softmax(base_logits)
        qwen14_report = dict(QWEN14_EXTERNAL_REPORT)
        qwen14_path = Path('/kaggle/working/qwen14_external.npz')
        if qwen14_path.exists() and qwen14_report.get('status') == 'applied':
            try:
                with np.load(qwen14_path, allow_pickle=False) as payload:
                    qwen14_ids = payload['ids'].astype(str).tolist()
                    qwen14_labels = payload['labels'].astype(str).tolist()
                    qwen14_logits = payload['logits'].astype(np.float64)
                assert qwen14_ids == ids
                assert qwen14_labels == labels
                assert qwen14_logits.shape == base_logits.shape
                qwen14_weight = float(qwen14_report['weight'])
                base_logp = (
                    base_logp
                    + qwen14_weight * _log_softmax(qwen14_logits)
                )
                qwen14_report['status'] = 'fused'
            except Exception as error:
                qwen14_report = {{
                    **qwen14_report,
                    'status': 'failed',
                    'error': repr(error),
                }}
        base_pred = base_logp.argmax(axis=1)"""
        source = replace_once(source, original, replacement)
        source = replace_once(
            source,
            '            "likelihood_weight": likelihood_weight,\n        }',
            '            "likelihood_weight": likelihood_weight,\n'
            '            "qwen14_external": qwen14_report,\n        }',
        )
        notebook["cells"][index]["source"] = source.splitlines(
            keepends=True
        )
        notebook["cells"].insert(index, code_cell(inference_source))
        notebook["cells"].insert(index + 1, code_cell(run_source))
        inserted = True
        break
    if not inserted:
        raise RuntimeError("Could not find the GLM pairwise cell")

    metadata["id"] = KERNEL_ID
    metadata["title"] = KERNEL_TITLE
    metadata["code_file"] = OUTPUT_NOTEBOOK.name
    metadata["is_private"] = True
    sources = list(metadata.get("dataset_sources", []))
    if DATASET_SLUG not in sources:
        sources.append(DATASET_SLUG)
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
