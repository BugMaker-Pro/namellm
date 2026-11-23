from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CLEAN_CELL = r'''
from pathlib import Path
import gc
import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

clean_roots = [
    Path('/kaggle/input/ntl-clean-contrastive-e14'),
    Path('/kaggle/input/datasets/yyyyhhhhwww/ntl-clean-contrastive-e14'),
]
CLEAN_ROOT = next((path for path in clean_roots if path.is_dir()), None)
assert CLEAN_ROOT is not None, clean_roots
for path in (CLEAN_ROOT / 'fold0', CLEAN_ROOT / 'fold4'):
    assert (path / 'model_state.pt').is_file(), path
assert (CLEAN_ROOT / 'predict_clean_contrastive.py').is_file()

try:
    import bitsandbytes
except ImportError:
    wheels = asset_paths('/kaggle/input/**/bitsandbytes-*.whl')
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--no-index', wheels[0]]
    )

sys.path.insert(0, str(CLEAN_ROOT))
sys.path.insert(0, '/kaggle/working')
clean_runtime = importlib.import_module('predict_clean_contrastive')

clean_query = pd.read_csv(V1_TEST_CSV, dtype={'id': str})
gc.collect()
torch.cuda.empty_cache()
clean_ids, clean_probabilities = clean_runtime.predict_frame(
    clean_query,
    checkpoint('qwen3', 2560, None),
    CLEAN_ROOT,
    batch_size=4,
    max_length=384,
)
assert clean_ids.tolist() == clean_query.id.astype(str).tolist()
assert clean_probabilities.shape == (len(clean_query), 18)
assert np.isfinite(clean_probabilities).all()
assert (clean_probabilities >= 0).all()
assert np.allclose(clean_probabilities.sum(1), 1.0)
np.savez_compressed(
    '/kaggle/working/clean_contrastive_probabilities.npz',
    ids=clean_ids,
    labels=np.asarray(clean_runtime.LABELS),
    probabilities=clean_probabilities,
)
'''


BLEND_CELL = r'''
with np.load(FINAL_LOGITS, allow_pickle=False) as saved:
    parent_ids = saved['ids'].astype(str)
    parent_labels = saved['labels'].astype(str).tolist()
    parent_logits = np.asarray(saved['logits'], dtype=np.float64)

assert parent_ids.tolist() == clean_ids.tolist()
assert parent_logits.shape == (len(parent_ids), 18)
assert set(parent_labels) == set(clean_runtime.LABELS)

parent_probabilities = softmax(parent_logits, axis=1)
clean_aligned = clean_probabilities[
    :, [clean_runtime.LABELS.index(label) for label in parent_labels]
]
parent_prediction = np.asarray(parent_labels)[parent_probabilities.argmax(1)]
excluded = {'gpt-5.6-terra', 'claude-fable-5'}
active = np.asarray(
    [value not in excluded for value in parent_prediction],
    dtype=bool,
)
blended = parent_probabilities.copy()
blended[active] = (
    0.82 * parent_probabilities[active]
    + 0.18 * clean_aligned[active]
)
blended /= blended.sum(axis=1, keepdims=True)

parent_backup = '/kaggle/working/final-logits-parent.npz'
if not Path(parent_backup).exists():
    shutil.copyfile(FINAL_LOGITS, parent_backup)
np.savez(
    FINAL_LOGITS,
    ids=parent_ids,
    labels=np.asarray(parent_labels),
    logits=np.log(np.clip(blended, 1e-12, 1.0)),
)
report = {
    'active_rows': int(active.sum()),
    'changed_argmax': int(
        np.sum(blended.argmax(1) != parent_probabilities.argmax(1))
    ),
    'weight': 0.18,
    'excluded_parent_predictions': sorted(excluded),
}
Path('/kaggle/working/clean_contrastive_report.json').write_text(
    json.dumps(report, indent=2),
    encoding='utf-8',
)
print('CLEAN_CONTRASTIVE_REPORT', report, flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="kaggle_0704_control",
    )
    parser.add_argument(
        "--output",
        default="kaggle_clean_contrastive_0704",
    )
    args = parser.parse_args()
    source = ROOT / args.source
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    notebook = json.loads(
        (source / "ntl-0704-control.ipynb").read_text(
            encoding="utf-8"
        )
    )
    insert_at = 20
    clean_cells = []
    for source_value in (CLEAN_CELL, BLEND_CELL):
        clean_cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source_value.strip() + "\n",
            }
        )
    notebook["cells"][insert_at:insert_at] = clean_cells
    notebook_path = output / "ntl-0704-clean-contrastive.ipynb"
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    metadata = json.loads(
        (source / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    metadata["id"] = "yyyyhhhhwww/name-that-llm-0704-clean-contrastive"
    metadata["title"] = "NTL 0704 Clean Contrastive"
    metadata["code_file"] = notebook_path.name
    metadata["is_private"] = True
    metadata["enable_gpu"] = True
    dataset_id = "yyyyhhhhwww/ntl-clean-contrastive-e14"
    if dataset_id not in metadata["dataset_sources"]:
        metadata["dataset_sources"].append(dataset_id)
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    shutil.copyfile(
        ROOT / "scripts" / "push_kaggle_kernel.py",
        output / "push_kaggle_kernel.py",
    )
    print(notebook_path)
    print(output / "kernel-metadata.json")


if __name__ == "__main__":
    main()
