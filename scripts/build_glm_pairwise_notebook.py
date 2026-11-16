from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("D:/tmp/namellm_public_fallback/notebook.ipynb")
DEFAULT_PAIR_MODEL = ROOT / "configs" / "glm_pairwise_model.json"


POSTPROCESS_TEMPLATE = r'''
# GLM_PAIRWISE_GATE: keep the V1 multiclass stage, then arbitrate only among GLM rows.
import base64
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_MODEL = json.loads(base64.b64decode("__PAIR_MODEL_B64__"))


def _log_softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    shifted = np.exp(values)
    return values - np.log(shifted.sum(axis=1, keepdims=True))


def _entropy(log_probabilities):
    return float(-(np.exp(log_probabilities) * log_probabilities).sum(axis=1).mean())


def _match_temperature(log_probabilities, target):
    low, high = 1e-3, 1e3
    for _ in range(100):
        middle = np.sqrt(low * high)
        if _entropy(_log_softmax(log_probabilities * middle)) > target:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def _merge_features(flash, glm53, ids):
    frame = (
        flash.set_index("id").add_prefix("flash:")
        .join(glm53.set_index("id").add_prefix("glm53:"), how="inner")
    )
    frame = frame.loc[ids]
    for column in ("lp_mean", "lp_q50", "ent_mean", "top1", "logrank_mean", "fdg"):
        frame[f"d:{column}"] = frame[f"flash:{column}"] - frame[f"glm53:{column}"]
    return frame


try:
    v1_dir = Path("/kaggle/temp/v1")
    feature_dir = Path("/kaggle/working/v1")
    flash_path = feature_dir / "flash_features.csv"
    glm_path = feature_dir / "glm53_features.csv"
    ll_model_path = v1_dir / "ll_model.json"
    if all(p.exists() for p in (flash_path, glm_path, ll_model_path)):
        ll_model = json.loads(ll_model_path.read_text())
        flash = pd.read_csv(flash_path, dtype={"id": str})
        glm53 = pd.read_csv(glm_path, dtype={"id": str})
        with np.load(FINAL_LOGITS, allow_pickle=False) as payload:
            ids = payload["ids"].astype(str).tolist()
            labels = payload["labels"].astype(str).tolist()
            base_logits = payload["logits"].astype(np.float64)
        common_ids = [
            value for value in ids
            if value in set(flash["id"]) & set(glm53["id"])
        ]
        assert common_ids, "no IDs have both GLM score tables"

        feature_frame = _merge_features(flash, glm53, common_ids)
        raw = feature_frame[ll_model["features"]].to_numpy(np.float64)
        scaled = (raw - np.asarray(ll_model["mean"])) / np.asarray(ll_model["scale"])
        member_logits = (
            scaled @ np.asarray(ll_model["coef"]).T
            + np.asarray(ll_model["intercept"])
        )
        member_logp = _log_softmax(member_logits)
        member_logp = member_logp[
            :, [ll_model["classes"].index(label) for label in labels]
        ]

        base_logp = _log_softmax(base_logits)
        base_pred = base_logp.argmax(axis=1)
        id_position = {value: index for index, value in enumerate(ids)}
        feature_rows = np.array([id_position[value] for value in common_ids], dtype=int)
        likelihood = np.zeros_like(base_logp)
        likelihood[feature_rows] = member_logp

        glm_labels = {"glm-5.3", "glm-5.3-flash"}
        glm_indices = [
            index for index, label in enumerate(labels)
            if label in glm_labels
        ]
        assert len(glm_indices) == 2
        mass = np.exp(base_logp[:, glm_indices]).sum(axis=1)
        order = np.argsort(-mass, kind="stable")
        scored_set = set(feature_rows.tolist())
        selected = np.array(
            [index for index in order if index in scored_set],
            dtype=int,
        )
        temperature = _match_temperature(
            base_logp,
            float(ll_model["entropy_ref"]),
        )
        likelihood_weight = float(ll_model["W"]) / temperature

        prediction = base_pred.copy()
        prediction[selected] = (
            base_logp[selected] + likelihood_weight * likelihood[selected]
        ).argmax(axis=1)
        gate = np.array(
            [index for index in selected if labels[prediction[index]] in glm_labels],
            dtype=int,
        )

        pair_frame = feature_frame.loc[[ids[index] for index in gate]]
        pair_raw = pair_frame[PAIR_MODEL["features"]].to_numpy(np.float64)
        pair_scaled = (
            pair_raw - np.asarray(PAIR_MODEL["mean"])
        ) / np.asarray(PAIR_MODEL["scale"])
        pair_logit = (
            pair_scaled @ np.asarray(PAIR_MODEL["coef"])
            + float(PAIR_MODEL["intercept"])
        )
        pair_probability = 1.0 / (
            1.0 + np.exp(-np.clip(pair_logit, -40, 40))
        )
        glm53_index = labels.index(PAIR_MODEL["positive_label"])
        glmflash_index = next(
            index for index in glm_indices if index != glm53_index
        )
        prediction[gate] = np.where(
            pair_probability >= 0.5,
            glm53_index,
            glmflash_index,
        )

        changed = float((prediction != base_pred).mean())
        if changed > 0.25:
            raise RuntimeError(f"pairwise gate changed {changed:.1%} of rows")
        submission = pd.DataFrame(
            {
                "id": ids,
                "model": np.asarray(labels)[prediction],
            }
        )
        assert len(submission) == len(ids) and submission["id"].is_unique
        submission.to_parquet(OUT, index=False)
        report = {
            "status": "applied",
            "scored_rows": len(common_ids),
            "gated_rows": len(gate),
            "changed": changed,
            "temperature": temperature,
            "likelihood_weight": likelihood_weight,
        }
    else:
        report = {"status": "skipped", "reason": "V1 feature files missing"}
    (feature_dir / "pairwise_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print("GLM_PAIRWISE_GATE", json.dumps(report, indent=2), flush=True)
except Exception:
    traceback.print_exc()
    print("GLM_PAIRWISE_GATE failed; keeping the existing V1 submission", flush=True)
'''


def replace_v1_likelihood_model(
    notebook: dict,
    ll_model_path: Path,
) -> str:
    model_bytes = ll_model_path.read_bytes()
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "V1_PAYLOAD =" not in source:
            continue
        payload_match = re.search(
            r"V1_PAYLOAD\s*=\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")",
            source,
        )
        if payload_match is None:
            continue
        raw = base64.b64decode(ast.literal_eval(payload_match.group(1)))
        members = []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            for member in archive.getmembers():
                extent = archive.extractfile(member)
                data = extent.read() if extent is not None else b""
                if member.name == "ll_model.json":
                    data = model_bytes
                members.append((member, data))
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w:gz") as archive:
            for member, data in members:
                info = tarfile.TarInfo(member.name)
                info.mode = member.mode
                info.mtime = member.mtime
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        digest = hashlib.sha256(output.getvalue()).hexdigest()
        source = (
            source[: payload_match.start(1)]
            + repr(encoded)
            + source[payload_match.end(1) :]
        )
        source, replacements = re.subn(
            r'assert hashlib\.sha256\(_raw\)\.hexdigest\(\) == "[0-9a-f]+"',
            (
                'assert hashlib.sha256(_raw).hexdigest() == '
                f'"{digest}"'
            ),
            source,
        )
        if replacements != 1:
            raise ValueError("Could not update the V1 payload hash assertion")
        cell["source"] = [line + "\n" for line in source.splitlines()]
        return digest
    raise ValueError("Could not locate the V1 payload")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--pair-model", default=str(DEFAULT_PAIR_MODEL))
    parser.add_argument(
        "--ll-model",
        help="Optional replacement ll_model.json inside the V1 payload.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ref",
        default="yyyyhhhhwww/name-that-llm-glm-pairwise",
    )
    parser.add_argument(
        "--title",
        default="Name That LLM GLM Pairwise",
    )
    args = parser.parse_args()

    source = Path(args.source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    if "cells" not in notebook:
        raise ValueError("Source is not a notebook")
    pair_payload = Path(args.pair_model).read_bytes()
    pair_b64 = base64.b64encode(pair_payload).decode("ascii")
    postprocess = POSTPROCESS_TEMPLATE.replace(
        "__PAIR_MODEL_B64__",
        pair_b64,
    ).strip()
    payload_sha = None
    if args.ll_model:
        payload_sha = replace_v1_likelihood_model(
            notebook,
            Path(args.ll_model),
        )

    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if "GLM_PAIRWISE_GATE" not in "".join(cell.get("source", []))
    ]
    insert_at = None
    for index, cell in enumerate(notebook["cells"]):
        source_text = "".join(cell.get("source", []))
        if "V1 stage: likelihood of the test" in source_text:
            insert_at = index + 1
            break
    if insert_at is None:
        raise ValueError("Could not locate the V1 likelihood stage")
    notebook["cells"].insert(
        insert_at,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                line + "\n"
                for line in postprocess.splitlines()
            ],
        },
    )

    metadata_source = source.with_name("kernel-metadata.json")
    metadata = (
        json.loads(metadata_source.read_text(encoding="utf-8"))
        if metadata_source.exists()
        else {}
    )
    metadata.update(
        {
            "id": args.ref,
            "title": args.title[:50],
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "notebook.ipynb").write_text(
        json.dumps(notebook, indent=2),
        encoding="utf-8",
    )
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(
        f"built {output_dir} cells={len(notebook['cells'])} "
        f"inserted={insert_at} "
        f"payload_sha256={payload_sha or 'unchanged'}"
    )


if __name__ == "__main__":
    main()
