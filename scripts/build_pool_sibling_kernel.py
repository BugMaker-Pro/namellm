from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "artifacts" / "kaggle_0701_pairwise"
SOURCE_NOTEBOOK = SOURCE_DIR / "name-that-llm-0701-glm-pairwise.ipynb"
SOURCE_METADATA = SOURCE_DIR / "kernel-metadata.json"
OUTPUT_DIR = ROOT / "kaggle_pool_siblings_0702"
OUTPUT_NOTEBOOK = OUTPUT_DIR / "name-that-llm-0702-pool-siblings.ipynb"
OUTPUT_METADATA = OUTPUT_DIR / "kernel-metadata.json"


WORKER_SOURCE = r'''%%writefile spec_worker.py
"""Train Open Pool sibling specialists and score train/test passages."""
import argparse
import glob
import hashlib
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier


def normalize_text(value):
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")),
    ).strip()


def text_key(value):
    return hashlib.md5(
        normalize_text(value).encode("utf-8")
    ).hexdigest()


def find_pool():
    hits = glob.glob(
        "/kaggle/input/**/ntl_open_pool.parquet",
        recursive=True,
    )
    if len(hits) != 1:
        raise RuntimeError(hits)
    return Path(hits[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("test_csv")
    parser.add_argument("train_csv")
    parser.add_argument("output")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    started = time.time()
    train = pd.read_csv(args.train_csv)
    train["text"] = train.text.fillna("").astype(str).map(normalize_text)
    test = pd.read_csv(args.test_csv, dtype={"id": str})
    labels = sorted(train.model.unique())
    train_text = train.text.fillna("").astype(str).map(normalize_text)
    test_text = test.text.fillna("").astype(str).map(normalize_text)
    official_keys = set(train_text.map(text_key)) | set(
        test_text.map(text_key)
    )

    pool = pd.read_parquet(find_pool())
    pool = pool[pool.model.isin(labels)].copy()
    for column in (
        "flag_train_prompt_regen",
        "flag_near_train_prompt",
        "flag_gemini37_routing_risk",
        "flag_kb_proxy_after_ruling",
    ):
        if column in pool.columns:
            pool = pool[~pool[column].fillna(False).astype(bool)]
    if "gen_served_model" in pool.columns:
        served = pool.gen_served_model.fillna("").astype(str)
        pool = pool[served.eq("") | served.eq(pool.model.astype(str))]
    if "n_words" in pool.columns:
        pool = pool[pool.n_words.between(36, 275)]
    pool["text"] = pool.text.fillna("").astype(str).map(normalize_text)
    pool = pool[pool.text.ne("")]
    pool["_key"] = pool.text.map(text_key)
    pool = pool[~pool._key.isin(official_keys)].drop_duplicates("_key")
    cap = 200 if args.smoke else 12000
    pool = (
        pool.sample(frac=1.0, random_state=20261118)
        .groupby("model", group_keys=False, sort=False)
        .head(cap)
    )
    print(
        f"specialist pool rows={len(pool)} counts="
        f"{pool.model.value_counts().sort_index().to_dict()}",
        flush=True,
    )

    arrays = {"ids": np.asarray(test.id.tolist(), dtype=str)}
    for lab in ("gemini", "gpt", "claude"):
        siblings = [
            label for label in labels
            if label.startswith(lab + "-")
        ]
        local = pool[pool.model.isin(siblings)].copy()
        official = train[train.model.isin(siblings)][["text", "model"]]
        local = pd.concat([local, official], ignore_index=True)
        if local.empty:
            raise RuntimeError(f"No specialist rows for {lab}")
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 5),
            sublinear_tf=True,
            min_df=3,
            max_features=600000,
            lowercase=False,
            dtype=np.float32,
        )
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_features=300000,
            token_pattern=r"(?u)\b\w+\b|[^\w\s]",
            lowercase=False,
            dtype=np.float32,
        )
        fit_x = sparse.hstack(
            [char.fit_transform(local.text), word.fit_transform(local.text)],
            format="csr",
        )
        query_x = sparse.hstack(
            [char.transform(test_text), word.transform(test_text)],
            format="csr",
        )
        model = SGDClassifier(
            loss="log_loss",
            alpha=2e-6,
            max_iter=20,
            tol=None,
            class_weight="balanced",
            random_state=20261118,
        ).fit(fit_x, local.model.to_numpy())
        classes = np.asarray(model.classes_, dtype=str)
        probabilities = model.predict_proba(query_x)
        scores = np.log(np.clip(probabilities, 1e-9, None))
        scores -= scores.mean(axis=1, keepdims=True)
        arrays[lab] = scores
        arrays[lab + "_labels"] = classes
        print(
            f"specialist {lab} rows={len(local)} "
            f"shape={fit_x.shape} classes={classes.tolist()} "
            f"seconds={time.time() - started:.0f}",
            flush=True,
        )
    np.savez(args.output, **arrays)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
'''


APPLY_SOURCE = r'''# Open Pool sibling specialists: preserve each lab's total mass.
from pathlib import Path
from scipy.special import log_softmax as _lsm

META_CONFIG = __META_CONFIG__
SPEC_OUT = "/kaggle/temp/spec_scores.npz"
_spec_cmd = [
    sys.executable,
    "spec_worker.py",
    os.path.join(DATA, "test.csv"),
    os.path.join(DATA, "train.csv"),
    SPEC_OUT,
]
if FAST_COMMIT:
    _spec_cmd.append("--smoke")
subprocess.run(_spec_cmd, check=True)

def spec_apply(logits_path, specialist_path):
    with np.load(logits_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    labels = arrays["labels"].astype(str).tolist()
    logits = np.asarray(arrays["logits"], dtype=np.float64).copy()
    with np.load(specialist_path, allow_pickle=False) as specialist:
        assert (
            specialist["ids"].astype(str).tolist()
            == arrays["ids"].astype(str).tolist()
        )
        base_logp = _lsm(logits, axis=1)
        for family, model in META_CONFIG["families"].items():
            siblings = [str(value) for value in model["labels"]]
            columns = [labels.index(label) for label in siblings]
            specialist_labels = specialist[
                family + "_labels"
            ].astype(str).tolist()
            specialist_columns = [
                specialist_labels.index(label)
                for label in siblings
            ]
            specialist_values = np.asarray(
                specialist[family],
                dtype=np.float64,
            )[:, specialist_columns]
            features = np.hstack(
                [
                    base_logp[:, columns],
                    specialist_values,
                ]
            )
            scaled = (
                features - np.asarray(model["mean"], dtype=np.float64)
            ) / np.asarray(model["scale"], dtype=np.float64)
            local = scaled @ np.asarray(
                model["coef"],
                dtype=np.float64,
            ).T + np.asarray(model["intercept"], dtype=np.float64)
            local = np.exp(
                local - local.max(axis=1, keepdims=True)
            )
            local /= local.sum(axis=1, keepdims=True)
            class_labels = [
                labels[int(index)]
                for index in model["classes"]
            ]
            order = [class_labels.index(label) for label in siblings]
            local = local[:, order]
            before = logits.argmax(axis=1)
            lab_mass = np.logaddexp.reduce(
                logits[:, columns],
                axis=1,
                keepdims=True,
            )
            logits[:, columns] = lab_mass + np.log(
                np.clip(local, 1e-12, None)
            )
            print(
                f"specialist {family}: meta-stack "
                f"changed={int((logits.argmax(axis=1) != before).sum())}",
                flush=True,
            )
    assert np.isfinite(logits).all()
    arrays["logits"] = logits.astype(np.asarray(arrays["logits"]).dtype)
    return arrays


if FAST_COMMIT:
    smoke_path = Path("/kaggle/temp/spec-smoke-logits.npz")
    smoke_ids = np.load(SPEC_OUT, allow_pickle=False)["ids"]
    smoke_labels = np.asarray(
        sorted(pd.read_csv(os.path.join(DATA, "train.csv")).model.unique())
    )
    np.savez(
        smoke_path,
        ids=smoke_ids,
        labels=smoke_labels,
        logits=np.random.default_rng(0).normal(
            size=(len(smoke_ids), len(smoke_labels))
        ),
    )
    spec_apply(str(smoke_path), SPEC_OUT)
    print("specialist application smoke passed", flush=True)
elif os.path.exists(FINAL_LOGITS):
    np.savez(FINAL_LOGITS, **spec_apply(FINAL_LOGITS, SPEC_OUT))
else:
    raise FileNotFoundError(FINAL_LOGITS)
'''


def make_cell(cell_type: str, source: str) -> dict[str, object]:
    cell: dict[str, object] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    meta_config = json.loads(
        (ROOT / "configs" / "family_meta_stack_c03.json").read_text(
            encoding="utf-8"
        )
    )
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    markers = (
        "%%writefile spec_worker.py",
        "# Open Pool sibling specialists: preserve",
    )
    cells = [
        cell
        for cell in cells
        if not any(
            "".join(cell.get("source", [])).startswith(marker)
            for marker in markers
        )
    ]
    insert_at = next(
        index
        for index, cell in enumerate(cells)
        if "GLM_PAIRWISE_GATE" in "".join(cell.get("source", []))
    )
    additions = [
        make_cell(
            "markdown",
            "## Open Pool sibling specialists\n\n"
            "Train one balanced char+word TF-IDF specialist for each "
            "Claude, GPT and Gemini sibling set on the filtered Open Pool. "
            "The specialists update only their lab's log-probability columns "
            "and preserve the lab's total mass.\n",
        ),
        make_cell("code", WORKER_SOURCE),
        make_cell(
            "code",
            APPLY_SOURCE.replace(
                "__META_CONFIG__",
                json.dumps(meta_config),
            ),
        ),
    ]
    cells[insert_at:insert_at] = additions
    notebook["cells"] = cells
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))
    metadata.pop("id_no", None)
    metadata["id"] = "yyyyhhhhwww/name-that-llm-0702-pool-siblings"
    metadata["title"] = "Name That LLM 0702 Pool Siblings"
    metadata["code_file"] = OUTPUT_NOTEBOOK.name
    metadata["is_private"] = True
    metadata["enable_gpu"] = True
    metadata["enable_internet"] = False
    OUTPUT_METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT_NOTEBOOK}")
    print(f"wrote {OUTPUT_METADATA}")
    print(f"cells={len(cells)} inserted_at={insert_at}")


if __name__ == "__main__":
    main()
