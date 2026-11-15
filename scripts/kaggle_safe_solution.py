from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]


def find_input_file(filename: str) -> Path:
    candidates = [
        Path("/kaggle/input/name-that-llm") / filename,
        Path("/kaggle/input") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


def blind_prompt_clusters(
    texts: list[str],
    distance_threshold: float = 0.90,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    features = vectorizer.fit_transform(texts)
    distances = np.clip(
        1 - (features @ features.T).toarray(),
        0,
        2,
    )
    np.fill_diagonal(distances, 0)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(distances)


def balance_probabilities(
    probabilities: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    clusters = blind_prompt_clusters(texts)
    normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
    log_probabilities = np.log(np.clip(normalized, 1e-12, None))
    centered = log_probabilities.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= 0.5 * log_probabilities[mask].mean(
            axis=0,
            keepdims=True,
        )
    centered -= centered.max(axis=1, keepdims=True)
    result = np.exp(centered)
    result /= result.sum(axis=1, keepdims=True)

    for _ in range(5):
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if mask.sum() < 8:
                continue
            block = result[mask]
            target = len(block) / len(LABELS)
            scale = target / np.clip(
                block.sum(axis=0, keepdims=True),
                1e-9,
                None,
            )
            block *= np.power(np.clip(scale, 1e-6, None), 0.625)
            block /= np.clip(block.sum(axis=1, keepdims=True), 1e-9, None)
            result[mask] = block
    return result


train = pd.read_csv(find_input_file("train.csv"))
test = pd.read_csv(find_input_file("test.csv"))
sample_path = find_input_file("sample_submission.parquet")
sample = pd.read_parquet(sample_path)

label_to_index = {label: index for index, label in enumerate(LABELS)}
labels = np.array([label_to_index[value] for value in train["model"]])

char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(1, 5),
    sublinear_tf=True,
    min_df=2,
    lowercase=False,
    dtype=np.float32,
).fit(train["text"])
word = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
).fit(train["text"])

train_features = hstack(
    [char.transform(train["text"]), word.transform(train["text"])]
).tocsr()
test_features = hstack(
    [char.transform(test["text"]), word.transform(test["text"])]
).tocsr()
classifier = LogisticRegression(C=3.0, max_iter=3000)
classifier.fit(train_features, labels)
probabilities = classifier.predict_proba(test_features)
probabilities = balance_probabilities(
    probabilities,
    test["text"].tolist(),
)

predictions = pd.DataFrame(
    {
        "id": test["id"],
        "model": np.array(LABELS)[probabilities.argmax(axis=1)],
    }
)
submission = sample[["id"]].merge(predictions, on="id", how="left")
if submission["model"].isna().any():
    raise ValueError("Missing predictions")
submission.to_parquet("/kaggle/working/submission.parquet", index=False)
print(submission.to_string(index=False))
