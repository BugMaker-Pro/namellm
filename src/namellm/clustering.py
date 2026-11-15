from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer


def blind_prompt_clusters(
    texts: list[str],
    distance_threshold: float = 0.90,
) -> np.ndarray:
    """Recover prompt-like topics without using a prompt_id.

    Average-linkage cosine clustering intentionally over-segments. Keeping two
    related prompts separate is safer than merging unrelated passages.
    """
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    features = vectorizer.fit_transform(texts)
    distances = np.clip(1 - (features @ features.T).toarray(), 0, 2)
    np.fill_diagonal(distances, 0)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    return model.fit_predict(distances)


def group_relative(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    result = matrix.copy()
    for group_id in np.unique(groups):
        mask = groups == group_id
        result[mask] -= result[mask].mean(axis=0, keepdims=True)
    return result

