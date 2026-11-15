from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from transformers import AutoModel, AutoTokenizer

ARTIFACT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACT_DIR))

try:
    from clustering import blind_prompt_clusters, group_relative
    from constants import LABELS
    from lm_stats import LMStatConfig, score_texts
    from style import style_frame
except ModuleNotFoundError:
    from namellm.clustering import blind_prompt_clusters, group_relative
    from namellm.constants import LABELS
    from namellm.lm_stats import LMStatConfig, score_texts
    from namellm.style import style_frame


class LayerSequenceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        extra_size: int,
        hidden_size: int,
        layer_count: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.position = nn.Parameter(torch.zeros(1, layer_count, hidden_size))
        self.encoder = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 6 + extra_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 6 + extra_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(LABELS)),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        extra: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.input_projection(sequence) + self.position
        output, _ = self.encoder(encoded)
        pooled = torch.cat(
            [
                output[:, -1, : self.hidden_size],
                output[:, 0, self.hidden_size :],
                output.mean(dim=1),
                output.max(dim=1).values,
            ],
            dim=-1,
        )
        return self.classifier(torch.cat([pooled, extra], dim=-1))


def find_input_file(filename: str) -> Path:
    for root in [
        Path("/kaggle/input/competitions/name-that-llm"),
        Path("/kaggle/input/name-that-llm"),
    ]:
        candidate = root / filename
        if candidate.exists():
            return candidate
    matches = list(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(filename)
    return matches[0]


def split_layers(
    features: np.ndarray,
    layers: list[int],
    wanted: list[int],
) -> np.ndarray:
    hidden_size = features.shape[1] // (len(layers) * 4)
    block_size = hidden_size * 4
    blocks = {
        layer: features[
            :,
            index * block_size : (index + 1) * block_size,
        ]
        for index, layer in enumerate(layers)
    }
    return np.concatenate([blocks[layer] for layer in wanted], axis=1)


def extra_features(
    frame: pd.DataFrame,
    clusters: np.ndarray,
    lm: np.ndarray,
) -> np.ndarray:
    style = style_frame(frame["text"]).to_numpy(dtype=np.float32)
    return np.hstack(
        [
            style,
            group_relative(style, clusters),
            lm,
            group_relative(lm, clusters),
        ]
    )


def normalize(probabilities: np.ndarray) -> np.ndarray:
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def center_by_texts(
    probabilities: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    normalized = normalize(probabilities)
    logs = np.log(np.clip(normalized, 1e-12, None))
    clusters = blind_prompt_clusters(texts, 0.90)
    centered = logs.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= 0.5 * logs[mask].mean(axis=0, keepdims=True)
    return centered


def rich_features(
    branches: list[np.ndarray],
    centered: list[np.ndarray],
) -> np.ndarray:
    normalized = [normalize(branch) for branch in branches]
    group_names = ["ClaudeGLM", "GPTKimiQwen", "Gemini"]

    def group_name(label: str) -> str:
        family = label.split("-")[0].replace("qwen3.8", "qwen")
        if family in {"claude", "glm"}:
            return "ClaudeGLM"
        if family == "gemini":
            return "Gemini"
        return "GPTKimiQwen"

    top_one_hot = []
    confidence = []
    group_mass = []
    for branch in normalized:
        one_hot = np.zeros_like(branch)
        one_hot[np.arange(len(branch)), branch.argmax(axis=1)] = 1
        top_one_hot.append(one_hot)
        confidence.extend(
            [
                branch.max(axis=1, keepdims=True),
                -(
                    branch * np.log(np.clip(branch, 1e-12, None))
                ).sum(axis=1, keepdims=True),
            ]
        )
        group_mass.append(
            np.stack(
                [
                    branch[
                        :,
                        [
                            index
                            for index, label in enumerate(LABELS)
                            if group_name(label) == name
                        ],
                    ].sum(axis=1)
                    for name in group_names
                ],
                axis=1,
            )
        )
    return np.hstack(
        [
            *normalized,
            *[np.log(np.clip(branch, 1e-12, None)) for branch in normalized],
            *centered,
            *top_one_hot,
            *confidence,
            *group_mass,
        ]
    ).astype(np.float32)


def balance_probabilities(
    probabilities: np.ndarray,
    texts: list[str],
) -> np.ndarray:
    clusters = blind_prompt_clusters(texts, 0.90)
    result = normalize(probabilities)
    logs = np.log(np.clip(result, 1e-12, None))
    centered = logs.copy()
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        centered[mask] -= 0.5 * logs[mask].mean(axis=0, keepdims=True)
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


def extract_embedding_features(
    text: list[str],
    model_dir: str,
    max_length: int,
) -> tuple[np.ndarray, list[int]]:
    device = torch.device("cuda")
    dtype = (
        torch.float16
        if torch.cuda.get_device_capability(device)[0] < 8
        else torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_dir,
        dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    layers = [8, 16, 20, 24, 28, 32, 36]
    hidden_size = int(model.config.hidden_size)
    output_features = np.empty(
        (len(text), len(layers) * 4 * hidden_size),
        dtype=np.float32,
    )
    with torch.inference_mode():
        for index, value in enumerate(text):
            encoded = tokenizer(
                value,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded = {
                name: tensor.to(device)
                for name, tensor in encoded.items()
            }
            with torch.autocast("cuda", dtype=dtype):
                output = model(
                    **encoded,
                    output_hidden_states=True,
                    return_dict=True,
                )
            mask = encoded["attention_mask"].bool()
            mask_float = mask.unsqueeze(-1).float()
            counts = mask_float.sum(dim=1).clamp_min(1)
            length = mask.sum(dim=1)
            blocks = []
            for layer in layers:
                hidden = output.hidden_states[layer].float()
                mean = (hidden * mask_float).sum(dim=1) / counts
                centered = (hidden - mean.unsqueeze(1)) * mask_float
                std = (centered.square().sum(dim=1) / counts).sqrt()
                last = hidden[0, length - 1]
                if hidden.shape[1] > 1:
                    differences = (hidden[:, 1:] - hidden[:, :-1]).abs()
                    adjacent = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
                    delta = (
                        differences * adjacent
                    ).sum(dim=1) / adjacent.float().sum(dim=1).clamp_min(1)
                else:
                    delta = torch.zeros_like(mean)
                blocks.extend([mean[0], std[0], last, delta[0]])
            output_features[index] = (
                torch.cat(blocks).cpu().numpy().astype(np.float32)
            )
            if (index + 1) % 250 == 0 or index + 1 == len(text):
                print(f"embedding {index + 1}/{len(text)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return output_features, layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--lm-model", required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--output", default="/kaggle/working/submission.parquet")
    args = parser.parse_args()

    train = pd.read_csv(find_input_file("train.csv"))
    test = pd.read_csv(find_input_file("test.csv"))
    sample = pd.read_parquet(find_input_file("sample_submission.parquet"))
    test_clusters = blind_prompt_clusters(test["text"].tolist(), 0.90)

    lm_features = score_texts(
        test["id"].astype(str).tolist(),
        test["text"].astype(str).tolist(),
        LMStatConfig(
            model_name=args.lm_model,
            max_length=args.max_length,
            load_in_4bit=False,
        ),
    )
    lm_values = lm_features.drop(columns="id").to_numpy(dtype=np.float32)
    gc.collect()
    torch.cuda.empty_cache()
    test_extra = extra_features(test, test_clusters, lm_values)

    compact = joblib.load(ARTIFACT_DIR / "compact_tfidf.joblib")
    compact_matrix = compact["char"].transform(test["text"])
    compact_matrix = hstack(
        [compact_matrix, compact["word"].transform(test["text"])]
    ).tocsr()
    compact_probabilities = compact["model"].predict_proba(compact_matrix)

    test_feature_values, layers = extract_embedding_features(
        test["text"].astype(str).tolist(),
        args.embedding_model,
        args.max_length,
    )
    linear_artifacts = joblib.load(
        ARTIFACT_DIR / "embedding_linear.joblib"
    )
    branch_probabilities = [compact_probabilities]
    for name in ["e512", "e256", "layer4"]:
        artifact = linear_artifacts[name]
        selected = split_layers(
            test_feature_values,
            layers,
            artifact["wanted_layers"],
        )
        pca_test = artifact["pca"].transform(
            artifact["scaler"].transform(selected)
        )
        combined = np.hstack(
            [
                pca_test,
                group_relative(pca_test, test_clusters),
                test_extra,
            ]
        )
        branch_probabilities.append(
            artifact["classifier"].predict_proba(
                artifact["combined_scaler"].transform(combined)
            )
        )

    hidden_size = test_feature_values.shape[1] // (len(layers) * 4)
    sequence_features = test_feature_values.reshape(
        len(test),
        len(layers),
        4 * hidden_size,
    )
    gru_transforms = joblib.load(ARTIFACT_DIR / "gru7_transforms.joblib")
    sequence_arrays = []
    for index, transform in enumerate(gru_transforms["layer_transforms"]):
        pca_test = transform["pca"].transform(
            transform["scaler"].transform(sequence_features[:, index])
        )
        sequence_arrays.append(
            np.hstack(
                [
                    pca_test,
                    group_relative(pca_test, test_clusters),
                ]
            ).astype(np.float32)
        )
    gru_sequence = np.stack(sequence_arrays, axis=1)
    gru_extra = gru_transforms["extra_scaler"].transform(test_extra).astype(
        np.float32
    )
    gru_checkpoint = torch.load(
        ARTIFACT_DIR / "gru7_model.pt",
        map_location="cuda",
        weights_only=False,
    )
    gru_model = LayerSequenceModel(
        input_size=gru_sequence.shape[-1],
        extra_size=test_extra.shape[1],
        hidden_size=128,
        layer_count=len(layers),
    ).cuda()
    gru_model.load_state_dict(gru_checkpoint["state_dict"])
    gru_model.eval()
    with torch.inference_mode():
        logits = gru_model(
            torch.from_numpy(gru_sequence).cuda(),
            torch.from_numpy(gru_extra).cuda(),
        )
        branch_probabilities.append(
            torch.softmax(logits.float(), dim=-1).cpu().numpy()
        )
    del gru_model
    gc.collect()
    torch.cuda.empty_cache()

    centered = [
        center_by_texts(branch, test["text"].tolist())
        for branch in branch_probabilities
    ]
    stack_features = rich_features(branch_probabilities, centered)
    stack = joblib.load(ARTIFACT_DIR / "rich_stack.joblib")
    probabilities = stack["model"].predict_proba(
        stack["scaler"].transform(stack_features)
    )
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
    submission.to_parquet(args.output, index=False)
    print(submission.head(20).to_string(index=False))
    print(f"saved {args.output} rows={len(submission)}")


if __name__ == "__main__":
    main()
