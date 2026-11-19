from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class TextRows(Dataset):
    def __init__(self, texts, indices):
        self.texts = texts
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], int(self.indices[index])


class SourceEncoder(nn.Module):
    def __init__(self, input_dim, width=1024):
        super().__init__()
        import torch.nn.functional as F

        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(width, 512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, 256),
        )

    def forward(self, values):
        import torch.nn.functional as F

        return F.normalize(self.net(values), dim=-1)


def style_features(text):
    words = re.findall(r"[A-Za-z\u2019']+", text)
    sents = [
        value
        for value in re.split(r"(?<=[.!?])\s+", text.strip())
        if value
    ]
    nw = max(len(words), 1)
    ns = max(len(sents), 1)
    nc = max(len(text), 1)
    lengths = [len(value.split()) for value in sents] or [0]

    def per_100(pattern):
        return 100 * len(re.findall(pattern, text)) / nw

    return [
        float(nc),
        float(nw),
        float(ns),
        float(np.mean(lengths)),
        float(np.std(lengths)),
        float(max(lengths)),
        float(min(lengths)),
        float(np.mean([len(word) for word in words])) if words else 0.0,
        sum(len(word) >= 9 for word in words) / nw,
        len({word.lower() for word in words}) / nw,
        per_100("\u2014"),
        per_100(" \u2014 "),
        per_100("\u2013"),
        per_100(" - "),
        per_100("--"),
        per_100("\u2019"),
        per_100("'"),
        per_100("[\u201c\u201d]"),
        per_100('"'),
        per_100(";"),
        per_100(":"),
        per_100(","),
        per_100(r"\u2026|\.\.\."),
        per_100("!"),
        per_100(r"\?"),
        per_100(r"\("),
        100 * sum(ord(char) > 127 for char in text) / nc,
        100 * sum(char.isdigit() for char in text) / nc,
        sum(word[0].isupper() for word in words) / nw,
        per_100(r"\b(?:I|me|my|we|our|us)\b"),
        per_100(r"\b(?:[Yy]ou|[Yy]our)\b"),
        sum(
            sent.startswith(("And ", "But ", "Yet ", "So "))
            for sent in sents
        )
        / ns,
        sum(sent.startswith("The ") for sent in sents) / ns,
        float(text.rstrip().endswith(".")),
        per_100(r"\b(?:not|n\u2019t|n't)\b"),
        per_100(r"\b(?:like|as if|as though)\b"),
    ]


def summarize_hidden(hidden_states, attention_mask, layers):
    mask = attention_mask.bool()
    mask_f = mask.unsqueeze(-1).to(torch.float32)
    counts = mask_f.sum(dim=1).clamp_min(1.0)
    lengths = mask.sum(dim=1)
    batch_indices = torch.arange(mask.shape[0], device=mask.device)
    blocks = []
    for layer in layers:
        hidden = hidden_states[layer].float()
        mean = (hidden * mask_f).sum(dim=1) / counts
        centered = (hidden - mean.unsqueeze(1)) * mask_f
        std = (centered.square().sum(dim=1) / counts).sqrt()
        last = hidden[batch_indices, lengths - 1]
        delta = (hidden[:, 1:] - hidden[:, :-1]).abs()
        adjacent = (mask[:, 1:] & mask[:, :-1]).unsqueeze(-1)
        adjacent_count = adjacent.float().sum(dim=1).clamp_min(1.0)
        delta = (delta * adjacent).sum(dim=1) / adjacent_count
        blocks.extend([mean, std, last, delta])
    return torch.cat(blocks, dim=-1).cpu().numpy().astype(np.float32)


def collate(tokenizer, max_length):
    def inner(items):
        texts, indices = zip(*items)
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch["row_indices"] = torch.tensor(indices, dtype=torch.long)
        return batch

    return inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=384)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv)
    texts = frame["text"].fillna("").astype(str).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = DataLoader(
        TextRows(texts, np.arange(len(texts))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate(tokenizer, args.max_length),
        num_workers=0,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    backbone = AutoModel.from_pretrained(
        args.model,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    backbone.config.use_cache = False
    hidden_size = int(backbone.config.hidden_size)
    layers = [24, 36]
    features = np.empty(
        (len(texts), len(layers) * 4 * hidden_size),
        dtype=np.float16,
    )
    with torch.inference_mode():
        for batch in loader:
            positions = batch.pop("row_indices").numpy()
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                output = backbone(
                    **batch,
                    output_hidden_states=True,
                    return_dict=True,
                )
            features[positions] = summarize_hidden(
                output.hidden_states,
                batch["attention_mask"],
                layers,
            ).astype(np.float16)
    del backbone
    torch.cuda.empty_cache()

    checkpoint = torch.load(
        args.encoder,
        map_location="cpu",
        weights_only=False,
    )
    encoder = SourceEncoder(int(checkpoint["input_dim"])).to("cuda")
    encoder.load_state_dict(checkpoint["state_dict"])
    encoder.eval()
    style = np.asarray([style_features(text) for text in texts], np.float32)
    x_mean = checkpoint["x_mean"]
    x_std = checkpoint["x_std"]
    style_mean = checkpoint["style_mean"]
    style_std = checkpoint["style_std"]
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), 256):
            stop = min(start + 256, len(texts))
            xb = (features[start:stop].astype(np.float32) - x_mean) / x_std
            sb = (style[start:stop] - style_mean) / style_std
            values = torch.from_numpy(np.concatenate([xb, sb], axis=1)).to(
                "cuda"
            )
            embeddings.append(encoder(values).cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    state = joblib.load(args.classifier)
    probabilities = state["classifier"].predict_proba(
        state["scaler"].transform(embeddings)
    )
    logits = np.log(np.clip(probabilities, 1e-12, None))
    np.savez_compressed(
        args.output,
        ids=frame["id"].astype(str).to_numpy(),
        probabilities=probabilities.astype(np.float32),
        logits=logits.astype(np.float32),
        embeddings=embeddings.astype(np.float32),
    )
    print(
        f"saved {args.output} rows={len(texts)} "
        f"probs={probabilities.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
