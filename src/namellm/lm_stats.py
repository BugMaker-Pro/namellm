from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass(frozen=True)
class LMStatConfig:
    model_name: str = "Qwen/Qwen3-1.7B-Base"
    max_length: int = 384
    logit_chunk: int = 32
    load_in_4bit: bool = False


def _passage_features(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    n_chars: int,
    logit_chunk: int,
) -> dict[str, float]:
    token_count = int(target_ids.numel())
    actual = logits.gather(-1, target_ids[:, None]).squeeze(-1)
    ranks = (logits > actual[:, None]).sum(dim=-1)
    log_norm = torch.logsumexp(logits.float(), dim=-1)
    lp = actual.float() - log_norm

    entropies: list[torch.Tensor] = []
    for start in range(0, token_count, logit_chunk):
        chunk = logits[start : start + logit_chunk].float()
        log_norm = torch.logsumexp(chunk, dim=-1, keepdim=True)
        log_probs = chunk - log_norm
        entropies.append(-(log_probs.exp() * log_probs).sum(dim=-1))

    entropy = torch.cat(entropies)
    rank = ranks.float()
    q = torch.quantile(
        lp,
        torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90], device=lp.device),
    )
    edge = min(20, token_count)
    return {
        "n_tok": float(token_count),
        "chars_per_tok": n_chars / max(token_count, 1),
        "lp_mean": lp.mean().item(),
        "lp_std": lp.std(unbiased=False).item(),
        "lp_min": lp.min().item(),
        "lp_p05": q[0].item(),
        "lp_p10": q[1].item(),
        "lp_p25": q[2].item(),
        "lp_p50": q[3].item(),
        "lp_p75": q[4].item(),
        "lp_p90": q[5].item(),
        "lp_first20": lp[:edge].mean().item(),
        "lp_last20": lp[-edge:].mean().item(),
        "ent_mean": entropy.mean().item(),
        "ent_std": entropy.std(unbiased=False).item(),
        "surprise_ratio": (-lp.mean() / entropy.mean().clamp_min(1e-6)).item(),
        "surprise_minus_ent": (-lp - entropy).mean().item(),
        "top1": (rank == 0).float().mean().item(),
        "top5": (rank < 5).float().mean().item(),
        "top10": (rank < 10).float().mean().item(),
        "top100": (rank < 100).float().mean().item(),
        "logrank_mean": torch.log1p(rank).mean().item(),
        "frac_lp_lt_-5": (lp < -5).float().mean().item(),
        "frac_lp_lt_-10": (lp < -10).float().mean().item(),
    }


def score_texts(
    ids: list[str],
    texts: list[str],
    config: LMStatConfig,
    device: str | None = None,
) -> pd.DataFrame:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if target_device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(target_device)
        dtype = (
            torch.bfloat16
            if major >= 8 and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if config.load_in_4bit:
        if target_device.type != "cuda":
            raise ValueError("4-bit loading requires CUDA in this project")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map={"": target_device.index or 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,
        ).to(target_device)
    model.eval()
    print(
        f"loaded {config.model_name} "
        f"({'4-bit' if config.load_in_4bit else str(dtype).replace('torch.', '')})",
        flush=True,
    )

    prefix_id = tokenizer.bos_token_id
    if prefix_id is None:
        prefix_id = tokenizer.eos_token_id
    if prefix_id is None:
        raise ValueError("Tokenizer has neither bos_token_id nor eos_token_id")

    rows: list[dict[str, float | str]] = []
    started = time.time()
    for index, (passage_id, text) in enumerate(zip(ids, texts), start=1):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids = token_ids[: config.max_length - 1]
        input_ids = torch.tensor(
            [[prefix_id, *token_ids]],
            dtype=torch.long,
            device=target_device,
        )
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, :-1]
            features = _passage_features(
                logits,
                input_ids[0, 1:],
                len(text),
                config.logit_chunk,
            )
        features["id"] = passage_id
        rows.append(features)
        if index % 50 == 0 or index == len(texts):
            elapsed = time.time() - started
            remaining = elapsed / index * (len(texts) - index)
            print(
                f"{index}/{len(texts)} scored; "
                f"{elapsed:.0f}s elapsed, {remaining:.0f}s remaining",
                flush=True,
            )
    return pd.DataFrame(rows)

