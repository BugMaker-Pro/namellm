from __future__ import annotations

import re

import numpy as np
import pandas as pd


def style_feats(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z’']+", text)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    nw = max(len(words), 1)
    ns = max(len(sents), 1)
    nc = max(len(text), 1)
    sent_lens = [len(s.split()) for s in sents] or [0]

    def per_100_words(pattern: str) -> float:
        return 100 * len(re.findall(pattern, text)) / nw

    return {
        "chars": float(nc),
        "words": float(nw),
        "sents": float(ns),
        "sent_len_mean": float(np.mean(sent_lens)),
        "sent_len_std": float(np.std(sent_lens)),
        "sent_len_max": float(max(sent_lens)),
        "sent_len_min": float(min(sent_lens)),
        "word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "long_words": sum(len(w) >= 9 for w in words) / nw,
        "ttr": len({w.lower() for w in words}) / nw,
        "emdash": per_100_words("—"),
        "emdash_spaced": per_100_words(" — "),
        "endash": per_100_words("–"),
        "hyphen_spaced": per_100_words(" - "),
        "dbl_hyphen": per_100_words("--"),
        "curly_apos": per_100_words("’"),
        "straight_apos": per_100_words("'"),
        "curly_dq": per_100_words("[“”]"),
        "straight_dq": per_100_words('"'),
        "semicolon": per_100_words(";"),
        "colon": per_100_words(":"),
        "comma": per_100_words(","),
        "ellipsis": per_100_words(r"…|\.\.\."),
        "excl": per_100_words("!"),
        "quest": per_100_words(r"\?"),
        "paren": per_100_words(r"\("),
        "nonascii": 100 * sum(ord(c) > 127 for c in text) / nc,
        "digits": 100 * sum(c.isdigit() for c in text) / nc,
        "caps_words": sum(w[0].isupper() for w in words) / nw,
        "first_person": per_100_words(r"\b(?:I|me|my|we|our|us)\b"),
        "second_person": per_100_words(r"\b(?:[Yy]ou|[Yy]our)\b"),
        "sent_and_but": sum(s.startswith(("And ", "But ", "Yet ", "So ")) for s in sents) / ns,
        "sent_the": sum(s.startswith("The ") for s in sents) / ns,
        "ends_period": float(text.rstrip().endswith(".")),
        "its_not": per_100_words(r"\b(?:not|n’t|n't)\b"),
        "like_as": per_100_words(r"\b(?:like|as if|as though)\b"),
    }


def style_frame(texts: list[str] | pd.Series) -> pd.DataFrame:
    return pd.DataFrame([style_feats(text) for text in texts])

