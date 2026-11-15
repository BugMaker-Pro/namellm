from __future__ import annotations

from pathlib import Path

LABELS = sorted(
    [
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-6-astra",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "qwen3.8-max",
        "glm-5.3",
        "glm-5.3-flash",
        "kimi-k3",
    ]
)

GROUPS = ["ClaudeGLM", "GPTKimiQwen", "Gemini"]

GROUP_BY_FAMILY = {
    "claude": "ClaudeGLM",
    "glm": "ClaudeGLM",
    "gemini": "Gemini",
    "gpt": "GPTKimiQwen",
    "kimi": "GPTKimiQwen",
    "qwen": "GPTKimiQwen",
}


def family(model: str) -> str:
    return model.split("-")[0].replace("qwen3.8", "qwen")


def group(model: str) -> str:
    return GROUP_BY_FAMILY[family(model)]


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "raw" / "train.csv"
TEST_CSV = DATA_DIR / "raw" / "test.csv"
FEATURES_DIR = ROOT / "features"
OOF_DIR = ROOT / "oof"
SUBMISSIONS_DIR = ROOT / "submissions"

