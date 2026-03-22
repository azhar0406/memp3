"""Benchmark configuration — OpenRouter models and settings."""

import os

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

ANSWER_MODELS = [
    "google/gemini-2.0-flash-001",
]

JUDGE_MODEL = "google/gemini-2.0-flash-001"

SEARCH_TOP_K = 10
MAX_MEMORY_CHARS = 2000  # truncate each memory around relevant window

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

LONGMEMEVAL_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
