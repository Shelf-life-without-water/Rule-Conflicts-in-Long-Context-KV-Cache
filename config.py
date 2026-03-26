from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from .env import MODEL_PATHS

@dataclass
class ModelSpec:
    name: str
    path: Path
    trust_remote_code: bool = True
    max_context_tokens: int = 8192
    preferred_dtype: str = "bfloat16"

@dataclass
class InterpConfig:
    n_probe_train_per_slot_family: int = 64
    n_probe_test_per_slot_family: int = 64
    n_erase_test_per_slot_family: int = 64
    distractor_min_chars: int = 600
    distractor_char_choices: Tuple[int, ...] = (800, 1600, 3200)
    old_rule_repeat_choices: Tuple[int, ...] = (1, 2, 4)
    edit_local_window: int = 1
    pin_sink_tokens: int = 8
    probe_hidden_dim: int = 512
    probe_epochs: int = 8
    probe_lr: float = 2e-3
    probe_batch_size: int = 64
    rank_grid: Tuple[int, ...] = (1, 4, 8)
    alpha_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    max_new_token_slack: int = 8
    bootstrap_B: int = 1000
    active_models: Tuple[str, ...] = ("llama_8b",)
    active_slots: Tuple[str, ...] = ("format", "style", "persona", "language")
    active_families: Tuple[str, ...] = ("true_override", "compatible_refinement", "false_trigger", "irrelevant_control")

CFG = InterpConfig()

MODEL_SPECS = {
    "llama_8b": ModelSpec("llama_8b", MODEL_PATHS["llama_8b"], max_context_tokens=8192),
    "qwen_14b": ModelSpec("qwen_14b", MODEL_PATHS["qwen_14b"], max_context_tokens=8192),
    "qwen_14b_1m": ModelSpec("qwen_14b_1m", MODEL_PATHS["qwen_14b_1m"], max_context_tokens=32768),
}

ACTIVE_MODELS = list(CFG.active_models)
