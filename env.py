from __future__ import annotations

import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 3407

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FS_ROOT = Path(os.getenv("KV_CONFLICT_FS_ROOT", PACKAGE_ROOT))
DEFAULT_TMP_ROOT = Path(os.getenv("KV_CONFLICT_TMP_ROOT", PACKAGE_ROOT / ".local"))

MODEL_PATHS = {
    "llama_8b": Path(os.getenv("KV_CONFLICT_MODEL_LLAMA_8B", DEFAULT_FS_ROOT / "models" / "Meta-Llama-3.1-8B-Instruct")),
    "qwen_14b": Path(os.getenv("KV_CONFLICT_MODEL_QWEN_14B", DEFAULT_FS_ROOT / "models" / "Qwen2.5-14B-Instruct")),
    "qwen_14b_1m": Path(os.getenv("KV_CONFLICT_MODEL_QWEN_14B_1M", DEFAULT_TMP_ROOT / ".cache" / "modelscope" / "Qwen" / "Qwen2___5-14B-Instruct-1M")),
}

DATASET_ROOT = Path(os.getenv("KV_CONFLICT_DATASET_ROOT", DEFAULT_TMP_ROOT / "datasets"))
RUN_NAME = os.getenv("KV_CONFLICT_RUN_NAME", "kv_conflict_interp_neurips_blueprint_full_v2")
RESULTS_ROOT = Path(os.getenv("KV_CONFLICT_RESULTS_ROOT", DEFAULT_TMP_ROOT / "results" / RUN_NAME))
ARTIFACT_ROOT = RESULTS_ROOT / "artifacts"
FIG_ROOT = RESULTS_ROOT / "figures"
RUNS_ROOT = RESULTS_ROOT / "runs"

def ensure_run_dirs() -> None:
    for path in (RESULTS_ROOT, ARTIFACT_ROOT, FIG_ROOT, RUNS_ROOT):
        path.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

ensure_run_dirs()
set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
