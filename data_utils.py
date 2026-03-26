from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from datasets import Dataset, DatasetDict, load_from_disk

from .config import CFG
from .env import DATASET_ROOT

def _find_dir_case_insensitive(root: Path, candidates: List[str]) -> Optional[Path]:
    if not root.exists():
        return None
    cand_lower = {x.lower() for x in candidates}
    for p in root.iterdir():
        if p.name.lower() in cand_lower:
            return p
    return None
DATASET_DIRS = {'raw_longbench': _find_dir_case_insensitive(DATASET_ROOT, ['_raw_longbench', 'raw_longbench', 'rawlongbench']), 'longbench': _find_dir_case_insensitive(DATASET_ROOT, ['LongBench', 'longbench']), 'longbench_v2': _find_dir_case_insensitive(DATASET_ROOT, ['LongBench-v2', 'longbench-v2', 'longbench_v2', 'longben_v2']), 'ifeval': _find_dir_case_insensitive(DATASET_ROOT, ['IFEval', 'ifeval']), 'multi_turn': _find_dir_case_insensitive(DATASET_ROOT, ['Multi-Turn-Instruct', 'multi-turn-instruct', 'multi_turn_instruct'])}
LIKELY_TEXT_KEYS = {'text', 'context', 'input', 'article', 'document', 'documents', 'prompt', 'story', 'passage', 'content', 'conversation', 'instruction', 'query', 'question'}

def safe_load_from_disk(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    try:
        return load_from_disk(str(path))
    except Exception as e:
        print(f'[WARN] load_from_disk failed for {path}: {e}')
        return None

def iter_dataset_records(ds_obj, max_records: Optional[int]=None):
    n = 0
    if ds_obj is None:
        return
    if isinstance(ds_obj, DatasetDict):
        for split_name in ds_obj.keys():
            split_ds = ds_obj[split_name]
            for ex in split_ds:
                yield ex
                n += 1
                if max_records is not None and n >= max_records:
                    return
    elif isinstance(ds_obj, Dataset):
        for ex in ds_obj:
            yield ex
            n += 1
            if max_records is not None and n >= max_records:
                return

def extract_long_strings(obj, min_chars: int=400):
    out = []
    if isinstance(obj, str):
        if len(obj) >= min_chars:
            out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if k.lower() in LIKELY_TEXT_KEYS or len(v) >= min_chars:
                    out.append(v)
            else:
                out.extend(extract_long_strings(v, min_chars=min_chars))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(extract_long_strings(x, min_chars=min_chars))
    return out

def collect_texts_from_saved_dataset(path: Optional[Path], max_records: int=3000, min_chars: int=400):
    ds_obj = safe_load_from_disk(path)
    if ds_obj is None:
        return []
    texts = []
    for ex in iter_dataset_records(ds_obj, max_records=max_records):
        texts.extend(extract_long_strings(ex, min_chars=min_chars))
    return texts

def collect_texts_from_raw_folder(path: Optional[Path], max_files: int=300, min_chars: int=400):
    if path is None or not path.exists():
        return []
    texts = []
    files = list(path.rglob('*'))
    files = [p for p in files if p.is_file() and p.suffix.lower() in {'.json', '.jsonl', '.txt', '.md'}][:max_files]
    for fp in files:
        try:
            if fp.suffix.lower() == '.jsonl':
                with open(fp, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            texts.extend(extract_long_strings(obj, min_chars=min_chars))
                        except Exception:
                            if len(line) >= min_chars:
                                texts.append(line)
            elif fp.suffix.lower() == '.json':
                with open(fp, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                texts.extend(extract_long_strings(obj, min_chars=min_chars))
            else:
                txt = fp.read_text(encoding='utf-8', errors='ignore')
                if len(txt) >= min_chars:
                    texts.append(txt)
        except Exception:
            continue
    return texts

def dedup_texts(texts: List[str], max_items: int=10000):
    seen = set()
    out = []
    for t in texts:
        key = hashlib.md5(t[:2000].encode('utf-8', errors='ignore')).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_items:
            break
    return out

def build_default_distractor_pool(min_chars: int | None = None, max_items: int = 8000) -> list[str]:
    effective_min_chars = CFG.distractor_min_chars if min_chars is None else min_chars
    distractor_texts = []
    distractor_texts += collect_texts_from_raw_folder(DATASET_DIRS["raw_longbench"], max_files=400, min_chars=effective_min_chars)
    distractor_texts += collect_texts_from_saved_dataset(DATASET_DIRS["longbench_v2"], max_records=1500, min_chars=effective_min_chars)
    distractor_texts += collect_texts_from_saved_dataset(DATASET_DIRS["multi_turn"], max_records=1500, min_chars=effective_min_chars)
    distractor_texts += collect_texts_from_saved_dataset(DATASET_DIRS["ifeval"], max_records=800, min_chars=effective_min_chars)
    return dedup_texts(distractor_texts, max_items=max_items)
