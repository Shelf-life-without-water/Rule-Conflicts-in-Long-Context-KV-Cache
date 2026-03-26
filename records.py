
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .kv_utils import extract_mean_rep
from .model_utils import ModelBundle
from .span_utils import prepare_segmented_sample


def collect_representation_records(bundle: ModelBundle, samples: List[Dict[str, Any]], layers: Optional[List[int]]=None, split_name: str='probe_train', max_context_tokens: Optional[int]=None, save_path: Optional[Path]=None):
    if layers is None:
        layers = list(range(bundle.num_layers))
    records = []
    for sample in tqdm(samples, desc=f'collect reps [{split_name}]'):
        prepared = prepare_segmented_sample(bundle, sample, max_context_tokens=max_context_tokens or bundle.spec.max_context_tokens)
        editable_positions = prepared['editable_old_positions']
        old_positions = prepared['old_instruction_positions']
        for layer_idx in layers:
            reps_edit = extract_mean_rep(prepared, bundle, layer_idx, editable_positions)
            reps_old = extract_mean_rep(prepared, bundle, layer_idx, old_positions)
            records.append({'split': split_name, 'sample_id': sample['sample_id'], 'family': sample['family'], 'target_slot': sample['target_slot'], 'event_type': sample['event_type'], 'layer': layer_idx, 'n_editable_old_positions': len(editable_positions), 'K_edit': reps_edit['K'], 'V_edit': reps_edit['V'], 'KV_edit': reps_edit['KV'], 'K_old': reps_old['K'], 'V_old': reps_old['V'], 'KV_old': reps_old['KV']})
        del prepared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(records, save_path)
        print('Saved representation records to:', save_path)
    return records

def records_to_dataframe(records: List[Dict[str, Any]], rep_key: str='V_edit') -> pd.DataFrame:
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        row['rep_key'] = rep_key
        row['dim'] = len(r[rep_key])
        rows.append(row)
    return pd.DataFrame(rows)

def add_conflict_labels(records: List[Dict[str, Any]], pos_families=('true_override',), neg_families=('compatible_refinement', 'false_trigger')):
    out = []
    for r in records:
        if r['family'] in pos_families:
            y = 1
        elif r['family'] in neg_families:
            y = 0
        else:
            y = None
        rr = dict(r)
        rr['y_conflict'] = y
        out.append(rr)
    return out
