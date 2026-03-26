
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .kv_utils import extract_headwise_mean_rep, flatten_k_tokens, flatten_v_tokens
from .model_utils import ModelBundle, encode_text, to_model_cache
from .span_utils import prepare_segmented_sample, truncate_text_by_tokens


def collect_temporal_reps(bundle: ModelBundle, sample: Dict[str, Any], layers: Optional[List[int]]=None):
    if layers is None:
        layers = list(range(bundle.num_layers))
    tokenizer = bundle.tokenizer
    device = bundle.device
    seg = sample['segments']
    task_ids = encode_text(tokenizer, seg['task_text'], add_bos=tokenizer.bos_token_id is not None, device=device)
    payload_ids = encode_text(tokenizer, seg['payload_text'], add_bos=False, device=device)
    old_ids = encode_text(tokenizer, seg['old_text'], add_bos=False, device=device)
    new_ids = encode_text(tokenizer, seg['new_text'], add_bos=False, device=device)
    stages = {}
    with torch.no_grad():
        ids_old = torch.cat([task_ids, payload_ids, old_ids], dim=1)
        out_old = bundle.model(input_ids=ids_old, use_cache=True, return_dict=True)
        stages['old_only'] = to_model_cache(out_old.past_key_values)
        distractor_short = truncate_text_by_tokens(tokenizer, seg['distractor_text'], 256)
        dist_ids = encode_text(tokenizer, distractor_short, add_bos=False, device=device)
        ids_mid = torch.cat([task_ids, payload_ids, old_ids, dist_ids], dim=1)
        out_mid = bundle.model(input_ids=ids_mid, use_cache=True, return_dict=True)
        stages['old_plus_short_distractor'] = to_model_cache(out_mid.past_key_values)
        out_full = bundle.model(input_ids=torch.cat([task_ids, payload_ids, old_ids, dist_ids], dim=1), use_cache=True, return_dict=True)
        past_full = to_model_cache(out_full.past_key_values)
        mask_new = torch.ones((1, ids_mid.shape[1] + new_ids.shape[1]), dtype=torch.long, device=device)
        out_new = bundle.model(input_ids=new_ids, past_key_values=past_full, attention_mask=mask_new, use_cache=True, return_dict=True)
        stages['after_new'] = to_model_cache(out_new.past_key_values)
    rows = []
    prepared = prepare_segmented_sample(bundle, sample, max_context_tokens=bundle.spec.max_context_tokens)
    positions = prepared['editable_old_positions']
    for stage_name, cache in stages.items():
        for layer in layers:
            if len(positions) == 0:
                continue
            K = flatten_k_tokens(cache, layer, positions).float().mean(dim=0).cpu().numpy()
            V = flatten_v_tokens(cache, layer, positions).float().mean(dim=0).cpu().numpy()
            rows.append({'sample_id': sample['sample_id'], 'family': sample['family'], 'target_slot': sample['target_slot'], 'stage': stage_name, 'layer': layer, 'K_norm': float(np.linalg.norm(K)), 'V_norm': float(np.linalg.norm(V))})
    return pd.DataFrame(rows)

def run_head_scan(prepared: Dict[str, Any], bundle: ModelBundle, which: str='V'):
    rows = []
    positions = prepared['editable_old_positions']
    if len(positions) == 0:
        return pd.DataFrame(rows)
    for layer in range(bundle.num_layers):
        headwise = extract_headwise_mean_rep(prepared, bundle, layer, positions, which=which)
        for head in range(headwise.shape[0]):
            rows.append({'layer': layer, 'head': head, f'{which}_norm': float(np.linalg.norm(headwise[head]))})
    return pd.DataFrame(rows)
