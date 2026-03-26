
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .interventions import apply_kv_projection_to_positions, run_query_and_generate
from .metrics import compute_sample_metrics, evaluate_none
from .model_utils import ModelBundle, clone_legacy_cache
from .span_utils import prepare_segmented_sample
from .subspace import fit_conflict_subspace_from_records


def run_subspace_erase_on_sample(bundle: ModelBundle, sample: Dict[str, Any], subspace_map: Dict[Tuple[int, Optional[str], str, int], np.ndarray], target: str='V', slot_specific: bool=True, rank: int=1, alpha: float=1.0, layers: Optional[List[int]]=None):
    prepared = prepare_segmented_sample(bundle, sample, max_context_tokens=bundle.spec.max_context_tokens)
    editable_positions = prepared['editable_old_positions']
    keep = torch.ones(prepared['prefix_len'] + prepared['new_len'], dtype=torch.long, device=bundle.device)
    if layers is None:
        layers = list(range(max(0, bundle.num_layers - 8), bundle.num_layers))
    past = clone_legacy_cache(prepared['past_after_new'])
    if len(editable_positions) > 0:
        for layer in layers:
            key = (layer, sample['target_slot'] if slot_specific else None, target, rank)
            if key not in subspace_map:
                continue
            U = subspace_map[key]
            apply_kv_projection_to_positions(past_key_values=past, bundle=bundle, layer_idx=layer, positions=editable_positions, U=U, alpha=alpha, target=target)
    gen = run_query_and_generate(bundle=bundle, past_prequery=past, prequery_keep_mask=keep, query_ids=prepared['query_ids'], gold_answer_text=sample['gold_current'], payload_row=sample['payload'])
    met = compute_sample_metrics(sample, gen['output_text'])
    return {**gen, **met, 'n_editable_old_positions': len(editable_positions)}

def fit_subspace_bank(records: List[Dict[str, Any]], layers: List[int], targets: List[str]=['V', 'K', 'KV'], ranks: List[int]=[1, 4], slot_specific: bool=True, pos_families=('true_override',), neg_families=('compatible_refinement', 'false_trigger')):
    bank = {}
    slots = sorted(set((r['target_slot'] for r in records)))
    for layer in layers:
        slot_list = slots if slot_specific else [None]
        for slot in slot_list:
            for target in targets:
                rep_field = {'K': 'K_edit', 'V': 'V_edit', 'KV': 'KV_edit'}[target]
                for rank in ranks:
                    U = fit_conflict_subspace_from_records(records, layer=layer, rep_field=rep_field, slot=slot, pos_families=pos_families, neg_families=neg_families, rank=rank)
                    if U is not None:
                        bank[layer, slot, target, rank] = U
    return bank

def run_causal_erase_suite(bundle: ModelBundle, probe_records: List[Dict[str, Any]], erase_samples: List[Dict[str, Any]], layers: Optional[List[int]]=None, targets=('V', 'K', 'KV'), ranks=(1, 4), alphas=(0.5, 1.0), slot_specific_options=(True, False), max_samples: Optional[int]=None, out_csv: Optional[Path]=None):
    if layers is None:
        layers = list(range(max(0, bundle.num_layers - 8), bundle.num_layers))
    rows = []
    erase_samples = erase_samples if max_samples is None else erase_samples[:max_samples]
    for sample in tqdm(erase_samples, desc='baseline none'):
        baseline = evaluate_none(bundle, sample)
        rows.append({'method': 'none', 'target': 'none', 'rank': 0, 'alpha': 0.0, 'slot_specific': False, 'sample_id': sample['sample_id'], 'family': sample['family'], 'target_slot': sample['target_slot'], **baseline})
    for slot_specific in slot_specific_options:
        bank = fit_subspace_bank(probe_records, layers=layers, targets=list(targets), ranks=list(ranks), slot_specific=slot_specific)
        for target in targets:
            for rank in ranks:
                for alpha in alphas:
                    desc = f'erase {target} rank={rank} alpha={alpha} slot_specific={slot_specific}'
                    for sample in tqdm(erase_samples, desc=desc):
                        res = run_subspace_erase_on_sample(bundle=bundle, sample=sample, subspace_map=bank, target=target, slot_specific=slot_specific, rank=rank, alpha=alpha, layers=layers)
                        rows.append({'method': 'erase', 'target': target, 'rank': rank, 'alpha': alpha, 'slot_specific': slot_specific, 'sample_id': sample['sample_id'], 'family': sample['family'], 'target_slot': sample['target_slot'], **res})
    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print('Saved erase suite to:', out_csv)
    return df
