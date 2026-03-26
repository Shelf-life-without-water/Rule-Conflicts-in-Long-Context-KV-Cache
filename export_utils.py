from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

def torch_load_any(path, map_location='cpu'):
    return torch.load(path, map_location=map_location, weights_only=False)

def _safe_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _df_head_records(df, n=10):
    if df is None:
        return []
    if len(df) == 0:
        return []
    out = []
    for _, row in df.head(n).iterrows():
        rec = {}
        for k, v in row.to_dict().items():
            if isinstance(v, (float, np.floating)):
                rec[k] = _safe_float(v)
            elif isinstance(v, (int, np.integer)):
                rec[k] = _safe_int(v)
            else:
                rec[k] = v
        out.append(rec)
    return out

def _read_csv_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

def _records_from_csv(path: Path, n=None):
    df = _read_csv_if_exists(path)
    if len(df) == 0:
        return []
    if n is not None:
        df = df.head(n)
    out = []
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.to_dict().items():
            if isinstance(v, (float, np.floating)):
                rec[k] = _safe_float(v)
            elif isinstance(v, (int, np.integer)):
                rec[k] = _safe_int(v)
            else:
                rec[k] = v
        out.append(rec)
    return out

def _best_probe_rows(probe_df: pd.DataFrame, top_k=5):
    if probe_df is None or len(probe_df) == 0:
        return []
    cand = probe_df.copy()
    score_col = None
    for c in ['test_acc', 'test_auc', 'acc', 'auc', 'score']:
        if c in cand.columns:
            score_col = c
            break
    if score_col is None:
        return _df_head_records(cand, n=top_k)
    cand = cand.sort_values(score_col, ascending=False).head(top_k)
    return _df_head_records(cand, n=top_k)

def _group_mean_table(df: pd.DataFrame, group_cols, metric_cols):
    if df is None or len(df) == 0:
        return []
    keep = [c for c in metric_cols if c in df.columns]
    if len(keep) == 0:
        return []
    grouped = df.groupby(group_cols, dropna=False)[keep].mean().reset_index()
    return _df_head_records(grouped, n=len(grouped))

def export_experiment_summary_json(results_root: Path, artifact_root: Path, runs_root: Path, cfg, active_models, interp_splits, out_path: Path=None):
    if out_path is None:
        out_path = results_root / 'experiment_summary.json'
    summary = {'meta': {'created_at': datetime.now().isoformat(), 'results_root': str(results_root), 'artifact_root': str(artifact_root), 'runs_root': str(runs_root), 'active_models': list(active_models), 'config': asdict(cfg) if hasattr(cfg, '__dataclass_fields__') else str(cfg)}, 'dataset_summary': {'probe_train_n': len(interp_splits.get('probe_train', [])), 'probe_test_n': len(interp_splits.get('probe_test', [])), 'erase_test_n': len(interp_splits.get('erase_test', []))}, 'artifacts': {}, 'results': {}}
    for split_name in ['probe_train', 'probe_test', 'erase_test']:
        samples = interp_splits.get(split_name, [])
        fam_counts = pd.Series([x['family'] for x in samples]).value_counts().to_dict() if len(samples) > 0 else {}
        slot_counts = pd.Series([x['target_slot'] for x in samples]).value_counts().to_dict() if len(samples) > 0 else {}
        summary['dataset_summary'][f'{split_name}_family_counts'] = fam_counts
        summary['dataset_summary'][f'{split_name}_slot_counts'] = slot_counts
    artifact_files = [artifact_root / 'probe_train_records.pt', artifact_root / 'probe_test_records.pt', runs_root / 'probe_suite_V.csv', runs_root / 'probe_suite_K.csv', runs_root / 'probe_suite_KV.csv', runs_root / 'probe_suite_all.csv', runs_root / 'slot_global_similarity.csv', runs_root / 'temporal_reps_full.csv', runs_root / 'causal_erase_results.csv', runs_root / 'causal_erase_results_full.csv', runs_root / 'causal_erase_summary_full.csv']
    for p in artifact_files:
        summary['artifacts'][p.name] = {'exists': p.exists(), 'path': str(p), 'size_bytes': p.stat().st_size if p.exists() else None}
    probe_all_path = runs_root / 'probe_suite_all.csv'
    probe_all_df = _read_csv_if_exists(probe_all_path)
    summary['results']['probe_suite'] = {'n_rows': int(len(probe_all_df)), 'best_rows': _best_probe_rows(probe_all_df, top_k=10), 'mean_by_rep_field': _group_mean_table(probe_all_df, group_cols=['rep_field'] if 'rep_field' in probe_all_df.columns else ['layer'], metric_cols=['test_acc', 'test_auc', 'acc', 'auc', 'score'])}
    slot_sim_path = runs_root / 'slot_global_similarity.csv'
    slot_sim_df = _read_csv_if_exists(slot_sim_path)
    summary['results']['slot_global_similarity'] = {'n_rows': int(len(slot_sim_df)), 'head_rows': _df_head_records(slot_sim_df, n=20), 'mean_by_slot': _group_mean_table(slot_sim_df, group_cols=['slot'] if 'slot' in slot_sim_df.columns else ['layer'], metric_cols=['sim_to_global'])}
    temporal_path = runs_root / 'temporal_reps_full.csv'
    temporal_df = _read_csv_if_exists(temporal_path)
    summary['results']['temporal'] = {'n_rows': int(len(temporal_df)), 'head_rows': _df_head_records(temporal_df, n=20)}
    erase_full_path = runs_root / 'causal_erase_results_full.csv'
    if not erase_full_path.exists():
        erase_full_path = runs_root / 'causal_erase_results.csv'
    erase_df = _read_csv_if_exists(erase_full_path)
    summary['results']['causal_erase'] = {'n_rows': int(len(erase_df)), 'head_rows': _df_head_records(erase_df, n=20), 'mean_by_target': _group_mean_table(erase_df, group_cols=['target'] if 'target' in erase_df.columns else ['method'], metric_cols=['payload_ok', 'ifc', 'current_slot_acc', 'stable_slot_acc', 'cr', 'cfr']), 'mean_by_target_rank_alpha': _group_mean_table(erase_df, group_cols=[c for c in ['target', 'rank', 'alpha', 'slot_specific'] if c in erase_df.columns], metric_cols=['payload_ok', 'ifc', 'current_slot_acc', 'stable_slot_acc', 'cr', 'cfr'])}
    optional_summary_csvs = [results_root / 'ALL_SUMMARY_OVERALL.csv', results_root / 'ALL_SUMMARY_BY_FAMILY.csv', results_root / 'ALL_DELTA_VS_NONE.csv']
    summary['results']['existing_summary_csvs'] = {}
    for p in optional_summary_csvs:
        summary['results']['existing_summary_csvs'][p.name] = {'exists': p.exists(), 'path': str(p), 'head_rows': _records_from_csv(p, n=20) if p.exists() else []}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('Saved experiment summary JSON to:', out_path)
    return (summary, out_path)

def make_results_archive(results_root: Path, archive_base: Path | None = None) -> Path:
    base = archive_base if archive_base is not None else results_root.parent / f"{results_root.name}_full_export"
    archive_path = shutil.make_archive(base_name=str(base), format="zip", root_dir=str(results_root))
    return Path(archive_path)
