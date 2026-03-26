
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG
from .env import SEED, set_seed
from .records import add_conflict_labels


class BinaryProbe(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

def _standardize(train_x: np.ndarray, test_x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True) + 1e-06
    return ((train_x - mu) / sigma, (test_x - mu) / sigma, mu, sigma)

def train_binary_probe(train_x, train_y, test_x, test_y, hidden_dim=512, epochs=8, lr=0.002, batch_size=64, seed=SEED):
    set_seed(seed)
    train_x, test_x, mu, sigma = _standardize(train_x, test_x)
    Xtr = torch.tensor(train_x, dtype=torch.float32)
    Ytr = torch.tensor(train_y, dtype=torch.float32)
    Xte = torch.tensor(test_x, dtype=torch.float32)
    Yte = torch.tensor(test_y, dtype=torch.float32)
    model = BinaryProbe(Xtr.shape[1], hidden_dim=min(hidden_dim, max(64, Xtr.shape[1] // 2))).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for st in range(0, len(perm), batch_size):
            mb = perm[st:st + batch_size]
            logits = model(Xtr[mb])
            loss = F.binary_cross_entropy_with_logits(logits, Ytr[mb])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        train_prob = torch.sigmoid(model(Xtr)).numpy()
        test_prob = torch.sigmoid(model(Xte)).numpy()
    train_pred = (train_prob >= 0.5).astype(np.int64)
    test_pred = (test_prob >= 0.5).astype(np.int64)
    out = {'train_acc': float((train_pred == train_y).mean()), 'test_acc': float((test_pred == test_y).mean()), 'train_prob': train_prob, 'test_prob': test_prob, 'mu': mu, 'sigma': sigma, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()}, 'input_dim': Xtr.shape[1]}
    return out

def run_layerwise_probe_suite(train_records: List[Dict[str, Any]], test_records: List[Dict[str, Any]], rep_field: str='V_edit', pos_families=('true_override',), neg_families=('compatible_refinement', 'false_trigger'), slots: Optional[List[str]]=None, out_csv: Optional[Path]=None):
    train_records = add_conflict_labels(train_records, pos_families=pos_families, neg_families=neg_families)
    test_records = add_conflict_labels(test_records, pos_families=pos_families, neg_families=neg_families)
    if slots is None:
        slots = sorted(set((r['target_slot'] for r in train_records)))
    rows = []
    layers = sorted(set((r['layer'] for r in train_records)))
    for slot in slots + ['__all__']:
        for layer in layers:
            tr = [r for r in train_records if r['layer'] == layer and r['y_conflict'] is not None and (slot == '__all__' or r['target_slot'] == slot)]
            te = [r for r in test_records if r['layer'] == layer and r['y_conflict'] is not None and (slot == '__all__' or r['target_slot'] == slot)]
            if len(tr) < 8 or len(te) < 8:
                continue
            Xtr = np.stack([r[rep_field] for r in tr], axis=0).astype(np.float32)
            ytr = np.array([r['y_conflict'] for r in tr], dtype=np.int64)
            Xte = np.stack([r[rep_field] for r in te], axis=0).astype(np.float32)
            yte = np.array([r['y_conflict'] for r in te], dtype=np.int64)
            res = train_binary_probe(Xtr, ytr, Xte, yte, hidden_dim=CFG.probe_hidden_dim, epochs=CFG.probe_epochs, lr=CFG.probe_lr, batch_size=CFG.probe_batch_size)
            rows.append({'rep_field': rep_field, 'slot': slot, 'layer': layer, 'train_acc': res['train_acc'], 'test_acc': res['test_acc'], 'n_train': len(tr), 'n_test': len(te)})
    df = pd.DataFrame(rows)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print('Saved probe suite to:', out_csv)
    return df
