
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from .model_utils import ModelBundle, to_legacy_cache


def get_layer_k_tensor(past_key_values, layer_idx: int):
    pkv = to_legacy_cache(past_key_values)
    return pkv[layer_idx][0]

def get_layer_v_tensor(past_key_values, layer_idx: int):
    pkv = to_legacy_cache(past_key_values)
    return pkv[layer_idx][1]

def flatten_k_tokens(past_key_values, layer_idx: int, positions: List[int]) -> torch.Tensor:
    k = get_layer_k_tensor(past_key_values, layer_idx)[0]
    x = k[:, positions, :].permute(1, 0, 2).contiguous().view(len(positions), -1)
    return x

def flatten_v_tokens(past_key_values, layer_idx: int, positions: List[int]) -> torch.Tensor:
    v = get_layer_v_tensor(past_key_values, layer_idx)[0]
    x = v[:, positions, :].permute(1, 0, 2).contiguous().view(len(positions), -1)
    return x

def write_flat_k_tokens_(past_key_values, layer_idx: int, positions: List[int], new_flat_k: torch.Tensor, num_kv_heads: int, head_dim: int):
    k = get_layer_k_tensor(past_key_values, layer_idx)[0]
    reshaped = new_flat_k.view(len(positions), num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
    k[:, positions, :] = reshaped

def write_flat_v_tokens_(past_key_values, layer_idx: int, positions: List[int], new_flat_v: torch.Tensor, num_kv_heads: int, head_dim: int):
    v = get_layer_v_tensor(past_key_values, layer_idx)[0]
    reshaped = new_flat_v.view(len(positions), num_kv_heads, head_dim).permute(1, 0, 2).contiguous()
    v[:, positions, :] = reshaped

def extract_mean_rep(prepared: Dict[str, Any], bundle: ModelBundle, layer_idx: int, positions: List[int]) -> Dict[str, np.ndarray]:
    if len(positions) == 0:
        return {'K': np.zeros(bundle.kv_dim, dtype=np.float32), 'V': np.zeros(bundle.kv_dim, dtype=np.float32), 'KV': np.zeros(bundle.kv_dim * 2, dtype=np.float32)}
    K = flatten_k_tokens(prepared['past_after_new'], layer_idx, positions).float().mean(dim=0).cpu().numpy()
    V = flatten_v_tokens(prepared['past_after_new'], layer_idx, positions).float().mean(dim=0).cpu().numpy()
    KV = np.concatenate([K, V], axis=0)
    return {'K': K, 'V': V, 'KV': KV}

def extract_headwise_mean_rep(prepared: Dict[str, Any], bundle: ModelBundle, layer_idx: int, positions: List[int], which: str='V') -> np.ndarray:
    if len(positions) == 0:
        return np.zeros((bundle.num_kv_heads, bundle.head_dim), dtype=np.float32)
    if which == 'K':
        t = get_layer_k_tensor(prepared['past_after_new'], layer_idx)[0][:, positions, :].float().mean(dim=1)
    else:
        t = get_layer_v_tensor(prepared['past_after_new'], layer_idx)[0][:, positions, :].float().mean(dim=1)
    return t.detach().cpu().numpy()

def compute_prefix_saliency(prepared: Dict[str, Any], bundle: ModelBundle):
    prefix_len = prepared['prefix_len']
    sal = {}
    attentions = prepared['out_new_attentions']
    for l, attn in enumerate(attentions):
        a = attn[0, :, :, :prefix_len].float()
        recv = a.sum(dim=1).sum(dim=0)
        if recv.sum() > 0:
            recv = recv / (recv.sum() + 1e-08)
        sal[l] = recv
    return sal
