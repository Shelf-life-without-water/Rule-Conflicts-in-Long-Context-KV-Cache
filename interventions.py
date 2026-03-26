
from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .config import CFG
from .kv_utils import flatten_k_tokens, flatten_v_tokens, get_layer_k_tensor, get_layer_v_tensor, write_flat_k_tokens_, write_flat_v_tokens_
from .model_utils import ModelBundle, encode_text, to_model_cache


def project_erase_flat(flat_x: torch.Tensor, U: torch.Tensor, alpha: float=1.0) -> torch.Tensor:
    proj = flat_x @ U @ U.T
    return flat_x - alpha * proj

def apply_kv_projection_to_positions(past_key_values, bundle: ModelBundle, layer_idx: int, positions: List[int], U: np.ndarray, alpha: float=1.0, target: str='V'):
    if len(positions) == 0:
        return
    kv_dim = bundle.kv_dim
    U_t = torch.tensor(U, dtype=torch.float32, device=bundle.device)
    if target == 'K':
        K = flatten_k_tokens(past_key_values, layer_idx, positions).float()
        assert K.shape[1] == kv_dim, f'K dim mismatch: got {K.shape[1]}, expected {kv_dim}'
        assert U_t.shape[0] == kv_dim, f'U dim mismatch for K erase: got {U_t.shape[0]}, expected {kv_dim}'
        K_new = project_erase_flat(K, U_t, alpha=alpha).to(get_layer_k_tensor(past_key_values, layer_idx).dtype)
        write_flat_k_tokens_(past_key_values, layer_idx, positions, K_new, bundle.num_kv_heads, bundle.head_dim)
    elif target == 'V':
        V = flatten_v_tokens(past_key_values, layer_idx, positions).float()
        assert V.shape[1] == kv_dim, f'V dim mismatch: got {V.shape[1]}, expected {kv_dim}'
        assert U_t.shape[0] == kv_dim, f'U dim mismatch for V erase: got {U_t.shape[0]}, expected {kv_dim}'
        V_new = project_erase_flat(V, U_t, alpha=alpha).to(get_layer_v_tensor(past_key_values, layer_idx).dtype)
        write_flat_v_tokens_(past_key_values, layer_idx, positions, V_new, bundle.num_kv_heads, bundle.head_dim)
    elif target == 'KV':
        assert U_t.shape[0] == 2 * kv_dim, f'U dim mismatch for KV erase: got {U_t.shape[0]}, expected {2 * kv_dim}'
        U_K = U_t[:kv_dim, :]
        U_V = U_t[kv_dim:, :]
        K = flatten_k_tokens(past_key_values, layer_idx, positions).float()
        V = flatten_v_tokens(past_key_values, layer_idx, positions).float()
        K_new = project_erase_flat(K, U_K, alpha=alpha).to(get_layer_k_tensor(past_key_values, layer_idx).dtype)
        V_new = project_erase_flat(V, U_V, alpha=alpha).to(get_layer_v_tensor(past_key_values, layer_idx).dtype)
        write_flat_k_tokens_(past_key_values, layer_idx, positions, K_new, bundle.num_kv_heads, bundle.head_dim)
        write_flat_v_tokens_(past_key_values, layer_idx, positions, V_new, bundle.num_kv_heads, bundle.head_dim)
    else:
        raise ValueError(f'Unsupported target: {target}')

def build_query_attention_mask(prequery_keep_mask: torch.Tensor, query_len: int):
    q = torch.ones(query_len, dtype=torch.long, device=prequery_keep_mask.device)
    return torch.cat([prequery_keep_mask, q], dim=0).unsqueeze(0)

def postprocess_output_text(output_text: str, payload_row: Dict[str, str]) -> str:
    t = output_text.strip()
    t = t.splitlines()[0].strip() if len(t.splitlines()) > 0 else t
    if payload_row['en'] in output_text:
        return payload_row['en']
    if payload_row['zh'] in output_text:
        return payload_row['zh']
    return t

def run_query_and_generate(bundle: ModelBundle, past_prequery, prequery_keep_mask: torch.Tensor, query_ids: torch.Tensor, gold_answer_text: str, payload_row: Dict[str, str]):
    tok = bundle.tokenizer
    model = bundle.model
    device = bundle.device
    gold_ids = encode_text(tok, gold_answer_text, add_bos=False, device=device)
    max_new_tokens = gold_ids.shape[1] + CFG.max_new_token_slack
    with torch.no_grad():
        t0 = time.perf_counter()
        attn_mask_query = build_query_attention_mask(prequery_keep_mask, query_ids.shape[1])
        out_query = model(input_ids=query_ids, past_key_values=to_model_cache(past_prequery), attention_mask=attn_mask_query, use_cache=True, output_hidden_states=False, output_attentions=False, return_dict=True)
        logits = out_query.logits[:, -1, :]
        first_token = torch.argmax(logits, dim=-1, keepdim=True)
        ttft = time.perf_counter() - t0
        generated = [int(first_token.item())]
        past = to_model_cache(out_query.past_key_values)
        running_mask = torch.cat([prequery_keep_mask, torch.ones(query_ids.shape[1], dtype=torch.long, device=device)], dim=0)
        t_gen0 = time.perf_counter()
        next_input = first_token
        for _ in range(max_new_tokens - 1):
            running_mask = torch.cat([running_mask, torch.ones(1, dtype=torch.long, device=device)], dim=0)
            out_step = model(input_ids=next_input, past_key_values=past, attention_mask=running_mask.unsqueeze(0), use_cache=True, output_hidden_states=False, output_attentions=False, return_dict=True)
            next_token = torch.argmax(out_step.logits[:, -1, :], dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            past = to_model_cache(out_step.past_key_values)
            next_input = next_token
            if tok.eos_token_id is not None and token_id == tok.eos_token_id:
                break
        total_gen_time = max(time.perf_counter() - t_gen0, 1e-08)
    raw_text = tok.decode(generated, skip_special_tokens=True).strip()
    out_text = postprocess_output_text(raw_text, payload_row)
    tok_per_sec = len(generated) / total_gen_time if total_gen_time > 0 else 0.0
    with torch.no_grad():
        loss0 = F.cross_entropy(logits, gold_ids[:, 0], reduction='none')
        if gold_ids.shape[1] > 1:
            attn_mask_gold = torch.cat([prequery_keep_mask, torch.ones(query_ids.shape[1] + gold_ids.shape[1] - 1, dtype=torch.long, device=device)], dim=0).unsqueeze(0)
            out_gold = model(input_ids=gold_ids[:, :-1], past_key_values=to_model_cache(out_query.past_key_values), attention_mask=attn_mask_gold, use_cache=False, return_dict=True)
            logits_rest = out_gold.logits.reshape(-1, out_gold.logits.shape[-1])
            tgt_rest = gold_ids[:, 1:].reshape(-1)
            loss_rest = F.cross_entropy(logits_rest, tgt_rest, reduction='none').view(1, -1).sum(dim=-1)
            nll = (loss0 + loss_rest) / gold_ids.shape[1]
        else:
            nll = loss0
    ppl = float(torch.exp(nll).item())
    return {'raw_text': raw_text, 'output_text': out_text, 'n_generated_tokens': len(generated), 'ttft_sec': float(ttft), 'tok_per_sec': float(tok_per_sec), 'ppl_gold': float(ppl)}
