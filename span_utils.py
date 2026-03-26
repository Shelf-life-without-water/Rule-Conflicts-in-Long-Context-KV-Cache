
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import CFG
from .model_utils import ModelBundle, encode_text, to_model_cache


def tokenize_with_offsets(tokenizer, text: str):
    try:
        return tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors=None)
    except Exception:
        enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
        enc['offset_mapping'] = None
        return enc

def find_substring_char_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    start = text.find(substring)
    if start < 0:
        return None
    return (start, start + len(substring))

def char_span_to_token_positions(offsets, char_start: int, char_end: int):
    if offsets is None:
        return []
    hits = []
    for i, (s, e) in enumerate(offsets):
        if e <= char_start:
            continue
        if s >= char_end:
            break
        if max(s, char_start) < min(e, char_end):
            hits.append(i)
    return hits

def expand_positions(positions: List[int], max_len: int, window: int) -> List[int]:
    out = set()
    for p in positions:
        for q in range(max(0, p - window), min(max_len, p + window + 1)):
            out.add(q)
    return sorted(out)

def truncate_text_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ''
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

def prepare_segmented_sample(bundle: ModelBundle, sample: Dict[str, Any], max_context_tokens: Optional[int]=None):
    tokenizer = bundle.tokenizer
    device = bundle.device
    seg = sample['segments']
    task_text = seg['task_text']
    payload_text = seg['payload_text']
    old_text = seg['old_text']
    distractor_text = seg['distractor_text']
    new_text = seg['new_text']
    query_text = seg['query_text']
    task_ids = encode_text(tokenizer, task_text, add_bos=tokenizer.bos_token_id is not None, device=device)
    payload_ids = encode_text(tokenizer, payload_text, add_bos=False, device=device)
    old_ids = encode_text(tokenizer, old_text, add_bos=False, device=device)
    new_ids = encode_text(tokenizer, new_text, add_bos=False, device=device)
    query_ids = encode_text(tokenizer, query_text, add_bos=False, device=device)
    max_ctx = max_context_tokens or bundle.spec.max_context_tokens
    reserve = new_ids.shape[1] + query_ids.shape[1] + 128
    budget_for_distractor = max(0, max_ctx - task_ids.shape[1] - payload_ids.shape[1] - old_ids.shape[1] - reserve)
    trunc_distractor = truncate_text_by_tokens(tokenizer, distractor_text, budget_for_distractor)
    distractor_ids = encode_text(tokenizer, trunc_distractor, add_bos=False, device=device)
    prefix_ids = torch.cat([task_ids, payload_ids, old_ids, distractor_ids], dim=1)
    prefix_len = prefix_ids.shape[1]
    new_len = new_ids.shape[1]
    with torch.no_grad():
        out_prefix = bundle.model(input_ids=prefix_ids, use_cache=True, output_hidden_states=False, output_attentions=False, return_dict=True)
        past_prefix = to_model_cache(out_prefix.past_key_values)
        attn_mask_new = torch.ones((1, prefix_len + new_len), dtype=torch.long, device=device)
        out_new = bundle.model(input_ids=new_ids, past_key_values=past_prefix, attention_mask=attn_mask_new, use_cache=True, output_hidden_states=True, output_attentions=True, return_dict=True)
        past_after_new = to_model_cache(out_new.past_key_values)
    old_offsets_enc = tokenize_with_offsets(tokenizer, old_text)
    old_rule_core_span = find_substring_char_span(old_text, sample['old_rule_core'])
    old_core_local_positions = []
    if old_rule_core_span is not None:
        old_core_local_positions = char_span_to_token_positions(old_offsets_enc.get('offset_mapping', None), old_rule_core_span[0], old_rule_core_span[1])
    old_core_local_positions = expand_positions(old_core_local_positions, old_ids.shape[1], CFG.edit_local_window)
    task_len = task_ids.shape[1]
    payload_len = payload_ids.shape[1]
    old_start = task_len + payload_len
    old_instruction_positions = list(range(old_start, old_start + old_ids.shape[1]))
    editable_old_positions = [old_start + p for p in old_core_local_positions]
    TRIGGER_PRIORS = {'actually': ('override', 0.8), 'instead': ('override', 0.8), 'update:': ('override', 0.85), 'ignore the previous instruction and': ('override', 0.95), 'correction:': ('correction', 0.92), 'actually, correction:': ('correction', 0.92), 'not the previous setting but': ('negation', 0.88), 'use not the earlier setting but': ('negation', 0.88), 'except for the current payload task,': ('exception', 0.75)}

    def lexical_prefilter_trigger(text: str):
        txt = text.lower()
        for trig, (etype, p) in sorted(TRIGGER_PRIORS.items(), key=lambda x: -len(x[0])):
            pos = txt.find(trig)
            if pos >= 0:
                return {'has_trigger': True, 'trigger_phrase': trig, 'event_type_prior': etype, 'p_prior': p, 'char_start': pos, 'char_end': pos + len(trig)}
        return {'has_trigger': False, 'trigger_phrase': None, 'event_type_prior': 'neutral', 'p_prior': 0.0, 'char_start': None, 'char_end': None}
    trigger_info = lexical_prefilter_trigger(new_text)
    new_offsets_enc = tokenize_with_offsets(tokenizer, new_text)
    trigger_new_token_idx = None
    if trigger_info['has_trigger']:
        hits = char_span_to_token_positions(new_offsets_enc.get('offset_mapping', None), trigger_info['char_start'], trigger_info['char_end'])
        trigger_new_token_idx = hits[0] if hits else 0
    new_full_positions = list(range(prefix_len, prefix_len + new_len))
    U_local_positions = list(range(min((trigger_new_token_idx or 0) + 1, new_len - 1), new_len))
    if len(U_local_positions) == 0:
        U_local_positions = [new_len - 1]
    return {'sample': sample, 'task_text': task_text, 'payload_text': payload_text, 'old_text': old_text, 'distractor_text': trunc_distractor, 'new_text': new_text, 'query_text': query_text, 'task_ids': task_ids, 'payload_ids': payload_ids, 'old_ids': old_ids, 'distractor_ids': distractor_ids, 'new_ids': new_ids, 'query_ids': query_ids, 'prefix_ids': prefix_ids, 'prefix_len': prefix_len, 'new_len': new_len, 'query_len': query_ids.shape[1], 'past_after_new': past_after_new, 'out_new_hidden_states': out_new.hidden_states, 'out_new_attentions': out_new.attentions, 'old_instruction_positions': old_instruction_positions, 'editable_old_positions': editable_old_positions, 'new_positions': new_full_positions, 'trigger_info': trigger_info, 'trigger_new_token_idx': trigger_new_token_idx, 'U_local_positions': U_local_positions}
