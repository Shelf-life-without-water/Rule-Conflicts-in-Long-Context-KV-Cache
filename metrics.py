
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import numpy as np
import torch

from .benchmark import PERSONA_TAGS, STYLE_TAGS
from .interventions import run_query_and_generate
from .model_utils import ModelBundle
from .span_utils import prepare_segmented_sample


def normalize_text(x: str) -> str:
    return re.sub('\\s+', ' ', x.strip())

def infer_format(output: str) -> str:
    t = output.strip()
    if t.startswith('{') and t.endswith('}'):
        return 'json'
    if t.startswith('<response>') and t.endswith('</response>'):
        return 'xml'
    if t.startswith('|field|value|'):
        return 'table'
    if t.startswith('- answer:'):
        return 'bullets'
    return 'plain'

def extract_surface(output: str) -> str:
    fmt = infer_format(output)
    t = output.strip()
    if fmt == 'json':
        try:
            obj = json.loads(t)
            return str(obj['answer'])
        except Exception:
            return ''
    elif fmt == 'xml':
        m = re.search('<answer>(.*?)</answer>', t)
        return m.group(1) if m else ''
    elif fmt == 'table':
        lines = [x.strip() for x in t.splitlines() if x.strip()]
        if len(lines) >= 3 and '|' in lines[2]:
            cols = [x for x in lines[2].split('|') if x != '']
            if len(cols) >= 2:
                return cols[1].strip()
        return ''
    elif fmt == 'bullets':
        return t.replace('- answer:', '', 1).strip()
    else:
        return t

def infer_persona(surface: str) -> str:
    for k, tag in PERSONA_TAGS.items():
        if tag and surface.startswith(tag):
            return k
    return 'neutral'

def infer_style(surface: str) -> Optional[str]:
    s = surface
    for k, tag in PERSONA_TAGS.items():
        if tag and s.startswith(tag):
            s = s[len(tag):]
            break
    for k, tag in STYLE_TAGS.items():
        if tag and s.startswith(tag):
            return k
    return None

def infer_payload_lang_and_correct(surface: str, payload_row: Dict[str, str]):
    if payload_row['zh'] in surface:
        return ('zh', True)
    if payload_row['en'] in surface:
        return ('en', True)
    return (None, False)

def parse_output_attributes(output: str, payload_row: Dict[str, str]) -> Dict[str, Any]:
    fmt = infer_format(output)
    surface = extract_surface(output)
    persona = infer_persona(surface)
    style = infer_style(surface)
    language, payload_ok = infer_payload_lang_and_correct(surface, payload_row)
    return {'format': fmt, 'surface': surface, 'persona': persona, 'style': style, 'language': language, 'payload_ok': payload_ok}

def slot_satisfied(parsed: Dict[str, Any], slot: str, value: str) -> bool:
    if slot == 'language':
        return parsed['language'] == value
    if slot == 'format':
        return parsed['format'] == value
    if slot == 'persona':
        return parsed['persona'] == value
    if slot == 'style':
        return parsed['style'] == value
    raise ValueError(slot)

def mean_slot_accuracy(parsed: Dict[str, Any], constraints: Dict[str, str]) -> float:
    if len(constraints) == 0:
        return 1.0
    return float(np.mean([slot_satisfied(parsed, s, v) for s, v in constraints.items()]))

def compute_sample_metrics(sample: Dict[str, Any], output_text: str):
    parsed = parse_output_attributes(output_text, sample['payload'])
    current_constraints = sample['current_constraints']
    stable_constraints = sample['stable_constraints']
    old_constraints = sample['old_constraints']
    overridden_slots = sample['overridden_slots']
    ifc = int(parsed['payload_ok'] and all((slot_satisfied(parsed, s, v) for s, v in current_constraints.items())))
    cr = 0 if len(overridden_slots) == 0 else int(any((slot_satisfied(parsed, s, old_constraints[s]) for s in overridden_slots if s in old_constraints)))
    cfr = 0 if len(stable_constraints) == 0 else int(any((not slot_satisfied(parsed, s, v) for s, v in stable_constraints.items())))
    exact_match_current = int(normalize_text(output_text) == normalize_text(sample['gold_current']))
    return {'ifc': ifc, 'cr': cr, 'cfr': cfr, 'exact_match_current': exact_match_current, 'payload_ok': int(parsed['payload_ok']), 'current_slot_acc': mean_slot_accuracy(parsed, current_constraints), 'stable_slot_acc': mean_slot_accuracy(parsed, stable_constraints), 'parsed_format': parsed['format'], 'parsed_language': parsed['language'], 'parsed_persona': parsed['persona'], 'parsed_style': parsed['style']}

def evaluate_none(bundle: ModelBundle, sample: Dict[str, Any]):
    prepared = prepare_segmented_sample(bundle, sample, max_context_tokens=bundle.spec.max_context_tokens)
    keep = torch.ones(prepared['prefix_len'] + prepared['new_len'], dtype=torch.long, device=bundle.device)
    gen = run_query_and_generate(bundle=bundle, past_prequery=prepared['past_after_new'], prequery_keep_mask=keep, query_ids=prepared['query_ids'], gold_answer_text=sample['gold_current'], payload_row=sample['payload'])
    met = compute_sample_metrics(sample, gen['output_text'])
    return {**gen, **met}
