
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from .config import CFG, InterpConfig
from .env import SEED


ALL_SLOTS = ['language', 'format', 'persona', 'style']
LANG_VALUES = ['en', 'zh']
FORMAT_VALUES = ['plain', 'json', 'xml', 'bullets', 'table']
PERSONA_VALUES = ['neutral', 'professor', 'advisor', 'auditor']
STYLE_VALUES = ['terse', 'courtly', 'academic']
SLOT_VALUE_SPACE = {'language': LANG_VALUES, 'format': FORMAT_VALUES, 'persona': PERSONA_VALUES, 'style': STYLE_VALUES}
PERSONA_TAGS = {'neutral': '', 'professor': '[PROFESSOR]', 'advisor': '[ADVISOR]', 'auditor': '[AUDITOR]'}
STYLE_TAGS = {'terse': '[TERSE]', 'courtly': '[COURTLY]', 'academic': '[ACADEMIC]'}
TRIGGER_LIBRARY = {'override': ['Actually,', 'Instead,', 'Update:', 'Ignore the previous instruction and'], 'correction': ['Correction:', 'Actually, correction:'], 'negation': ['Not the previous setting but', 'Use not the earlier setting but'], 'exception': ['Except for the current payload task,'], 'neutral': ['Actually,', 'Instead,']}

def build_payload_bank(n: int=2000):
    greek_en = ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'SIGMA', 'OMEGA', 'THETA', 'EPSILON']
    greek_zh = ['阿尔法', '贝塔', '伽马', '德尔塔', '西格玛', '欧米伽', '西塔', '艾普西龙']
    colors_en = ['RED', 'BLUE', 'GREEN', 'SILVER', 'GOLD', 'BLACK', 'WHITE', 'PURPLE']
    colors_zh = ['红', '蓝', '绿', '银', '金', '黑', '白', '紫']
    objs_en = ['SPHERE', 'CUBE', 'RING', 'TOKEN', 'ORBIT', 'PANEL', 'BEACON', 'SEAL']
    objs_zh = ['球体', '立方体', '环', '令牌', '轨道', '面板', '信标', '封印']
    bank = []
    for i in range(n):
        en = f'PAYLOAD_{i:04d}_{greek_en[i % len(greek_en)]}_{colors_en[i * 3 % len(colors_en)]}_{objs_en[i * 5 % len(objs_en)]}'
        zh = f'载荷_{i:04d}_{greek_zh[i % len(greek_zh)]}_{colors_zh[i * 3 % len(colors_zh)]}_{objs_zh[i * 5 % len(objs_zh)]}'
        bank.append({'payload_id': f'ITEM_{i:04d}', 'en': en, 'zh': zh})
    return bank
PAYLOAD_BANK = build_payload_bank(2500)

def choose_other(slot: str, old_value: str, rng: random.Random):
    vals = [x for x in SLOT_VALUE_SPACE[slot] if x != old_value]
    return rng.choice(vals)

def render_surface(payload_row: Dict[str, str], constraints: Dict[str, str]) -> str:
    lang = constraints.get('language', 'en')
    persona = constraints.get('persona', 'neutral')
    style = constraints.get('style', None)
    payload = payload_row['zh'] if lang == 'zh' else payload_row['en']
    prefix_parts = []
    if PERSONA_TAGS[persona]:
        prefix_parts.append(PERSONA_TAGS[persona])
    if style is not None and STYLE_TAGS.get(style, ''):
        prefix_parts.append(STYLE_TAGS[style])
    return ''.join(prefix_parts) + payload

def render_answer(payload_row: Dict[str, str], constraints: Dict[str, str]) -> str:
    fmt = constraints.get('format', 'plain')
    surface = render_surface(payload_row, constraints)
    if fmt == 'plain':
        return surface
    elif fmt == 'json':
        return json.dumps({'answer': surface}, ensure_ascii=False, separators=(',', ':'))
    elif fmt == 'xml':
        return f'<response><answer>{surface}</answer></response>'
    elif fmt == 'bullets':
        return f'- answer: {surface}'
    elif fmt == 'table':
        return f'|field|value|\n|---|---|\n|answer|{surface}|'
    else:
        raise ValueError(fmt)

def instruction_piece(slot: str, value: str) -> str:
    if slot == 'language':
        return 'use Chinese output' if value == 'zh' else 'use English output'
    if slot == 'format':
        return {'plain': 'output in plain text', 'json': 'output as compact JSON', 'xml': 'output as XML', 'bullets': 'output as a bullet list', 'table': 'output as a markdown table'}[value]
    if slot == 'persona':
        return {'neutral': 'use no persona tag', 'professor': 'prefix with [PROFESSOR]', 'advisor': 'prefix with [ADVISOR]', 'auditor': 'prefix with [AUDITOR]'}[value]
    if slot == 'style':
        return {'terse': 'prefix with [TERSE]', 'courtly': 'prefix with [COURTLY]', 'academic': 'prefix with [ACADEMIC]'}[value]
    raise ValueError(slot)

def constraints_to_natural_instruction(constraints: Dict[str, str], scope_text: str='For the current payload task', suffix: Optional[str]=None) -> str:
    parts = [instruction_piece(k, v) for k, v in constraints.items()]
    if not parts:
        s = f'{scope_text}, keep the answer unconstrained.'
    else:
        s = f'{scope_text}, ' + ', and '.join(parts) + '.'
    if suffix:
        s += ' ' + suffix
    return s

def sample_distractor(rng: random.Random, pool: List[str], target_chars: int) -> str:
    if not pool:
        return 'DUMMY DISTRACTOR. ' * 200
    txt = rng.choice(pool)
    if len(txt) <= target_chars:
        return txt
    start = rng.randint(0, max(0, len(txt) - target_chars))
    return txt[start:start + target_chars]

def compose_prompt(payload_row: Dict[str, str], old_block: str, distractor: str, new_block: str, query: str) -> Dict[str, str]:
    task_text = '[TASK]\nYou are solving a payload formatting task.\nUse the canonical payload surface given in [PAYLOAD].\nA newer rule for the current payload overrides an earlier conflicting rule.\nKeep earlier current-payload rules only when they are compatible.\nIgnore rules that apply to other tasks.\nReturn exactly one final answer and nothing else.\n'
    payload_text = f"[PAYLOAD]\npayload_id={payload_row['payload_id']}\nenglish_surface={payload_row['en']}\nchinese_surface={payload_row['zh']}\n"
    return {'task_text': task_text, 'payload_text': payload_text, 'old_text': f'[OLD_RULES]\n{old_block}\n', 'distractor_text': f'[DISTRACTOR]\n{distractor}\n', 'new_text': f'[CURRENT_RULE]\n{new_block}\n', 'query_text': f'[QUERY]\n{query}\n[ANSWER]\n'}

def build_one_interp_sample(family: str, target_slot: str, idx: int, rng: random.Random, distractor_pool: List[str]) -> Dict[str, Any]:
    payload = PAYLOAD_BANK[idx % len(PAYLOAD_BANK)]
    distractor_chars = rng.choice(CFG.distractor_char_choices)
    old_repeat = rng.choice(CFG.old_rule_repeat_choices)
    old_value = rng.choice(SLOT_VALUE_SPACE[target_slot])
    new_value = choose_other(target_slot, old_value, rng)
    other_slots = [s for s in ALL_SLOTS if s != target_slot]
    stable_slot = rng.choice(other_slots)
    stable_value = rng.choice(SLOT_VALUE_SPACE[stable_slot])
    old_constraints = {target_slot: old_value, stable_slot: stable_value}
    current_constraints = {}
    stable_constraints = {}
    new_constraints = {}
    overridden_slots = []
    if family == 'true_override':
        current_constraints = {target_slot: new_value, stable_slot: stable_value}
        stable_constraints = {stable_slot: stable_value}
        new_constraints = {target_slot: new_value}
        overridden_slots = [target_slot]
        trigger_type = rng.choice(['override', 'correction', 'negation'])
    elif family == 'compatible_refinement':
        add_slot = rng.choice([s for s in other_slots if s != stable_slot]) if len(other_slots) > 1 else stable_slot
        add_value = rng.choice(SLOT_VALUE_SPACE[add_slot])
        current_constraints = {target_slot: old_value, stable_slot: stable_value, add_slot: add_value}
        stable_constraints = {target_slot: old_value, stable_slot: stable_value}
        new_constraints = {add_slot: add_value}
        overridden_slots = []
        trigger_type = rng.choice(['override', 'correction'])
    elif family == 'false_trigger':
        current_constraints = old_constraints.copy()
        stable_constraints = old_constraints.copy()
        new_constraints = old_constraints.copy()
        overridden_slots = []
        trigger_type = 'neutral'
    elif family == 'irrelevant_control':
        current_constraints = {stable_slot: stable_value}
        stable_constraints = {stable_slot: stable_value}
        new_constraints = current_constraints.copy()
        overridden_slots = [target_slot]
        trigger_type = rng.choice(['override', 'correction'])
    else:
        raise ValueError(family)
    trigger_phrase = rng.choice(TRIGGER_LIBRARY[trigger_type])
    old_instruction_scope = 'For the current payload task'
    if family == 'irrelevant_control':
        old_instruction_scope = 'For weather-report tasks only'
    old_rule_core = instruction_piece(target_slot, old_value)
    old_instruction = constraints_to_natural_instruction(old_constraints, scope_text=old_instruction_scope, suffix='This was an earlier task-specific rule.')
    if family == 'true_override':
        if trigger_type == 'negation':
            new_instruction = f'{trigger_phrase} for the current payload task, replace the earlier conflicting rule. The valid rule now is: {instruction_piece(target_slot, new_value)}.'
        else:
            new_instruction = f"{trigger_phrase} for the current payload task, replace the earlier conflicting rule. The valid rule now is: {constraints_to_natural_instruction({target_slot: new_value}, scope_text='For the current payload task', suffix='This rule is currently valid.')}"
    elif family == 'compatible_refinement':
        new_instruction = f"{trigger_phrase} for the current payload task, keep the earlier compatible rules and add this new valid rule: {constraints_to_natural_instruction(new_constraints, scope_text='For the current payload task', suffix='This additional rule is currently valid.')}"
    elif family == 'false_trigger':
        new_instruction = f'{trigger_phrase} this is only a clarification for the current payload task. Do not replace the earlier valid rule. The previously valid task-specific constraints remain unchanged.'
    elif family == 'irrelevant_control':
        new_instruction = f"{trigger_phrase} the earlier rule applied only to weather-report tasks and is not valid for the current payload task. For the current payload task, the valid rule is: {constraints_to_natural_instruction(current_constraints, scope_text='For the current payload task', suffix='This rule is currently valid.')}"
    old_block = '\n'.join([old_instruction] * old_repeat)
    distractor = sample_distractor(rng, distractor_pool, target_chars=distractor_chars)
    query = f"Output exactly the final answer string for payload_id={payload['payload_id']}.\nUse the canonical payload surface from [PAYLOAD].\nFollow only the currently valid rules for the current payload task.\nDo not explain. Do not mention rules. Return only the final rendered answer.\n"
    segments = compose_prompt(payload, old_block, distractor, new_instruction, query)
    return {'sample_id': f'{family}_{target_slot}_{idx:06d}', 'family': family, 'target_slot': target_slot, 'event_type': trigger_type, 'trigger_phrase': trigger_phrase, 'payload': payload, 'old_constraints': old_constraints, 'new_constraints': new_constraints, 'current_constraints': current_constraints, 'stable_constraints': stable_constraints, 'overridden_slots': overridden_slots, 'old_rule_core': old_rule_core, 'target_old_value': old_value, 'target_new_value': new_value, 'old_repeat': old_repeat, 'distractor_chars': distractor_chars, 'segments': segments, 'gold_current': render_answer(payload, current_constraints), 'gold_old_full': render_answer(payload, old_constraints) if old_constraints else ''}

def build_interp_splits(cfg: InterpConfig, distractor_pool: List[str], seed: int=SEED):
    rng = random.Random(seed)
    splits = {'probe_train': [], 'probe_test': [], 'erase_test': []}
    counts = {'probe_train': cfg.n_probe_train_per_slot_family, 'probe_test': cfg.n_probe_test_per_slot_family, 'erase_test': cfg.n_erase_test_per_slot_family}
    global_idx = 0
    for split_name, n in counts.items():
        for family in cfg.active_families:
            for target_slot in cfg.active_slots:
                for _ in range(n):
                    splits[split_name].append(build_one_interp_sample(family, target_slot, global_idx, rng, distractor_pool))
                    global_idx += 1
    return splits
