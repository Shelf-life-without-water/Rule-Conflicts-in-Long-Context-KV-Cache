
from __future__ import annotations

import copy
import gc
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_SPECS, ModelSpec

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    from transformers import DynamicCache


@dataclass
class ModelBundle:
    name: str
    spec: ModelSpec
    tokenizer: Any
    model: Any
    device: torch.device
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    kv_dim: int

def get_main_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_bundle(model_name: str) -> ModelBundle:
    spec = MODEL_SPECS[model_name]
    assert spec.path.exists(), f'Model path not found: {spec.path}'
    tokenizer = AutoTokenizer.from_pretrained(str(spec.path), trust_remote_code=spec.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if spec.preferred_dtype == 'bfloat16' else torch.float16
    else:
        dtype = torch.float32
    try:
        import accelerate
        use_device_map = torch.cuda.is_available()
    except Exception:
        use_device_map = False
    model_kwargs = dict(trust_remote_code=spec.trust_remote_code, dtype=dtype, low_cpu_mem_usage=True, attn_implementation='eager')
    if use_device_map:
        model_kwargs['device_map'] = 'auto'
    try:
        model = AutoModelForCausalLM.from_pretrained(str(spec.path), **model_kwargs)
    except TypeError as e:
        if 'dtype' in str(e):
            model_kwargs_fallback = dict(model_kwargs)
            model_kwargs_fallback.pop('dtype', None)
            model_kwargs_fallback['torch_dtype'] = dtype
            model = AutoModelForCausalLM.from_pretrained(str(spec.path), **model_kwargs_fallback)
        else:
            raise
    if not use_device_map and torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    config = model.config
    num_layers = int(config.num_hidden_layers)
    hidden_size = int(config.hidden_size)
    if hasattr(config, 'num_attention_heads'):
        num_heads = int(config.num_attention_heads)
    elif hasattr(config, 'num_heads'):
        num_heads = int(config.num_heads)
    else:
        raise AttributeError('No num_attention_heads or num_heads found in config.')
    num_kv_heads = int(getattr(config, 'num_key_value_heads', num_heads))
    head_dim = int(getattr(config, 'head_dim', hidden_size // num_heads))
    kv_dim = num_kv_heads * head_dim
    return ModelBundle(name=model_name, spec=spec, tokenizer=tokenizer, model=model, device=get_main_device(model), num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, kv_dim=kv_dim)

def cleanup_bundle(bundle: Optional[ModelBundle]):
    if bundle is None:
        return
    del bundle.model
    del bundle.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _layer_to_kv_tuple(layer_obj):
    if isinstance(layer_obj, (tuple, list)):
        return tuple(layer_obj)
    for k_name, v_name in [('keys', 'values'), ('key', 'value'), ('key_states', 'value_states'), ('k', 'v')]:
        if hasattr(layer_obj, k_name) and hasattr(layer_obj, v_name):
            return (getattr(layer_obj, k_name), getattr(layer_obj, v_name))
    tensor_attrs = []
    for name in dir(layer_obj):
        if name.startswith('_'):
            continue
        try:
            x = getattr(layer_obj, name)
            if torch.is_tensor(x):
                tensor_attrs.append((name, x))
        except Exception:
            continue
    if len(tensor_attrs) >= 2:
        tensor_attrs_sorted = sorted(tensor_attrs, key=lambda kv: 0 if 'key' in kv[0].lower() or kv[0].lower() in {'k', 'keys'} else 1 if 'value' in kv[0].lower() or kv[0].lower() in {'v', 'values'} else 2)
        return (tensor_attrs_sorted[0][1], tensor_attrs_sorted[1][1])
    raise TypeError(f'Cannot convert layer cache object of type {type(layer_obj)} to (k, v)')

def to_legacy_cache(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)
    if hasattr(past_key_values, 'to_legacy_cache'):
        try:
            pkv = past_key_values.to_legacy_cache()
            if isinstance(pkv, (tuple, list)):
                return tuple(pkv)
        except Exception:
            pass
    if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
        try:
            return tuple(((k, v) for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)))
        except Exception:
            pass
    if hasattr(past_key_values, 'layers'):
        return tuple((_layer_to_kv_tuple(layer) for layer in past_key_values.layers))
    if hasattr(past_key_values, '__len__') and hasattr(past_key_values, '__getitem__'):
        return tuple((_layer_to_kv_tuple(past_key_values[i]) for i in range(len(past_key_values))))
    raise TypeError(f'Unsupported cache type: {type(past_key_values)}')

def to_model_cache(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values
    if isinstance(past_key_values, (tuple, list)):
        legacy = tuple(past_key_values)
        if hasattr(DynamicCache, 'from_legacy_cache'):
            try:
                return DynamicCache.from_legacy_cache(legacy)
            except Exception:
                pass
        cache = DynamicCache()
        for layer_idx, layer in enumerate(legacy):
            layer = _layer_to_kv_tuple(layer)
            k, v = (layer[0], layer[1])
            try:
                cache.update(k, v, layer_idx)
            except TypeError:
                cache.update(k, v, layer_idx, cache_kwargs=None)
        return cache
    raise TypeError(f'Unsupported cache type for model forward: {type(past_key_values)}')

def clone_legacy_cache(past_key_values):
    pkv = to_legacy_cache(past_key_values)
    if pkv is None:
        return None
    cloned = []
    for layer in pkv:
        new_layer = []
        for x in layer:
            if torch.is_tensor(x):
                new_layer.append(x.clone())
            else:
                new_layer.append(copy.deepcopy(x))
        cloned.append(tuple(new_layer))
    return tuple(cloned)

def encode_text(tokenizer, text: str, add_bos: bool=False, device: Optional[torch.device]=None):
    ids = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids
    if add_bos and tokenizer.bos_token_id is not None:
        bos = torch.tensor([[tokenizer.bos_token_id]], dtype=ids.dtype)
        ids = torch.cat([bos, ids], dim=1)
    if device is not None:
        ids = ids.to(device)
    return ids
