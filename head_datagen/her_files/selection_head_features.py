"""Frozen-actor features for the candidate-set selection head.

The state prefix (actor prompt + screenshot) is encoded once; each candidate is
teacher-forced after a copy of the prefix KV cache, exactly as the actor would
have generated it. Qwen3-VL continuation positions come from the prefill's
rope deltas, so every call passes an explicit cache_position.
"""
from __future__ import annotations

import copy

import torch

from openwebrl.selection_head import CANDIDATE_POOLS

END = "<|im_end|>"
THINK_CLOSE = "</think>"


def candidate_tokens(tokenizer, response):
    """Token ids for one response (+ end token) and its thought/action token spans."""
    text = response if response.endswith(END) else response + END
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    close = text.rfind(THINK_CLOSE)
    action_start = 0 if close < 0 else close + len(THINK_CLOSE)
    action_end = len(text) - len(END)
    thought = [i for i, (a, b) in enumerate(offsets) if close >= 0 and b <= close]
    action = [i for i, (a, b) in enumerate(offsets) if a >= action_start and b <= action_end]
    span = lambda idx: (idx[0], idx[-1] + 1) if idx else None
    return ids, {"thought": span(thought), "action": span(action)}


def candidate_tokens_from_ids(tokenizer, ids):
    """Spans for the actor's exact generated ids (same semantics as candidate_tokens)."""
    end = tokenizer.convert_tokens_to_ids(END)
    ids = list(ids)
    # the actor's normalized output ends "<|im_end|>\n"; training candidates end at <|im_end|>, so drop the newline
    newline = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline) == 1 and len(ids) >= 2 and ids[-2] == end and ids[-1] == newline[0]:
        ids = ids[:-1]
    ids = ids if ids and ids[-1] == end else ids + [end]
    close = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    at = max((i for i, t in enumerate(ids) if t == close), default=-1)
    thought = (0, at) if at > 0 else None
    action = (at + 1, len(ids) - 1) if len(ids) - 1 > at + 1 else None
    return ids, {"thought": thought, "action": action}


def pool_candidate(hidden, spans, layers, pools=CANDIDATE_POOLS):
    """hidden: tuple of per-layer [T,H] for the candidate tokens -> [L*P, H] (layer-major)."""
    out = []
    for layer in layers:
        h = hidden[layer]
        for pool in pools:
            if pool == "last":
                out.append(h[-1])
                continue
            span = spans[pool.removesuffix("_mean")]
            # An empty span (no reasoning, or unparseable action) falls back to the end token.
            out.append(h[span[0]:span[1]].mean(0) if span else h[-1])
    return torch.stack(out)


def encode_prefix(model, prefix_inputs, layers, use_cache=True):
    """Prompt + screenshot forward: state vector [L,H] and its KV cache."""
    device = prefix_inputs["input_ids"].device
    prefix_len = prefix_inputs["input_ids"].shape[1]
    out = model(**prefix_inputs, use_cache=use_cache, output_hidden_states=True,
                cache_position=torch.arange(prefix_len, device=device))
    state = torch.stack([out.hidden_states[l][0, -1] for l in layers])
    return state, out.past_key_values if use_cache else None


def encode_candidates(model, cache, prefix_len, candidates, layers, pools=CANDIDATE_POOLS,
                      batch_size=1, pad_id=0):
    """Teacher-force candidates after a copy of the prefix cache -> [N, L*P, H].

    Shorter candidates are right-padded; pads come after every real token, so
    causal attention keeps real states exact. Differentiable when grad is enabled
    (the prefix cache itself is treated as a constant).
    """
    device = next(model.parameters()).device
    features = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long, device=device)
        for row, (tokens, _) in enumerate(chunk):
            ids[row, :len(tokens)] = torch.tensor(tokens, device=device)
        past = copy.deepcopy(cache)
        if len(chunk) > 1:
            past.batch_repeat_interleave(len(chunk))
        step = model(input_ids=ids, past_key_values=past, use_cache=True, output_hidden_states=True,
                     cache_position=torch.arange(prefix_len, prefix_len + width, device=device))
        for row, (tokens, spans) in enumerate(chunk):
            hidden = [h[row, :len(tokens)] for h in step.hidden_states]
            features.append(pool_candidate(hidden, spans, layers, pools))
        del past, step
    return torch.stack(features)


@torch.inference_mode()
def extract(model, prefix_inputs, candidates, layers, pools=CANDIDATE_POOLS, reuse_prefix=True,
            batch_size=1, pad_id=0):
    """Encode one state and its candidates.

    prefix_inputs: processor output for the actor prompt (batch 1, on device).
    candidates: list of (token_ids, spans) from candidate_tokens.
    Returns state [L,H] (last prefix token) and candidate features [N, L*P, H].
    reuse_prefix=False runs a full forward per candidate (reference for checks).
    """
    device = prefix_inputs["input_ids"].device
    prefix_len = prefix_inputs["input_ids"].shape[1]
    state, cache = encode_prefix(model, prefix_inputs, layers, use_cache=reuse_prefix)
    if reuse_prefix:
        return state, encode_candidates(model, cache, prefix_len, candidates, layers, pools, batch_size, pad_id)
    features = []
    for ids, spans in candidates:
        joined = dict(prefix_inputs)
        joined["input_ids"] = torch.cat([prefix_inputs["input_ids"], torch.tensor([ids], device=device)], 1)
        joined["attention_mask"] = torch.ones_like(joined["input_ids"])
        step = model(**joined, use_cache=False, output_hidden_states=True)
        hidden = [h[0, prefix_len:] for h in step.hidden_states]
        features.append(pool_candidate(hidden, spans, layers, pools))
    return state, torch.stack(features)
