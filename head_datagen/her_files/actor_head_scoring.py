"""Score candidates with the frozen selection head from hidden states captured inside the actor's sglang server.

`openwebrl.actor_head_capture` exports, per candidate, the hidden states the actor computed while generating it
(last prompt token + all generated tokens but the final one), and a follow-up request computes the final token.
Features are pooled with the same `pool_candidate` code and float16 rounding as the offline and sidecar paths.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch

from openwebrl.selection_head_features import END, candidate_tokens_from_ids, pool_candidate

END_ID, NEWLINE_ID = 151645, 198      # Qwen3 <|im_end|> and "\n"


def load_export(directory, rid, timeout=30.0):
    """Wait for the scheduler's export file (written just before the HTTP response is sent)."""
    path = Path(directory) / f"{rid}.pt"
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"no capture export for {rid}")
        time.sleep(0.005)
    payload = torch.load(path, map_location="cpu")
    path.unlink()
    return payload


def tail_text(tokenizer, output_ids):
    """The candidate as text, ending with the end token, for the follow-up request."""
    ids, _ = candidate_tokens_from_ids(tokenizer, output_ids)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    assert text.endswith(END)
    return text


def candidate_features(gen, tail, tokenizer, layers, pools):
    """(state [L,H], features [L*P,H], info) for one candidate from its generation and follow-up exports."""
    if not gen["complete"] or (tail is not None and not tail["complete"]):
        raise ValueError("incomplete capture")
    n = gen["input_len"]
    ids, spans = candidate_tokens_from_ids(tokenizer, gen["output_ids"])
    rows = dict(zip(gen["positions"], gen["hidden"]))
    if tail is None:                                 # end token fed during generation: every row is in `rows`
        tail_rows, m, aligned = {}, n + len(ids), True
    else:
        tail_rows, m = dict(zip(tail["positions"], tail["hidden"])), tail["input_len"]
        # the follow-up re-tokenizes prompt + candidate text; it aligns when its length and final ids match
        k = min(len(tail["input_tail_ids"]), len(ids))
        aligned = m == n + len(ids) and tail["input_tail_ids"][-k:] == ids[-k:]
    # tokens never fed back during generation (the final one, or two after a length stop) come from the follow-up,
    # matched from its end so a re-tokenized prefix cannot shift them
    picked, approximated = [], 0
    for j in range(len(ids)):
        h = rows.get(n + j)
        if h is None:
            h = tail_rows.get(m - (len(ids) - j))
        if h is None and j == len(ids) - 2 and len(gen["output_ids"]) == len(ids) - 1 and picked:
            # length stop: the last generated token may be cached by the server without ever being exported; it is one
            # token of a ~1,000-token mean, so the preceding token's state stands in (counted in info)
            h, approximated = picked[-1], approximated + 1
        if h is None:
            raise ValueError(f"missing candidate token {j} of {len(ids)} (follow-up length {m}, expected {n + len(ids)})")
        picked.append(h)
    stack = torch.stack(picked)                                           # [T, L, H]
    hidden = {layer: stack[:, k] for k, layer in enumerate(layers)}
    state = rows[n - 1]                                                   # [L, H]
    return state, pool_candidate(hidden, spans, layers, pools), dict(aligned=aligned, tokens=len(ids), approximated=approximated)


def score(head, config, states_features):
    """states_features: list of (state, features) for the five candidates; returns head logits (list of floats)."""
    state = states_features[0][0]
    feats = torch.stack([f for _, f in states_features]).float()
    # offline features were stored as float16 before scoring; round identically
    feats = feats.to(torch.float16).to(torch.float32)
    s = state.float().to(torch.float16).to(torch.float32)
    with torch.no_grad():
        logits = head(feats.reshape(1, len(states_features), -1), s.reshape(1, -1))[0][0]
    return [float(x) for x in logits]


_BUNDLE = {}


def bundle(head_path, tokenizer_path):
    """Head (CPU) and tokenizer, loaded once per process."""
    key = (str(head_path), str(tokenizer_path))
    if key not in _BUNDLE:
        import hashlib
        from transformers import AutoTokenizer
        from openwebrl.selection_head import CandidateSetHead, HeadConfig
        saved = torch.load(head_path, map_location="cpu")
        config = HeadConfig.from_json(saved["config"])
        head = CandidateSetHead(config); head.load_state_dict(saved["state_dict"]); head.eval()
        _BUNDLE[key] = dict(head=head, config=config, tokenizer=AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True),
                            head_params=sum(p.numel() for p in head.parameters()),
                            head_sha256=hashlib.sha256(Path(head_path).read_bytes()).hexdigest())
    return _BUNDLE[key]


async def generate_with_tail(generate, url, prompt_text, images, rid, *, tokenizer, capture_dir, timeout=180):
    """Run one candidate's generation, then immediately its follow-up, so follow-ups overlap slower candidates.

    `generate` is the actor request coroutine; returns (actor output tuple, generation export, follow-up meta).
    The controller normalizes candidates in place (appending "<|im_end|>\\n"); the server's ids are authoritative.
    """
    import asyncio
    import httpx
    output = await generate
    gen = await asyncio.to_thread(load_export, capture_dir, rid)
    ids, server = list(output[1]), gen["output_ids"]
    if ids[:len(server)] != server or ids[len(server):] not in ([], [END_ID], [NEWLINE_ID], [END_ID, NEWLINE_ID]):
        raise ValueError("capture export differs from the generated candidate")
    if gen.get("final_fed"):                         # the actor already computed the end token: no follow-up
        return output, gen, None
    payload = dict(text=prompt_text + tail_text(tokenizer, server), rid=rid + "-tail",
                   sampling_params=dict(temperature=0.0, max_new_tokens=1))
    if images:
        payload["image_data"] = list(images)
    async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
    return output, gen, r.json()["meta_info"]


async def score_cached(gens, metas, rids, *, head_path, tokenizer_path, capture_dir, params_for_flops):
    """Scores from the generation and follow-up exports, plus telemetry for the work added to the actor."""
    import asyncio
    b = bundle(head_path, tokenizer_path)
    tok, config = b["tokenizer"], b["config"]
    started = time.monotonic()

    def features():
        return [candidate_features(gen, None if meta is None else load_export(capture_dir, rid + "-tail"), tok,
                                   config.layers, config.pools) for gen, meta, rid in zip(gens, metas, rids)]
    cap = await asyncio.to_thread(features)
    scores = score(b["head"], config, [(s, f) for s, f, _ in cap])
    tails = [m for m in metas if m is not None]
    uncached = sum(m["prompt_tokens"] - (m.get("cached_tokens") or 0) for m in tails)
    sampled = sum(m.get("completion_tokens", 1) for m in tails) + (len(metas) - len(tails))   # extra decode step each
    return dict(scores=scores, input_tokens=uncached, output_tokens=sampled, seconds=time.monotonic() - started,
                flops=2 * params_for_flops * (uncached + sampled) + 2 * b["head_params"] * len(cap),
                tails_aligned=sum(x["aligned"] for _, _, x in cap), candidate_tokens=sum(x["tokens"] for _, _, x in cap),
                approximated_tokens=sum(x["approximated"] for _, _, x in cap), follow_ups=len(tails), head_sha256=b["head_sha256"])
