"""Capture the actor's own hidden states inside its sglang server, for a selection head that reuses the actor's KV cache.

Loaded through sglang's `--forward-hooks` (the hook factory is imported in the scheduler process). Forward hooks
read the residual stream at the head's layers while the actor generates; nothing is recomputed. For requests whose
`rid` starts with `actorhead-`, the scheduler writes, when the request finishes, the hidden states of the last prompt
token and of every generated token that was fed back through the model to `$ACTORHEAD_DIR/<rid>.pt`. A generated token's hidden
state is computed only when it is fed back, so a capture request that stops on its end token is finished one decode
step later (sglang's own deferred `to_finish`): that step feeds <|im_end|>, and the one extra sampled token is removed
with its log-probabilities before the response is sent. Outputs are unchanged. Only a candidate stopped by the
token limit still needs a follow-up request (prompt plus full candidate), which computes just its final tokens.

Layer numbering follows the head's features (Hugging Face `hidden_states` as recorded by the offline extractor):
index i is the residual stream after decoder layer i-1 (sum of the layer output and its residual). For i = 36 this
is the last decoder layer's output before the final RMSNorm; parity tests confirmed this against the reference.

The hooks never synchronize the GPU: each forward stores its kept rows, starts a non-blocking copy of its
request-slot indices to pinned memory and records a CUDA event. Ownership is resolved only when a request finishes,
by indexing completed records (event done) in forward order so that for each (slot, position) the newest record
wins; slot reuse therefore cannot leak another request's states. Exports copy on a side stream, so they wait only
for the forwards they read, never for work queued after them.

CUDA graphs: hooks do not run when a captured decode graph is replayed. While sglang captures each decode batch
size, the hooks instead record a copy of the residual stream into a fixed per-(batch size, layer) buffer inside the
graph; the patched `CudaGraphRunner.replay` then turns that buffer into a record for the real batch. sglang captures
its graphs before it registers forward hooks, so the first decode replay re-captures them once with the hooks active.
Prefill/extend forwards (never graphed here) keep the eager hook path.
"""
from collections import OrderedDict
import os
from pathlib import Path
import threading

import torch

PREFIX = "actorhead-"
TAIL = 8             # follow-up requests (rid ending "-tail") export up to this many final prompt positions
RING = 6000          # forwards kept; a request spans at most ~1,100 forwards (prefill chunks + 1,024 decodes)
_lock = threading.Lock()
_records = OrderedDict()      # id -> dict(pool, req_of, positions, index, rows={layer: tensor}, event)
_index = {}                   # slot -> {position: (record id, row)}
_state = dict(layers=None, patched=False, current=None, next_id=0, indexed=-1, side=None, capturing=False)
_static = {}                  # (padded batch size, layer) -> graph-owned buffer written during decode-graph replay


def _to_device(values, device):
    return torch.tensor(values, dtype=torch.int64).pin_memory().to(device, non_blocking=True)


def _plan(fb):
    """Slots, positions and row indices (None = every row) that this forward keeps."""
    mode = fb.forward_mode
    if mode.is_idle():
        return None
    if mode.is_decode():
        return dict(pool=fb.req_pool_indices, req_of=None, positions=[int(n) - 1 for n in fb.seq_lens_cpu.tolist()], index=None)
    keep_all = os.environ.get("ACTORHEAD_EXTEND") == "all"     # parity tests only: keep every extend token
    positions, index, req_of, row = [], [], [], 0
    for k, (prefix, length) in enumerate(zip(fb.extend_prefix_lens_cpu, fb.extend_seq_lens_cpu)):
        # short extends (a cache hit's uncached tail, e.g. a follow-up's final tokens) are kept whole
        rows = range(length) if keep_all or length <= TAIL else [length - 1]
        positions += [prefix + r for r in rows]; index += [row + r for r in rows]; req_of += [k] * len(rows)
        row += length
    return dict(pool=fb.req_pool_indices, req_of=req_of, positions=positions, index=_to_device(index, fb.req_pool_indices.device))


def hook_specs(layers=(18, 27, 36)):
    """The `--forward-hooks` value for these hidden-state indices (one spec per decoder layer)."""
    return [dict(name=f"actorhead-{l}", hook_factory="openwebrl.actor_head_capture:factory",
                 target_modules=[f"model.layers.{l - 1}"], config=dict(layer=l, layers=list(layers))) for l in layers]


def factory(config):
    layer, layers = config["layer"], list(config["layers"])
    _state["layers"] = layers
    _patch_scheduler()

    def hook(module, inputs, output):
        if _state["capturing"]:                      # sglang is capturing decode graphs
            if torch.cuda.is_current_stream_capturing():
                hidden, residual = output
                stream = hidden + residual if residual is not None else hidden
                key = (stream.shape[0], layer)
                if key not in _static:
                    _static[key] = torch.empty_like(stream)
                _static[key].copy_(stream)           # recorded into the graph; replayed every decode step
            return                                   # warm-up runs before capture use dummy batches: keep nothing
        if layer == min(layers):                     # first captured layer of this forward: start a record
            plan = _plan(inputs[2])
            if plan is not None:     # slot indices to pinned host memory without waiting for the GPU
                pool = plan.pop("pool")
                plan["pool_cpu"] = torch.empty(pool.shape, dtype=pool.dtype, pin_memory=True)
                plan["pool_cpu"].copy_(pool, non_blocking=True)
            _state["current"] = None if plan is None else dict(plan, rows={})
            if plan is not None:
                _append(_state["current"])
        record = _state["current"]
        if record is None:
            return
        hidden, residual = output                    # decoder layer -> (hidden, residual)
        stream = hidden + residual if residual is not None else hidden
        record["rows"][layer] = (stream if record["index"] is None else stream.index_select(0, record["index"])).detach().clone()
        if layer == max(layers):
            record["event"] = torch.cuda.Event()
            record["event"].record()
    return hook


def _append(record):
    with _lock:
        _records[_state["next_id"]] = record
        _state["next_id"] += 1
        while len(_records) > RING:
            _records.popitem(last=False)


def _patch_graph_runner():
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    capture, replay = CudaGraphRunner.capture, CudaGraphRunner.replay

    def capture_with_buffers(self, *args, **kwargs):
        _state["capturing"] = True
        try:
            return capture(self, *args, **kwargs)
        finally:
            _state["capturing"] = False

    def replay_with_capture(self, forward_batch, *args, **kwargs):
        if not _state.get("recaptured"):             # graphs were captured before the hooks existed
            _state["recaptured"] = True
            import sglang.srt.model_executor.cuda_graph_runner as runner_module
            torch.cuda.synchronize()
            with runner_module.model_capture_mode():   # as sglang's own capture at start-up
                capture_with_buffers(self)
            torch.cuda.synchronize()
        out = replay(self, forward_batch, *args, **kwargs)
        if forward_batch.forward_mode.is_decode() and all((self.bs, l) in _static for l in _state["layers"]):
            n = self.raw_bs
            pool = forward_batch.req_pool_indices
            pool_cpu = torch.empty(pool.shape, dtype=pool.dtype, pin_memory=True)
            pool_cpu.copy_(pool, non_blocking=True)
            record = dict(pool_cpu=pool_cpu, req_of=None, index=None,
                          positions=[int(s) - 1 for s in forward_batch.seq_lens_cpu.tolist()[:n]],
                          rows={l: _static[(self.bs, l)][:n].clone() for l in _state["layers"]})
            record["event"] = torch.cuda.Event()
            record["event"].record()
            _append(record)
        return out

    CudaGraphRunner.capture = capture_with_buffers
    CudaGraphRunner.replay = replay_with_capture


def _refresh_index(block=False):
    """Index completed records in forward order (newest wins); stops at the first record not yet complete,
    or waits for it when block=True."""
    with _lock:
        pending = [(i, r) for i, r in _records.items() if i > _state["indexed"]]
    for i, record in pending:
        event = record.get("event")
        if event is None or not (event.query() or (block and event.synchronize() is None)):
            break
        pool = record["pool_cpu"].tolist()
        slots = pool if record["req_of"] is None else [pool[k] for k in record["req_of"]]
        for k, (s, p) in enumerate(zip(slots, record["positions"])):
            _index.setdefault(s, {})[p] = (i, k)
        _state["indexed"] = i


def _gather(slot, positions):
    layers, found = _state["layers"], {}
    _refresh_index()
    entries = _index.get(slot, {})
    if not all(p in entries for p in positions):     # rare: a needed forward not yet complete
        _refresh_index(block=True)
        entries = _index.get(slot, {})
    for p in positions:
        hit = entries.pop(p, None)
        record = _records.get(hit[0]) if hit else None
        if record is not None:
            found[p] = [record["rows"][l][hit[1]] for l in layers]
    return found


def _export(req):
    """Write the captured states of one finished request; positions are absolute token positions.

    Generation requests: the last prompt token and every generated token that was fed back.
    Follow-up requests (rid ending "-tail"): whichever of the final TAIL prompt positions were computed here.
    """
    n = len(req.origin_input_ids)
    tail = req.rid.endswith("-tail")
    fed = len(req.output_ids) if getattr(req, "_actorhead_deferred", False) else len(req.output_ids) - 1
    positions = range(max(0, n - TAIL), n) if tail else range(n - 1, n + fed)
    rows = _gather(req.req_pool_idx, positions)
    complete = n - 1 in rows if tail else len(rows) == len(positions)
    payload = dict(rid=req.rid, layers=_state["layers"], complete=complete, input_len=n, output_ids=list(req.output_ids),
                   final_fed=getattr(req, "_actorhead_deferred", False),
                   input_tail_ids=list(req.origin_input_ids[-TAIL:]), positions=sorted(rows))
    if rows:
        if _state["side"] is None:
            _state["side"] = torch.cuda.Stream()
        side = _state["side"]
        with torch.cuda.stream(side):                # the rows' forwards are complete; wait for nothing else
            stacked = torch.stack([torch.stack(rows[p]) for p in sorted(rows)])
            host = torch.empty(stacked.shape, dtype=stacked.dtype, pin_memory=True)
            host.copy_(stacked, non_blocking=True)
        side.synchronize()
        payload["hidden"] = host                     # [rows, L, H]
    out = Path(os.environ.get("ACTORHEAD_DIR", "/dev/shm/actorhead"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f".{req.rid}.pt"
    torch.save(payload, tmp)
    os.replace(tmp, out / f"{req.rid}.pt")


def _patch_scheduler():
    if _state["patched"]:
        return
    from sglang.srt.managers.scheduler import Scheduler
    original = Scheduler.stream_output

    def stream_output(self, reqs, *args, **kwargs):
        for req in reqs:
            if req.rid.startswith(PREFIX) and req.finished() and not getattr(req, "_actorhead_exported", False):
                req._actorhead_exported = True
                if getattr(req, "_actorhead_deferred", False):   # drop the extra token sampled after <|im_end|>
                    keep = len(req.output_ids) - 1
                    del req.output_ids[keep:]
                    for name in ("output_token_logprobs_val", "output_token_logprobs_idx", "output_top_logprobs_val",
                                 "output_top_logprobs_idx", "output_token_ids_logprobs_val", "output_token_ids_logprobs_idx"):
                        values = getattr(req, name, None)
                        if isinstance(values, list) and len(values) > keep:
                            del values[keep:]
                _export(req)
        return original(self, reqs, *args, **kwargs)

    Scheduler.stream_output = stream_output
    _patch_graph_runner()

    # This sglang version inserts `req.fill_ids` into the radix cache at finish, and decoding never extends it, so a
    # candidate's generated tokens are not cached and its follow-up would re-prefill the whole candidate. For capture
    # requests, cache every generated token whose KV was computed (kv_committed_len), so the follow-up computes only
    # the final token. Sampling and outputs are unchanged.
    from sglang.srt.mem_cache.radix_cache import RadixCache
    cache_finished = RadixCache.cache_finished_req

    def cache_finished_req(self, req, *args, **kwargs):
        if req.rid.startswith(PREFIX) and not req.rid.endswith("-tail") and not self.disable:
            req.fill_ids = (req.origin_input_ids + req.output_ids)[:req.kv_committed_len]
        return cache_finished(self, req, *args, **kwargs)

    RadixCache.cache_finished_req = cache_finished_req

    from sglang.srt.managers.schedule_batch import Req
    check_finished = Req.check_finished

    def check_finished_after_feeding_end(self, *args, **kwargs):
        check_finished(self, *args, **kwargs)
        if (self.finished_reason is not None and not getattr(self, "_actorhead_deferred", False)
                and self.rid.startswith(PREFIX) and not self.rid.endswith("-tail")
                and len(self.output_ids) < self.sampling_params.max_new_tokens
                and type(self.finished_reason).__name__ in ("FINISH_MATCHED_TOKEN", "FINISH_MATCHED_STR")):
            # finish one decode step later so the end token is fed through the model and its hidden state captured
            self.to_finish, self.finished_reason, self._actorhead_deferred = self.finished_reason, None, True

    Req.check_finished = check_finished_after_feeding_end
    _state["patched"] = True
