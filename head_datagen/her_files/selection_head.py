"""Candidate-set selection head on frozen actor features (shared actor-Q idea).

The actor's own hidden states summarize the state prefix and each sampled
response. A small permutation-equivariant transformer compares the N candidates
jointly (listwise, like SelectionARM) and returns one selection logit per
candidate plus an auxiliary absolute-value logit. There are no position
embeddings, so candidate order cannot bias the choice.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Pools computed per selected layer. The state prefix contributes one vector
# per layer ("prefix_last"); candidate pools are concatenated per layer.
CANDIDATE_POOLS = ("thought_mean", "action_mean", "last")
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
POINT_KEYS = ("point_2d", "start_point_2d", "end_point_2d")


@dataclass
class HeadConfig:
    hidden_size: int
    layers: tuple[int, ...]
    pools: tuple[str, ...] = CANDIDATE_POOLS
    d_model: int = 512
    n_heads: int = 8
    depth: int = 2
    dropout: float = 0.1
    listwise: bool = True  # False: score each candidate alone (pointwise ablation)
    # Factorial switches (None = derived from `listwise`, so older heads load unchanged):
    # interaction "joint" lets each candidate's score depend on the others, "independent" does not;
    # scorer "transformer" or "mlp"; mlp_depth hidden layers for the MLP scorer.
    interaction: str | None = None
    scorer: str | None = None
    mlp_depth: int = 1
    # MLP scorer hidden width (None = d_model, the original layout); widen it to match the transformer's parameters
    mlp_width: int | None = None
    # readout "linear": one logit per candidate from its own output; "pointer" (joint transformer only): a learned
    # <decide> token attends to every candidate and scores q(decide) . k(candidate), as in Kev's pointer
    readout: str = "linear"

    def resolved(self):
        joint = self.listwise if self.interaction is None else self.interaction == "joint"
        scorer = self.scorer or ("transformer" if self.listwise else "mlp")
        if scorer not in ("transformer", "mlp"):
            raise ValueError(f"unknown scorer {scorer!r}")
        return ("joint" if joint else "independent"), scorer

    @property
    def candidate_dim(self):
        return self.hidden_size * len(self.layers) * len(self.pools)

    @property
    def state_dim(self):
        return self.hidden_size * len(self.layers)

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        data["layers"], data["pools"] = tuple(data["layers"]), tuple(data["pools"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class Standardize(nn.Module):
    """Fixed per-dimension standardization fitted on training features.

    Unlike a per-token LayerNorm, this keeps candidates on a common scale, so
    differences between candidates in the same panel survive the input layer.
    """
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("inv_std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, features, eps=1e-4):
        flat = features.reshape(-1, features.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.inv_std.copy_(flat.std(0).clamp_min(eps).reciprocal())

    def forward(self, x):
        return (x.float() - self.mean) * self.inv_std


def _projector(dim, d_model, dropout):
    return nn.Sequential(Standardize(dim), nn.Linear(dim, d_model), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_model, d_model))


class CandidateSetHead(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.candidate_proj = _projector(config.candidate_dim, d, config.dropout)
        self.state_proj = _projector(config.state_dim, d, config.dropout)
        self.type_embed = nn.Parameter(torch.zeros(2, d))  # 0 = state, 1 = candidate
        nn.init.normal_(self.type_embed, std=0.02)
        self.interaction, self.scorer = config.resolved()
        if self.scorer == "transformer" and config.depth > 0:
            layer = nn.TransformerEncoderLayer(d, config.n_heads, 4 * d, config.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, config.depth, enable_nested_tensor=False)
        else:
            self.encoder = None
            # MLP scorer on [candidate, state] (independent) or [candidate, state, candidate - mean of the others] (joint)
            width, h = (3 if self.interaction == "joint" else 2) * d, config.mlp_width or d
            mlp = [nn.LayerNorm(width), nn.Linear(width, h), nn.GELU(), nn.Dropout(config.dropout)]
            for _ in range(max(config.mlp_depth, 1) - 1):
                mlp += [nn.Linear(h, h), nn.GELU(), nn.Dropout(config.dropout)]
            if h != d:
                mlp.append(nn.Linear(h, d))
            self.pointwise = nn.Sequential(*mlp)
        if config.readout == "pointer":
            if self.encoder is None or self.interaction != "joint":
                raise ValueError("the pointer readout needs the joint transformer scorer")
            self.decide = nn.Parameter(torch.zeros(d))
            nn.init.normal_(self.decide, std=0.02)
            self.pointer_q, self.pointer_k = nn.Linear(d, d), nn.Linear(d, d)
        elif config.readout != "linear":
            raise ValueError(f"unknown readout {config.readout!r}")
        self.out_norm = nn.LayerNorm(d)
        self.select = nn.Linear(d, 1)
        self.value = nn.Linear(d, 1)

    def fit_standardizers(self, candidates, state, mask=None):
        """Fit input statistics on training features (real candidates only)."""
        if mask is not None:
            candidates = candidates[mask]
        self.candidate_proj[0].fit(candidates)
        self.state_proj[0].fit(state)

    def forward(self, candidates, state, mask=None):
        """candidates [B,N,Dc], state [B,Ds], mask [B,N] (True = real candidate).

        Returns selection logits [B,N] (padded slots = -inf) and value logits [B,N].
        """
        batch, count, _ = candidates.shape
        if mask is None:
            mask = torch.ones(batch, count, dtype=torch.bool, device=candidates.device)
        c = self.candidate_proj(candidates) + self.type_embed[1]
        s = self.state_proj(state)[:, None] + self.type_embed[0]
        pointer = self.config.readout == "pointer"
        if self.encoder is not None:
            lead = [s, self.decide.expand(batch, 1, -1)] if pointer else [s]
            tokens = torch.cat(lead + [c], 1)
            padding = torch.cat([torch.zeros_like(mask[:, :1])] * len(lead) + [~mask], 1)
            attn = None
            if self.interaction == "independent":
                # each candidate attends only to the state and itself; the state attends only to itself
                n = count + 1
                attn = ~torch.eye(n, dtype=torch.bool, device=c.device)
                attn[1:, 0] = False
            out = self.encoder(tokens, mask=attn, src_key_padding_mask=padding)
            c = out[:, len(lead):]
            if pointer:
                c = self.out_norm(c)
                q, k = self.pointer_q(self.out_norm(out[:, 1])), self.pointer_k(c)
                logits = (k @ q[..., None]).squeeze(-1) / math.sqrt(k.shape[-1])
                return logits.masked_fill(~mask, float("-inf")), self.value(c).squeeze(-1)
        else:
            parts = [c, s.expand_as(c)]
            if self.interaction == "joint":
                m = mask[..., None].to(c.dtype)
                others = ((c * m).sum(1, keepdim=True) - c * m) / (m.sum(1, keepdim=True) - m).clamp_min(1)
                parts.append(c - others)
            c = self.pointwise(torch.cat(parts, -1))
        c = self.out_norm(c)
        logits = self.select(c).squeeze(-1).masked_fill(~mask, float("-inf"))
        return logits, self.value(c).squeeze(-1)


def selection_loss(logits, target, mask):
    """Soft cross-entropy against a target distribution over real candidates."""
    logp = torch.log_softmax(logits, -1).masked_fill(~mask, 0.)
    return -(target * logp).sum(-1).mean()


def pointwise_bce_loss(logits, target, mask):
    """Per-candidate binary cross-entropy: each candidate's target is its own share of the label (no softmax over the set)."""
    finite = logits.masked_fill(~mask, 0.)
    loss = F.binary_cross_entropy_with_logits(finite, target.clamp(0, 1), reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def value_loss(value_logits, successes, trials, mask):
    """Binomial NLL for observed continuation counts; slots with no trials are ignored."""
    weight = trials * mask
    if not weight.any():
        return value_logits.sum() * 0.
    rate = torch.where(trials > 0, successes / trials.clamp_min(1), torch.zeros_like(trials))
    loss = F.binary_cross_entropy_with_logits(value_logits, rate, weight=weight, reduction="sum")
    return loss / weight.sum()


# ---------------------------------------------------------------- action keys

def parse_calls(response):
    """Executable tool calls after the thinking block, or None if unparseable."""
    text = response.rsplit("</think>", 1)[-1].replace("<|im_end|>", "")
    blocks = TOOL_CALL.findall(text)
    if not blocks or len(blocks) != text.count("<tool_call>"):
        return None
    try:
        calls = [json.loads(b) for b in blocks]
    except json.JSONDecodeError:
        return None
    if any(not isinstance(c, dict) or not isinstance(c.get("name"), str)
           or not isinstance(c.get("arguments", {}), dict) for c in calls):
        return None
    return [{"name": c["name"], "arguments": c.get("arguments", {})} for c in calls]


def equivalent(left, right, tolerance=5.):
    """Same calls, allowing point coordinates within `tolerance` normalized units."""
    if left is None or right is None or len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a["name"] != b["name"] or a["arguments"].keys() != b["arguments"].keys():
            return False
        for key, value in a["arguments"].items():
            other = b["arguments"][key]
            if key in POINT_KEYS:
                try:
                    if math.dist(value, other) > tolerance:
                        return False
                except (TypeError, ValueError):
                    return False
            elif value != other:
                return False
    return True


def equivalence_target(calls, selected, tolerance=5.):
    """Spread the teacher's choice uniformly over action-equivalent candidates."""
    chosen = calls[selected]
    hits = [i == selected or equivalent(chosen, c, tolerance) for i, c in enumerate(calls)]
    total = sum(hits)
    return [h / total for h in hits]


def topk_agreement(logits, target):
    """Fraction of panels where the argmax lands on a target-supported candidate."""
    picked = logits.argmax(-1)
    return (target.gather(-1, picked[:, None]).squeeze(-1) > 0).float().mean().item()
