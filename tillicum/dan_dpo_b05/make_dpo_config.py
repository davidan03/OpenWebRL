#!/usr/bin/env python3
"""Frozen config for train_dpo_r1_ddp.py: Zixian's joint-v2 DPO config (runs/joint-v2-dpo-2gpu-r2/predecessor-config.json)
with only the data, actor, output, sizes and save/validation points changed. Same LoRA r16/a32/0.05, beta 0.1, lr 1e-5,
32 pairs/update, 1 epoch, warmup 512 examples, full-response sum score, no auxiliary SFT, reference = actor with
adapters disabled (= pi0 here). Stdlib only (login node); streams the data files.
Usage: PYTHONPATH=<her worktree>/scripts python3 make_dpo_config.py --data DIR --actor DIR --output RUN_DIR --config OUT.json
       [--lora-rank R --lora-alpha A --learning-rate LR --wandb-group G]   (2026-10-20: rank / lr arms; defaults = hers;
       read by train_dpo_r1_ddp.py edit 5)
"""
import argparse, hashlib, json, math
from pathlib import Path
from train_arm_joint_sft import digest, encoded, training_order, REPO

HERS = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/runs/joint-v2-dpo-2gpu-r2/predecessor-config.json')
ap = argparse.ArgumentParser()
for k in ('data', 'actor', 'output', 'config'): ap.add_argument(f'--{k}', type=Path, required=True)
ap.add_argument('--lora-rank', type=int); ap.add_argument('--lora-alpha', type=int)
ap.add_argument('--learning-rate', type=float); ap.add_argument('--wandb-group')
ap.add_argument('--beta', type=float)  # 2026-10-23 DPO beta sweep (default = hers, 0.1)
ap.add_argument('--sft-weight', type=float, default=0.0)  # 2026-10-22 RPO ablation: train_dpo_r1_ddp.py edit 6 (0 = plain DPO)
a = ap.parse_args()
c = json.loads(HERS.read_text())
rows = [dict(id=d['id'], source=d['source']) for d in map(json.loads, open(a.data / 'joint_pairs.train.jsonl'))]
# panels of every source present, as train_dpo_r1_ddp.py loads them (round 1 = C2 only; joint data = C2 + Piotr)
panel_files = [f'validation-panel-{s.lower()}.jsonl' for s in sorted({r['source'] for r in rows})]
val = sum(sum(1 for _ in open(a.data / f)) for f in panel_files)
order = training_order(rows, c['seed'])
n = len(rows); updates = math.ceil(n / c['batch_size'])
q = [round(updates * f) for f in (.25, .5, .75)] + [updates]  # her 44/87/131/174 = quarters of 174
for name in c['model_metadata_hashes']:
    assert digest(a.actor / name) == c['model_metadata_hashes'][name], f'actor differs from her OpenWebRL-4B-SFT: {name}'
c.update(
    actor=str(a.actor), data=str(a.data), output=str(a.output),
    train_examples=n, validation_examples=val, total_updates=updates, final_batch_size=n - (updates - 1) * c['batch_size'],
    checkpoint_updates=q, validation_updates=[0] + q,
    data_hashes={f: digest(a.data / f) for f in ['joint_pairs.train.jsonl'] + panel_files},
    order_sha256=hashlib.sha256(encoded([rows[i]['id'] for i in order]).encode()).hexdigest(),
    code_hashes={f: digest(REPO / 'scripts' / f) for f in ('train_arm_joint_sft.py', 'train_arm_c2.py')},
    trainer_copy_sha256=digest(Path(__file__).with_name('train_dpo_r1_ddp.py')),
    reference='OpenWebRL-4B-SFT (pi0) with adapters disabled', wandb_group='iterdpo-r1',
    derived_from=str(HERS))
for k in ('lora_rank', 'lora_alpha', 'learning_rate', 'wandb_group', 'beta'):
    if getattr(a, k) is not None:
        c[k] = getattr(a, k)
assert c['final_batch_size'] % c['world_size'] == 0 and c['objective'] == 'dpo' and c['auxiliary_sft_weight'] == 0
c['auxiliary_sft_weight'] = a.sft_weight
a.config.parent.mkdir(parents=True, exist_ok=True)
assert not a.config.exists(), a.config
a.config.write_text(json.dumps(c, indent=2))
print(json.dumps({k: c[k] for k in ('train_examples', 'validation_examples', 'total_updates', 'final_batch_size',
      'checkpoint_updates', 'validation_updates', 'learning_rate', 'beta', 'auxiliary_sft_weight', 'warmup_examples', 'lora_rank', 'lora_alpha', 'lora_dropout', 'order_sha256')}, indent=1))
