#!/usr/bin/env python3
"""Zixian's JOINT DPO corpus (klone-mirror copy, 5,540 train pairs) filtered to pairs whose chosen and rejected responses
hold the SAME number of tool calls (user go 2026-10-21; arm F1 = the control's LoRA recipe unchanged, data only).
Why: count-mismatched pairs (1,076 chosen-more : 385 chosen-fewer) teach "another <tool_call> after </tool_call>", the
channel that broke the r64 / lr3e-5 dose arms; the filter keeps stop-vs-continue pairs at the same 455:755 ratio.
Train rows are copied byte-for-byte in file order. The validation panels (her 658 held-out pairs, all kinds) are copied
unchanged so held-out metrics stay comparable with the control. images/ is a symlink (rows use relative image paths).
make_dpo_config.py asserts final_batch_size % world_size == 0 (32 pairs/update, 2 ranks): if the kept count leaves an
odd final batch, the LAST kept row in file order is dropped and recorded in filter-audit.json.
Usage (login node): python3 filter_joint_eqcount.py --data <joint data dir> --out <new dir>
"""
import argparse, collections, hashlib, json, os, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--data', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--drop-for-batch', help='id to drop for an even final batch instead of the last kept row '
                '(2026-10-22: Piotr:ep9__t6 reproduces Piotr\'s HF openwebrl-arm-matched-pairs @5c91573d exactly)')
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
c = collections.Counter(); kept = []
for line in open(a.data / 'joint_pairs.train.jsonl', 'rb'):
    d = json.loads(line); c['in'] += 1
    assert d['status'] == 'retained', d['id']
    nc, nr = len(d['chosen']['actions']), len(d['rejected']['actions'])
    assert nc >= 1 and nr >= 1, d['id']
    if nc != nr: c['drop_call_count_mismatch'] += 1; continue
    kept.append((d['id'], line))
BATCH, WORLD = 32, 2
final = len(kept) - (-(-len(kept) // BATCH) - 1) * BATCH
dropped_for_batch = None
if final % WORLD:
    i = len(kept) - 1 if a.drop_for_batch is None else [k for k, _ in kept].index(a.drop_for_batch)
    dropped_for_batch = kept.pop(i)[0]
h = hashlib.sha256()
with open(a.out / 'joint_pairs.train.jsonl', 'wb') as g:
    for _, line in kept:
        g.write(line); h.update(line)
for f in ('validation-panel-c2.jsonl', 'validation-panel-piotr.jsonl'):
    shutil.copyfile(a.data / f, a.out / f)
os.symlink((a.data / 'images').resolve(), a.out / 'images')
# every kept row re-read and checked: equal counts, no duplicate ids, train/validation task groups disjoint
ids, groups, stop = set(), set(), collections.Counter()
for line in open(a.out / 'joint_pairs.train.jsonl'):
    d = json.loads(line); cn = [x['name'] for x in d['chosen']['actions']]; rn = [x['name'] for x in d['rejected']['actions']]
    assert len(cn) == len(rn) and d['id'] not in ids; ids.add(d['id']); groups.add(d['task_group'])
    c[f"source_{d['source']}"] += 1; c[f"type_{d['pair_type']}"] += 1
    if 'done' in cn and 'done' not in rn: stop['stop_chosen'] += 1
    if 'done' in rn and 'done' not in cn: stop['continue_chosen'] += 1
val_groups = {json.loads(l)['task_group'] for f in ('validation-panel-c2.jsonl', 'validation-panel-piotr.jsonl') for l in open(a.out / f)}
assert not groups & val_groups, 'train/validation task overlap'
n = len(ids); final = n - (-(-n // BATCH) - 1) * BATCH
assert final % WORLD == 0 and c['in'] == n + c['drop_call_count_mismatch'] + (dropped_for_batch is not None)
audit = dict(source=str(a.data / 'joint_pairs.train.jsonl'), sha256=h.hexdigest(), kept=n, updates=-(-n // BATCH),
             final_batch=final, dropped_for_even_final_batch=dropped_for_batch, **dict(sorted(c.items())), **dict(stop),
             validation_panels='copied unchanged from --data', images=str((a.data / 'images').resolve()))
json.dump(audit, open(a.out / 'filter-audit.json', 'w'), indent=2)
print(json.dumps(audit, indent=2))
