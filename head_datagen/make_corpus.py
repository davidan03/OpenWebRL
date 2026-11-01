#!/usr/bin/env python3
"""Head-executed on-policy RPO corpus (user 2026-10-29): copy of actor_states_arm/make_corpus.py on prepare_head_pairs.py
output (two passes, seeds 42/43); only the label check and the id prefixes (HD:/HD2:) differ. Original:
Actor-states RPO corpus (user 2026-10-28): prepare_shadow_pairs.py output -> trainer rows, with the filters
merge_c2pass2.py applied to the C2 pass-2 rows of the on-policy corpus (data_onpolicy_c2_c2p2, 5,206 rows):
  - both builder splits are training rows (source tasks = her 2,091 minus the 113 C2 validation groups; asserted),
  - status retained, terminal_success True, equal tool-call counts, prompt + longer response <= 27,869 tokens.
No size-match (user: keep all states). Two passes (user 2026-10-28: second no-ARM pass, seed 43, like C2's two passes):
--pairs PASS1 PASS2; pass-2 states already in pass 1 (same prompt_sha256 + screenshot hashes) are dropped, as
merge_c2pass2.py did. Ids 'AS:<task>:<turn>' (pass 1) / 'AS2:<task>:<turn>' (pass 2).
Validation panels + images symlink copied from the on-policy corpus, so held-out metrics compare with it.
Final batch made even for 2 ranks by dropping the last row, as onpolicy_only.py does.
Usage (login node): python3 -I make_corpus.py --pairs BUILDER_OUT [BUILDER_OUT_PASS2] --onpolicy DIR --out NEW_DIR
"""
import argparse, collections, hashlib, json, os, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--pairs', type=Path, nargs='+', required=True); ap.add_argument('--onpolicy', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--head-name', default='piotr-data', choices=['piotr-data', 'lookahead-k3-1380'])
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
val_groups = {json.loads(l)['task_group'] for f in ('validation-panel-c2.jsonl', 'validation-panel-piotr.jsonl')
              for l in open(a.onpolicy / f)}
MAX_SEQ = 27869
key = lambda d: (d['prompt_sha256'], tuple(i['sha256'] for i in d['images']))
c = collections.Counter(); rows = []; ids = set(); seen = set()
for p, (prefix, pairs) in enumerate(zip(('HD:', 'HD2:'), a.pairs)):
  for f in ('joint_pairs.train.jsonl', 'joint_pairs.validation.jsonl'):
    for l in open(pairs / f):
        d = json.loads(l); c[f'in_pass{p + 1}'] += 1
        assert d['source'] == 'C2' and d['label_source'].startswith(f'SelectionHead {a.head_name} (actor_cache_head); executed'), d['id']
        assert d['task_group'] not in val_groups, ('row in a validation task group', d['id'])
        if d['status'] != 'retained': c['drop_not_retained'] += 1; continue
        if d['terminal_success'] is not True: c[f"drop_success_{d['terminal_success']}"] += 1; continue
        nc, nr = len(d['chosen']['actions']), len(d['rejected']['actions'])
        assert nc >= 1 and nr >= 1, d['id']
        if nc != nr: c['drop_call_count_mismatch'] += 1; continue
        if d['prompt_tokens'] + max(len(d['chosen']['token_ids']), len(d['rejected']['token_ids'])) > MAX_SEQ:
            c['drop_longer_than_proven'] += 1; continue
        if key(d) in seen: c[f'drop_pass{p + 1}_state_in_earlier_pass'] += 1; continue
        seen.add(key(d))
        d['id'] = prefix + d['id'].split(':', 1)[1]
        assert d['id'] not in ids, d['id']; ids.add(d['id'])
        rows.append(d)
BATCH, WORLD = 32, 2
n = len(rows); final = n - (-(-n // BATCH) - 1) * BATCH
dropped = rows.pop()['id'] if final % WORLD else None
h = hashlib.sha256()
with open(a.out / 'joint_pairs.train.jsonl', 'wb') as g:
    for d in rows:
        l = (json.dumps(d, ensure_ascii=False) + '\n').encode(); g.write(l); h.update(l)
for f in ('validation-panel-c2.jsonl', 'validation-panel-piotr.jsonl'):
    shutil.copyfile(a.onpolicy / f, a.out / f)
os.symlink((a.onpolicy / 'images').resolve(), a.out / 'images')
# re-read everything written: unique ids, equal call counts, every screenshot present with its recorded hash
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
checked = {}; n = 0; stop = collections.Counter(); picked = collections.Counter(); seen = set(); by_pass = collections.Counter()
for l in open(a.out / 'joint_pairs.train.jsonl'):
    d = json.loads(l); n += 1; assert d['id'] not in seen; seen.add(d['id']); by_pass[d['id'].split(':', 1)[0]] += 1
    cn = [x['name'] for x in d['chosen']['actions']]; rn = [x['name'] for x in d['rejected']['actions']]
    assert len(cn) == len(rn)
    if 'done' in cn and 'done' not in rn: stop['stop_chosen'] += 1
    if 'done' in rn and 'done' not in cn: stop['continue_chosen'] += 1
    picked[f"chosen_index_{d['chosen_index']}"] += 1
    for i in d['images']:
        p = Path(i['path']); p = p if p.is_absolute() else a.out / p
        if p not in checked: checked[p] = digest(p)
        assert checked[p] == i['sha256'], ('screenshot hash mismatch', d['id'], p)
final = n - (-(-n // BATCH) - 1) * BATCH; assert final % WORLD == 0 and n == len(rows)
audit = dict(pairs=[str(x) for x in a.pairs], onpolicy=str(a.onpolicy), sha256=h.hexdigest(), rows=n, rows_by_pass=dict(by_pass), updates=-(-n // BATCH),
             final_batch=final, dropped_for_even_final_batch=dropped, screenshots_hashed=len(checked),
             **dict(sorted(c.items())), **dict(stop), **dict(sorted(picked.items())),
             validation_panels='copied unchanged from --onpolicy')
json.dump(audit, open(a.out / 'corpus-audit.json', 'w'), indent=2)
print(json.dumps(audit, indent=2))
