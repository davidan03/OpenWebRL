"""Exhaustive integrity checks of the head-executed data path.
(A) every head turn: 5 finite scores, no fallback, executed index == argmax (ties -> lowest), executed text == candidate.
(B) every corpus row: source turn exists, chosen_index == executed index, chosen/rejected text == that turn's raw candidate
    responses, chosen and rejected actions differ, episode valid + reward 1, label source = head."""
import json, glob, math, collections, sys
C = collections.Counter(); bad = collections.defaultdict(list)
turn_cache = {}
ARGS = sys.argv[1:] or ['/gpfs/scrubbed/dan29/head_onpolicy/full1978_s42', '/gpfs/scrubbed/dan29/head_onpolicy/full1978_s43',
        '/gpfs/scrubbed/dan29/head_onpolicy/data_head_2pass']  # usage: integrity.py [coll_s42 coll_s43 corpus]
for coll in ARGS[:2]:
    for f in glob.glob(coll + '/attempts/*/attempt-0001/turn-*.json'):
        d = json.load(open(f)); C['turns'] += 1
        sc = d.get('scores')
        if d.get('mode') != 'actor_cache_head': bad['mode'].append(f); continue
        if not (isinstance(sc, list) and len(sc) == 5 and all(math.isfinite(x) for x in sc)): bad['scores'].append(f); continue
        if d.get('fallback'): bad['fallback'].append(f)
        am = max(range(5), key=lambda i: (sc[i], -i))
        if am != int(d['selected_index']): bad['not_argmax'].append((f, sc, d['selected_index']))
        if len(d['raw_candidates']) != 5: bad['ncand'].append(f)
        turn_cache[f] = (int(d['selected_index']), [c['response'] for c in d['raw_candidates']])
print('turns', C['turns'], {k: len(v) for k, v in bad.items()})
for k, v in bad.items(): print('  e.g.', k, str(v[:2])[:300])
R = collections.Counter(); rb = collections.defaultdict(list)
outcome = {}
for line in open(ARGS[2] + '/joint_pairs.train.jsonl'):
    d = json.loads(line); R['rows'] += 1; sp = d['source_path']
    if sp not in turn_cache: rb['no_source'].append(d['id']); continue
    sel, cands = turn_cache[sp]
    if int(d['chosen_index']) != sel: rb['chosen_not_executed'].append(d['id'])
    if d['chosen']['text'] != cands[int(d['chosen_index'])]: rb['chosen_text'].append(d['id'])
    if d['rejected']['text'] != cands[int(d['rejected_index'])]: rb['rejected_text'].append(d['id'])
    if d['chosen'].get('actions') == d['rejected'].get('actions'): rb['same_actions'].append(d['id'])
    if d.get('labeller') != 'head': rb['labeller'].append(d['id'])
    att = sp.rsplit('/', 1)[0]
    if att not in outcome: outcome[att] = json.load(open(att + '/outcome.json'))
    o = outcome[att]
    if not (str(o.get('valid')) == 'True' and float(o.get('reward') or 0) == 1.0): rb['episode_not_success'].append(d['id'])
print('corpus rows', R['rows'], {k: len(v) for k, v in rb.items()})
for k, v in rb.items(): print('  e.g.', k, v[:3])
