#!/usr/bin/env python3
"""Live-vs-offline parity for the head-executed collection (pilot gate): re-score N saved turns with Zixian's offline
SelectionHeadScorer (openwebrl_selection_head.py from zixianma/action-selection-heads @97252f71, unchanged; it
re-encodes prompt + screenshot + candidate token ids through the HF actor) and compare with the live picks, which came
from hidden states captured inside the actor's sglang server. Her card: 34-37/40 same pick, the rest near-ties.
Usage (GPU, actor server stopped): python parity_check.py --collection OUT --heads DIR --actor DIR [--n 40]
"""
import argparse, glob, importlib.util, json, random, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--collection', type=Path, required=True); p.add_argument('--heads', type=Path, required=True)
p.add_argument('--actor', required=True); p.add_argument('--n', type=int, default=40)
p.add_argument('--head-name', default='piotr-data', choices=['piotr-data', 'lookahead-k3-1380'])
p.add_argument('--batch-size', type=int, default=5)  # 1 on an 18 GB MIG slice (her card: same scores, one candidate at a time)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('osh', a.heads / 'openwebrl_selection_head.py')
osh = importlib.util.module_from_spec(spec); sys.modules['osh'] = osh; spec.loader.exec_module(osh)  # dataclasses need the module registered
from PIL import Image
turns = []
for f in sorted(glob.glob(str(a.collection / 'attempts/*/attempt-*/turn-*.json'))):
    c = json.loads(Path(f).read_text())
    if c['mode'] == 'actor_cache_head' and len(c['raw_candidates']) == 5 and c.get('scores'):
        turns.append(c)
random.Random(0).shuffle(turns)
turns = turns[:a.n]
scorer = osh.SelectionHeadScorer(a.heads / a.head_name, actor=a.actor, revision=None, device='cuda:0', batch_size=a.batch_size)
same, gaps = 0, []
for c in turns:
    img = Image.open(c['images'][0]['path'])
    off = scorer.score(c['prompt'], img, [x['response_token_ids'] for x in c['raw_candidates']])
    pick = max(range(5), key=lambda i: (off[i], -i))
    live = c['scores']; same += pick == c['selected_index']
    srt = sorted(live, reverse=True); gaps.append(srt[0] - srt[1])
    print(json.dumps(dict(task=c['task_id'], turn=c['turn'], live_pick=c['selected_index'], offline_pick=pick,
                          live=[round(x, 3) for x in live], offline=[round(x, 3) for x in off])), flush=True)
print(f'PARITY same pick {same}/{len(turns)}; live top-2 gap median {sorted(gaps)[len(gaps) // 2]:.3f}', flush=True)
