#!/usr/bin/env python3
"""Head-executed on-policy pairs (user 2026-10-29): copy of iterdpo_r1/prepare_c2pass2_pairs.py run on
head_onpolicy/sbatch_collect_head.sh output (Zixian's selection head picks and executes, mode actor_cache_head). Only
changes: the record's mode must be actor_cache_head and the label is the head's argmax over its 5 logits (no
selector_raw), checked against the executed index; label_source text. Original description follows.

C2 pass-2 pairs (OUT of sbatch_collect_c2pass2.sh: OpenWebRL-4B-SFT + our SelectionARM only, seed 43, terra judge).

Copy of prepare_round1_pairs.py; the only changes: the labeller is always the ARM (pass 2 has no mixed selector, so
selector_raw must be exactly {"selection": <int>}), label_source names the pass, the tokenizer is 4B-SFT (= C2's actor;
template identical to pi1's). As in round 1, Zixian's prepare_arm_joint_data.py (@fe65405) with two changes:
  1. Reads EVERY executed turn of every attempt, not only C2's success-only training.jsonl;
     each state is tagged terminal_success = True / False / None (None = invalid episode).
     Filter on that tag at training time; nothing outcome-based is dropped here.
  2. No Piotr/HF phase. Labeller (arm | gpt-5.5) is recorded from selector_raw.
Pair selection, validation, dedupe, task-group split and audit are her code, unchanged.

Usage (login node, single core):
  PYTHONPATH=<her joint repo>/scripts python prepare_round1_pairs.py \
      --collection /gpfs/scrubbed/dan29/iterdpo_r1/pilot20 --output <new dir>
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
from urllib.parse import urlsplit

import prepare_arm_joint_data as base
from prepare_arm_joint_data import Builder, schemas, sha, task_text

# PACKAGE (2026-11-01): paths from env (README.md). We ran with COLLECTOR_REPO = 1065034 + the c2pass2 patch; the head
# worktree (1065034 + arm_c2_head.patch) has the same openwebrl/base/utils.py parser and eval file, so it can stand in.
JOINT_REPO = Path(os.environ['JOINT_REPO'])  # zixianma/OpenWebRL @fe65405 (scripts/prepare_arm_joint_data.py)
COLLECTOR_REPO = Path(os.environ['WT'])
SFT4B = Path(os.environ['BASE'])


def labeller(c):  # head: 5 finite logits, executed index = argmax (ties to the lowest index)
    s = c.get('scores')
    if c.get('selector_raw') is not None or not isinstance(s, list) or len(s) != len(c['raw_candidates']) or not all(math.isfinite(x) for x in s):
        raise ValueError('not a head record')
    if c['selected_index'] != max(range(len(s)), key=lambda j: (s[j], -j)): raise ValueError('executed index is not the head argmax')
    return 'head'


class Pass2Builder(Builder):
    def c2(self):
        root = self.a.collection
        config = json.loads((root / 'frozen-config.json').read_text())
        n_tasks = config['task_count']
        results = sorted((root / 'results').glob('*.json'))
        if len(results) != n_tasks and not self.a.allow_incomplete:
            raise ValueError(f'collection incomplete: {len(results)}/{n_tasks} results')
        h = hashlib.sha256(); count = 0
        for result_path in results:
            result_bytes = result_path.read_bytes(); h.update(result_bytes)
            outcome = json.loads(result_bytes)
            success = None if not outcome.get('valid') else outcome.get('reward') == 1
            self.stats['C2']['episodes_' + str(success)] += 1
            ep = Path(outcome['attempt'])
            execution = json.loads((ep / 'execution.json').read_text())
            if execution['task_id'] != outcome['task_id']: raise ValueError('cross-task execution join')
            for t in execution['turns']:
                stub = {'id': 'C2:' + outcome['task_id'] + ':' + str(t['turn']), 'source': 'C2'}
                self.stats['C2']['source_states'] += 1; count += 1
                try:
                    if not t['executed']: raise ValueError('not_executed')
                    raw = Path(t['source']).read_bytes()
                    if sha(raw) != t['source_sha256']: raise ValueError('source_hash')
                    c = json.loads(raw)
                    if (c['task_id'], c['turn'], c['attempt']) != (outcome['task_id'], t['turn'], outcome['attempt']):
                        raise ValueError('source_join')
                    if sha(c['prompt'].encode()) != c['prompt_sha256']: raise ValueError('prompt_hash')
                    if c.get('fallback') or c['mode'] != 'actor_cache_head': raise ValueError('selector_fallback')
                    images = [self.image_info(x['path'], x['sha256']) for x in c['images']]
                    context = self.context(c['prompt'], images, t['prompt_token_ids'], t['image_grid_thw'])
                    who = labeller(c)
                    self.stats['C2']['labeller_' + who] += 1
                    state = {**stub, **context, 'base_id': stub['id'], 'episode_id': 'C2:' + outcome['task_id'],
                             'task': task_text(c['prompt']), 'host': urlsplit(c['url']).hostname or '',
                             'url': c['url'], 'turn': t['turn'], 'prompt': c['prompt'], 'images': images,
                             'source_path': t['source'], 'source_sha256': t['source_sha256'],
                             'label_source': f'SelectionHead {HEAD_NAME} (actor_cache_head); executed in head-collection trajectory (gpt-5.6-terra judge)',
                             'labeller': who, 'terminal_success': success,
                             'terminate_reason': outcome.get('terminate_reason')}
                    candidates = [{'text': x['response'], 'finish': x['finish_type'], 'ids': x['response_token_ids']}
                                  for x in c['raw_candidates']]
                    self.accept_draw(state, candidates, c['selected_index'], schemas(c['prompt']))
                except (ValueError, TypeError, KeyError, OSError) as error:
                    self.event(stub, 'state_integrity_failure', detail=str(error))
                if count % 500 == 0:
                    self.db.commit(); print(f'C2 {count} states', flush=True)
                time.sleep(.005)
        self.c2_hash = h.hexdigest(); self.db.commit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true', help='for a pilot / partial run only')
    p.add_argument('--repo', type=Path, default=COLLECTOR_REPO)
    p.add_argument('--model', type=Path, default=SFT4B)
    p.add_argument('--eval', type=Path, default=COLLECTOR_REPO / 'openwebrl/data/eval/online-mind2web.jsonl')
    p.add_argument('--cpu-seconds', type=int, default=240)
    a = p.parse_args()
    a.resume = False
    global HEAD_NAME  # which head executed (from the collection's selector health; user 2026-10-30 adds lookahead-k3-1380)
    sha = json.loads((a.collection / 'frozen-config.json').read_text())['teacher_manifest']['selector_health']['head_sha256']
    HEAD_NAME = {'755d4cd4913765e2': 'piotr-data', '11d5eff8ffe1966c': 'lookahead-k3-1380'}[sha[:16]]
    print('head', HEAD_NAME, sha, flush=True)
    assert Path(base.__file__).resolve() == (JOINT_REPO / 'scripts/prepare_arm_joint_data.py').resolve(), base.__file__
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (a.cpu_seconds, a.cpu_seconds + 5))
    b = Pass2Builder(a)
    b.c2(); b.finish()


if __name__ == '__main__': main()
