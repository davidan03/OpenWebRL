"""Score paired WebVoyager / DeepShop runs from arm_eval results dirs (one results/*.json per task).

  python bench_n3.py --bench webvoyager --base <dir_r1> <dir_r2> ... --arm <dir_r1> <dir_r2> ... [--csv out.csv]
  python bench_n3.py --bench webvoyager --sites <dir>        # per-site successes + first-page titles of one run

Each dir is <pass>/<label>/<bench> (contains results/). Base and arm dirs are paired by position (same eval job).
Metric of record = successes / all scheduled tasks (= summary.json success_rate_all_scheduled; invalid = failure).
  full   : all tasks.
  clean  : tasks with valid=True in every run of both arms (valid = arm_eval's own flag: finished + scored).
  noblock: tasks with no block page in any run of both arms. A block page is read from the observations only
           (active-tab URL + title), never from the agent's wording: Cloudflare/CloudFront/403/429/Access Denied/
           CAPTCHA titles, Google /sorry/, Amazon /checkpoint/ and /ap/ (forced sign-in).
Stats: per-task mean over runs, arm - base; 95% CI = task bootstrap (10k); p = sign-flip permutation (200k).
Loads one result file at a time (files are 5-15 MB)."""
import argparse, collections, csv, glob, json, os, random, re

TAB = re.compile(r'- Tab \d+ \(active\): (\S+) - ([^\n|]*)')
BLOCK_TITLE = re.compile(r'just a moment|attention required|access denied|captcha|are you a robot|security check|'
                         r'verify you are|unusual traffic|forbidden|\b403\b|\b429\b|pardon our interruption|robot check|'
                         r'human verification|request could not be satisfied', re.I)
BLOCK_URL = re.compile(r'google\.com/sorry|amazon\.com/checkpoint/|amazon\.com/ap/', re.I)


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, 'results', '*.json'))):
        r = json.load(open(f)); trail = []
        for m in (r.get('metadata') or {}).get('messages', []):
            if m['role'] == 'user' and isinstance(m['content'], list):
                for x in m['content']: trail += TAB.findall(x.get('text') or '')
        valid = str(r.get('valid')) == 'True'
        out[r['task_id']] = dict(ok=valid and (r.get('reward') or 0) >= 1, valid=valid,
                                 block=any(BLOCK_TITLE.search(t) or BLOCK_URL.search(u) for u, t in trail),
                                 first=trail[0][1][:50] if trail else '')
        del r
    return out


def stats(B, A, ts):
    d = [sum(x[t]['ok'] for x in A) / len(A) - sum(x[t]['ok'] for x in B) / len(B) for t in ts]
    n = len(d); m = sum(d) / n
    rng = random.Random(0); bs = sorted(sum(rng.choice(d) for _ in d) / n for _ in range(10000))
    rng = random.Random(1); obs = abs(sum(d)); K = 200000
    h = sum(abs(sum(x if rng.random() < .5 else -x for x in d)) >= obs - 1e-12 for _ in range(K))
    return f'{100*m:+.2f} pp [{100*bs[250]:+.1f}, {100*bs[9750]:+.1f}], p {(h+1)/(K+1):.4f} (n {n})'


ap = argparse.ArgumentParser()
ap.add_argument('--bench', required=True); ap.add_argument('--base', nargs='*'); ap.add_argument('--arm', nargs='*')
ap.add_argument('--sites'); ap.add_argument('--csv')
a = ap.parse_args()
if a.sites:
    R = load(a.sites); by = collections.defaultdict(list)
    for t, v in R.items(): by[t.split('--')[0]].append(v)
    print(f'{a.sites}: {sum(v["ok"] for v in R.values())}/{len(R)} successes, {sum(v["block"] for v in R.values())} episodes with a block page')
    for s, vs in sorted(by.items()):
        top = collections.Counter(v['first'] for v in vs).most_common(2)
        print(f'  {s:22s} {sum(v["ok"] for v in vs):3d}/{len(vs):<3d} block {sum(v["block"] for v in vs):3d}  first page: {top}')
    raise SystemExit
B = [load(d) for d in a.base]; A = [load(d) for d in a.arm]
ids = sorted(B[0]); assert all(sorted(x) == ids for x in B + A), 'runs cover different task sets'
for lab, X in (('base', B), ('arm', A)):
    print(f'{lab:5s} successes {[sum(x[t]["ok"] for t in ids) for x in X]} mean {sum(sum(x[t]["ok"] for t in ids) for x in X)/len(X):.2f}/{len(ids)}'
          f'; valid {[sum(x[t]["valid"] for t in ids) for x in X]}; block-page episodes {[sum(x[t]["block"] for t in ids) for x in X]}')
print('full   :', stats(B, A, ids))
print('clean  :', stats(B, A, [t for t in ids if all(x[t]['valid'] for x in B + A)]))
nb = [t for t in ids if not any(x[t]['block'] for x in B + A)]
print('noblock:', stats(B, A, nb) if len(nb) >= 30 else f'only {len(nb)} tasks, not tested')
if a.csv:
    with open(a.csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['task_id', 'arm', 'run', 'success', 'valid', 'block_page'])
        for lab, X in (('base', B), ('arm', A)):
            for k, x in enumerate(X, 1):
                for t in ids: w.writerow([t, lab, k, int(x[t]['ok']), int(x[t]['valid']), int(x[t]['block'])])
