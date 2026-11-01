"""Live selector repeat-avoidance from selection logs (her collector format: selections/*.jsonl, one line per turn with
candidates [{thought, action}], selected_index). Previous step = the executed candidate of the previous turn of the same
attempt. Over turns whose candidate set mixes repeats and non-repeats of the previous executed action (identical tool-call
signature): pick rate of a repeat vs chance (share of candidates that repeat; duplicates counted, as the selector sees
them), split by the previous action. Usage: python3 -I live_repeat_picks.py <root> [<root> ...]  (searched recursively)"""
import json, os, re, sys, collections
CALL = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)

def sig(text):
    out = []
    for c in CALL.findall(text or ''):
        try: d = json.loads(c); out.append((d.get('name'), json.dumps(d.get('arguments'), sort_keys=True)))
        except Exception: out.append(('UNPARSED', c[:60]))
    return tuple(out)

def files(root):
    for dp, dn, fn in os.walk(root):
        if os.path.basename(dp) == 'selections':
            for f in fn:
                if f.endswith('.jsonl'): yield os.path.join(dp, f)

for root in sys.argv[1:]:
    hit = chance = n = turns = 0; by = collections.defaultdict(lambda: [0, 0., 0])
    for f in files(root):
        rows = [json.loads(l) for l in open(f)]
        rows.sort(key=lambda r: int(r['turn']))
        prev = None
        for r in rows:
            turns += 1
            c = [sig(x.get('action') if isinstance(x, dict) else x) for x in r['candidates']]
            sel = int(r['selected_index'])
            if prev:
                rep = [s == prev for s in c]
                if any(rep) and not all(rep):
                    k = '+'.join(a[0] for a in prev); share = sum(rep) / len(rep)
                    hit += rep[sel]; chance += share; n += 1
                    by[k][0] += rep[sel]; by[k][1] += share; by[k][2] += 1
            prev = c[sel] if 0 <= sel < len(c) else None
    print(f'=== {root} ({turns} turns)')
    if n: print(f'  mixed-repeat turns {n}: picks repeat {hit/n:.3f} vs chance {chance/n:.3f}')
    for k, (h, cs, m) in sorted(by.items(), key=lambda kv: -kv[1][2])[:3]:
        print(f'     prev={k:10s} {m:5d} turns: picks repeat {h/m:.3f} vs chance {cs/m:.3f}')
