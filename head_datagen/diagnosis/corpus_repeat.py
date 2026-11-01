"""Per pair: does chosen / rejected repeat the previous executed action (last assistant turn in the prompt)?
Counts pairs that teach 'repeat' (chosen repeats, rejected does not) vs 'don't repeat' (the reverse)."""
import json, re, sys, collections
TC = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)
def key(js, tol=25):
    try: a = json.loads(js)
    except Exception: return ('<bad>',)
    args = {k: v for k, v in (a.get('arguments') or {}).items() if k not in ('clicks', 'button', 'is_hotkey', 'presses', 'interval')}
    for k, v in list(args.items()):
        if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v): args[k] = [round(x / tol) for x in v]
    return (a.get('name'), json.dumps(args, sort_keys=True))
def akey(acts, tol=25):
    return tuple(key(json.dumps(a), tol) for a in acts)
for path in sys.argv[1:]:
    C = collections.Counter(); by = collections.Counter()
    for line in open(path):
        d = json.loads(line)
        segs = d['prompt'].split('<|im_start|>assistant')[1:-1]  # last = generation prefix
        if not segs: C['turn0'] += 1; continue
        prev = tuple(key(m.group(1)) for m in TC.finditer(segs[-1].split('<|im_end|>')[0]))
        cr = akey(d['chosen'].get('actions') or []) == prev; rr = akey(d['rejected'].get('actions') or []) == prev
        C['with_prev'] += 1; C[('ch_rep' if cr else 'ch_new') + '_' + ('rj_rep' if rr else 'rj_new')] += 1
        if cr != rr: by[(d.get('pair_type'), 'teach_repeat' if cr else 'teach_new')] += 1
    n = C['with_prev']
    print(f"== {path.split('/')[-2]}: pairs with a previous action {n} (turn-0 {C['turn0']})")
    print(f"   teach REPEAT (chosen repeats, rejected new) {C['ch_rep_rj_new']} = {C['ch_rep_rj_new']/n:.3f} | teach NEW (rejected repeats) {C['ch_new_rj_rep']} = {C['ch_new_rj_rep']/n:.3f} | both repeat {C['ch_rep_rj_rep']}")
    print('   by pair type:', sorted(by.items(), key=lambda z: -z[1])[:8])
