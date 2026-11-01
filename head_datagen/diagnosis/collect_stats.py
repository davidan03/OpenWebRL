"""Selector behaviour in a collection (head vs 4B-ARM): per executed turn, does the pick repeat the previous executed
action, pick stop/done, and how does that relate to episode outcome. Candidates are iid draws from the same actor in both
collections, so pick rates conditioned on the candidate set compare the selectors.
Usage: collect_stats.py <collection_dir> [max_tasks]"""
import json, os, re, sys, collections, glob, math
C = sys.argv[1]; LIM = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
TC = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.S)

def acts(cand):
    out = []
    for m in TC.finditer(cand.get('action') or ''):
        try: out.append(json.loads(m.group(1)))
        except Exception: out.append({'name': '<bad>'})
    return out

def key(a_list, tol=25):
    # coarse identity of an action list: names + args with coordinates bucketed
    ks = []
    for a in a_list:
        args = dict(a.get('arguments') or {})
        for k, v in list(args.items()):
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                args[k] = [round(x / tol) for x in v]
        ks.append((a.get('name'), json.dumps(args, sort_keys=True)))
    return tuple(ks)

STOP = {'terminate', 'answer', 'done', 'stop', 'finish'}
def is_stop(a_list): return any((a.get('name') or '').lower() in STOP for a in a_list)

S = collections.Counter(); names = collections.Counter(); ep = collections.Counter(); steps_succ = []
rep_runs = []
for i, att in enumerate(sorted(glob.glob(C + '/attempts/*/attempt-0001'))):
    if i >= LIM: break
    try: o = json.load(open(att + '/outcome.json'))
    except Exception: continue
    # judge-offline collections are re-judged into results/<task>.json; attempts/*/outcome.json keeps the unpatched verdict
    r = os.path.join(C, 'results', os.path.basename(os.path.dirname(att)) + '.json')
    if os.path.exists(r) and json.load(open(r)).get('attempt', '').endswith('/attempt-0001'): o = json.load(open(r))
    valid = str(o.get('valid')) == 'True'; succ = valid and float(o.get('reward') or 0) == 1.0
    tr = o.get('terminate_reason'); ep['valid'] += valid; ep['succ'] += succ; ep[f'tr:{tr}'] += valid
    turns = sorted(glob.glob(att + '/turn-*.json')); prev = None; run = 0; maxrun = 0
    for t in turns:
        d = json.load(open(t)); cands = [acts(c) for c in d['candidates']]; sel = int(d['selected_index'])
        ex = cands[sel]; keys = [key(c) for c in cands]
        for a in ex: names[a.get('name')] += 1
        g = 'S' if succ else ('F' if valid else 'X')
        S[f'turns_{g}'] += 1
        # stop choice when the candidate set is mixed
        st = [is_stop(c) for c in cands]
        if any(st) and not all(st):
            S[f'mixedstop_{g}'] += 1; S[f'pickstop_{g}'] += st[sel]
            S[f'mixedstop_frac_stop_sum_{g}'] += sum(st)  # chance baseline = mean share of stop candidates
        if prev is not None:
            rp = [k == prev for k in keys]
            S[f'withprev_{g}'] += 1; S[f'exrepeat_{g}'] += rp[sel]
            if any(rp) and not all(rp):
                S[f'mixedrep_{g}'] += 1; S[f'pickrep_{g}'] += rp[sel]; S[f'mixedrep_frac_sum_{g}'] += sum(rp)
            run = run + 1 if rp[sel] else 0; maxrun = max(maxrun, run)
        prev = keys[sel]
        sc = d.get('scores')
        if isinstance(sc, list) and len(sc) == 5 and all(isinstance(x, (int, float)) for x in sc):
            m = max(sc); z = [math.exp(x - m) for x in sc]; p = max(z) / sum(z)
            S['conf_n'] += 1; S['conf_sum'] += p; S['conf_lt05'] += p < 0.5
    if succ: steps_succ.append(len(turns)); rep_runs.append(maxrun)
print('collection', C)
print('episodes', dict(ep))
def r(a, b): return f"{S[a]/max(S[b],1):.3f} ({S[a]}/{S[b]})"
for g, lab in (('S', 'success eps'), ('F', 'failed eps'), ('X', 'invalid eps')):
    print(f"[{lab}] turns {S['turns_'+g]} | executed repeats prev {r('exrepeat_'+g,'withprev_'+g)}"
          f" | mixed-repeat sets: picks repeat {r('pickrep_'+g,'mixedrep_'+g)} vs chance {S['mixedrep_frac_sum_'+g]/max(5*S['mixedrep_'+g],1):.3f}"
          f" | mixed-stop sets: picks stop {r('pickstop_'+g,'mixedstop_'+g)} vs chance {S['mixedstop_frac_stop_sum_'+g]/max(5*S['mixedstop_'+g],1):.3f}")
if steps_succ:
    steps_succ.sort(); print(f"success ep turns: mean {sum(steps_succ)/len(steps_succ):.2f} median {steps_succ[len(steps_succ)//2]} >=20: {sum(x>=20 for x in steps_succ)}")
    print(f"success eps with a repeat run >=2: {sum(x>=2 for x in rep_runs)} / {len(rep_runs)}; >=4: {sum(x>=4 for x in rep_runs)}")
if S['conf_n']: print(f"head top-prob mean {S['conf_sum']/S['conf_n']:.3f}, <0.5 on {S['conf_lt05']/S['conf_n']:.3f} of turns")
print('action names', names.most_common(12))
