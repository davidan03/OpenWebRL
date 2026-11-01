"""Compare pair corpora: pair types, chosen/rejected action types, stop pairs, turn index, response lengths."""
import json, sys, collections
def first(acts): return (acts[0].get('name') if acts else '<none>')
for path in sys.argv[1:]:
    n = 0; pt = collections.Counter(); ch = collections.Counter(); rj = collections.Counter(); cr = collections.Counter()
    turns = []; ctok = []; rtok = []; ncalls = collections.Counter(); src = collections.Counter()
    for line in open(path):
        d = json.loads(line); n += 1
        src[d['id'].split(':')[0]] += 1
        pt[d.get('pair_type')] += 1
        c, r = d['chosen'], d['rejected']; fc, fr = first(c.get('actions') or []), first(r.get('actions') or [])
        ch[fc] += 1; rj[fr] += 1; cr[(fc == 'done', fr == 'done')] += 1
        turns.append(int(d['turn'])); ctok.append(c['tokens']); rtok.append(r['tokens']); ncalls[len(c.get('actions') or [])] += 1
    turns.sort()
    print(f'== {path.split("/")[-2]}  rows {n}  sources {dict(src)}')
    print('  pair_type', [(k, round(v / n, 3)) for k, v in pt.most_common(8)])
    print('  chosen first action', [(k, round(v / n, 3)) for k, v in ch.most_common(8)])
    print('  rejected first action', [(k, round(v / n, 3)) for k, v in rj.most_common(8)])
    print(f'  done chosen/rejected: chosen-done&rej-not {cr[(True, False)]}, rej-done&chosen-not {cr[(False, True)]}, both {cr[(True, True)]}')
    print(f'  turn index: mean {sum(turns)/n:.2f} median {turns[n//2]} p90 {turns[int(.9*n)]} >=10 {sum(t>=10 for t in turns)/n:.3f}')
    print(f'  tokens chosen {sum(ctok)/n:.0f} rejected {sum(rtok)/n:.0f}; calls per chosen {dict(ncalls)}')
