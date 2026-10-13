# Analysis of examples/trace_routing.uc output (docs/EXPERT_STREAMING.md): peek recall,
# LRU + router-peek prefetch simulation, static working sets, step-time model.
#   python3 routing_analysis.py route.txt
import sys
from collections import defaultdict, Counter, OrderedDict
dec = defaultdict(lambda: defaultdict(list)); pre = defaultdict(lambda: defaultdict(list))
peek = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # d -> name -> layer -> [list]
for line in open(sys.argv[1]):
    parts = line.rstrip('\n').split(' | ')
    h = parts[0].split(); ph, name, L = h[0], h[1], int(h[2]); act = tuple(map(int, h[3:11]))
    if ph == 'P': pre[name][L].append(act); continue
    dec[name][L].append(act)
    for q in parts[1:]:
        t = q.split(); peek[int(t[0][1:])][name][L].append(list(map(int, t[1:])))
names = sorted(dec); layers = sorted(dec[names[0]])
ntok = sum(len(dec[n][layers[0]]) for n in names)
print("samples %d, decode tokens %d, MoE layers %d" % (len(names), ntok, len(layers)))

print("\n[1] recall of actual top-8 experts within the router's top-B guess made d layers earlier")
print("%-4s" % "d", *["B=%-3d" % b for b in (8, 12, 16, 24, 32)])
for d in (1, 2, 4):
    row = []
    for B in (8, 12, 16, 24, 32):
        h = t = 0
        for n in names:
            for L in layers:
                if L - d < layers[0]: continue
                for a, p in zip(dec[n][L], peek[d][n][L]):
                    s = set(p[:B]); h += sum(e in s for e in a); t += 8
        row.append(h / t)
    print("%-4d" % d, *["%.3f " % r for r in row])
# chance level for reference: B/288
print("chance: B/288 ->", *["%.3f" % (b / 288) for b in (8, 12, 16, 24, 32)])

def sim(C, d=None, B=0, policy='lru'):
    """per layer LRU of C slots; before layer L's compute the top-B peek is prefetched.
    returns (demand misses per token, prefetch transfers per token, hit rate at access)"""
    dm = pf = hit = tot = 0; toks = 0
    for n in names:
        for L in layers:
            c = OrderedDict()
            pk = (peek[d][n][L] or None) if d else None
            for i, a in enumerate(dec[n][L]):
                if pk is not None and B:
                    for e in pk[i][:B]:
                        if e not in c:
                            pf += 1; c[e] = 1
                            if len(c) > C: c.popitem(last=False)
                        else: c.move_to_end(e)
                for e in a:
                    tot += 1
                    if e in c: hit += 1; c.move_to_end(e)
                    else:
                        dm += 1; c[e] = 1
                        if len(c) > C: c.popitem(last=False)
        toks = ntok
    return dm / toks, pf / toks, hit / tot

print("\n[2] LRU cache of C experts per layer (42 layers) + prefetch of router-peek top-B (distance d)")
print("    columns: demand misses/token (stalls, all layers) | prefetch transfers/token | hit rate seen by the layer")
for f in (0.25, 0.5):
    C = int(f * 288)
    base = sim(C)
    print("C=%d (%.0f%%) no prefetch:            miss %6.1f  pref %6.1f  hit %.3f" % (C, f * 100, *base))
    for d in (1, 2, 4):
        for B in (8, 12, 16):
            r = sim(C, d, B)
            print("   d=%d B=%2d:                        miss %6.1f  pref %6.1f  hit %.3f" % (d, B, *r))
print("(all-miss reference: %d expert fetches per token)" % (8 * len(layers)))

def st(train, test, C):
    cn = Counter(e for ex in train for e in ex)
    res = set(e for e, _ in cn.most_common(C)); h = t = 0
    for ex in test:
        for e in ex: t += 1; h += e in res
    return h, t
print("\n[3] static resident set: from this prompt's PREFILL routing, tested on its decode")
for f in (0.125, 0.25, 0.5):
    C = int(f * 288); H = T = H2 = T2 = H3 = T3 = H4 = T4 = 0
    for n in names:
        for L in layers:
            h, t = st(pre[n][L], dec[n][L], C); H += h; T += t
            h, t = st(dec[n][L], dec[n][L], C); H2 += h; T2 += t
            h, t = st(dec[n][L][:32], dec[n][L][32:], C); H3 += h; T3 += t
            others = [e for o in names if o != n for e in pre[o][L]]
            h, t = st(others, dec[n][L], C); H4 += h; T4 += t
    print("C=%3d (%.3f): prefill->decode %.3f | oracle %.3f | first-32-decode->rest %.3f | other prompts' prefill %.3f"
          % (C, f, H / T, H2 / T2, H3 / T3, H4 / T4))

print("\n[4] decode-step time model: compute T0=30 ms/token; expert = 9.4 MB; G GPUs share the transfers (expert parallel)")
print("    no prefetch: T0 + misses*t_x/G (stalls in series) | prefetch d=1,B=8: max(T0, (miss+pref)*t_x/G)")
for f in (0.25, 0.5):
    C = int(f * 288)
    m0, _, _ = sim(C); m1, p1, _ = sim(C, 1, 8); m2, p2, _ = sim(C, 1, 16)
    for bw in (3.0, 12.0, 25.0):
        tx = 9.4 / bw  # ms per expert at GB/s (9.4 MB / (bw GB/s) = ms)
        for G in (4, 2):
            a = 30 + m0 * tx / G
            b = max(30, (m1 + p1) * tx / G + 0) ; b = max(b, 30 + m1 * tx / G * 0)  # overlapped
            c = max(30, (m2 + p2) * tx / G)
            print("C=%3d%% link %4.1f GB/s G=%d: demand-only %6.1f ms | d1B8 %6.1f ms | d1B16 %6.1f ms" % (f * 100, bw, G, a, b, c))
