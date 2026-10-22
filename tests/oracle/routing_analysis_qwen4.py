# Analysis of examples/trace_routing_qwen4.uc output (docs/EXPERT_STREAMING.md): decode hit rate of a per-layer
# LRU, a global decayed-count cache (what a residency manager would keep), static oracles and a leave-one-out
# profile, as a function of the resident fraction.   python3 routing_analysis_qwen4.py route.txt
import sys, numpy as np
from collections import defaultdict, OrderedDict
NL, NE, K = 48, 512, 10
pre = defaultdict(lambda: [[] for _ in range(NL)]); dec = defaultdict(lambda: [[] for _ in range(NL)])
for line in open(sys.argv[1]):
    h = line.split(); (pre if h[0]=='P' else dec)[h[1]][int(h[2])].append(tuple(map(int,h[3:3+K])))
names = sorted(dec); ntok = {n: len(dec[n][0]) for n in names}
print("samples %d, decode tokens %d" % (len(names), sum(ntok.values())))
# token overlap / window distinctness
ov=[];d3=[]
for n in names:
    for L in range(NL):
        a=dec[n][L]
        ov += [len(set(a[i])&set(a[i-1])) for i in range(1,len(a))]
        d3 += [len(set(a[i])|set(a[i+1])|set(a[i+2])) for i in range(0,len(a)-2,3)]
print("consecutive-token expert overlap %.2f of %d; distinct experts in a 3-token window %.1f of 30" % (np.mean(ov),K,np.mean(d3)))
print("prefill touches %.0f of %d experts/layer (mean)" % (np.mean([len(set(e for a in pre[n][L] for e in a)) for n in names for L in range(NL)]), NE))

def warm_lru(n, L, C):
    lru = OrderedDict()
    for a in pre[n][L]:
        for e in a: lru[e]=1; lru.move_to_end(e)
        while len(lru) > C: lru.popitem(last=False)
    return lru
def sim_lru(f, warm=True):
    C=int(NE*f); hit=tot=0; rk=[0]*K; rm=[0]*K
    for n in names:
        for L in range(NL):
            lru = warm_lru(n,L,C) if warm else OrderedDict()
            for a in dec[n][L]:
                for r,e in enumerate(a):
                    tot+=1; rk[r]+=1
                    if e in lru: hit+=1; lru.move_to_end(e)
                    else: rm[r]+=1
                for e in a: lru[e]=1; lru.move_to_end(e)
                while len(lru)>C: lru.popitem(last=False)
    return hit/tot, rm, rk
def counts_of(ns, src):
    c=np.zeros((NL,NE))
    for n in ns:
        for L in range(NL):
            for a in src[n][L]:
                for e in a: c[L,e]+=1
    return c
def topmask(c, N):
    m=np.zeros(NL*NE,bool); m[np.argpartition(-c.ravel(), N-1)[:N]]=True; return m.reshape(NL,NE)
def eval_mask(m, n):
    h=t=0
    for L in range(NL):
        for a in dec[n][L]:
            h+=sum(m[L,e] for e in a); t+=K
    return h,t
def sim_global(f, decay, every=4, warm=True):
    N=int(NL*NE*f); H=T=0
    for n in names:
        c=np.zeros((NL,NE))
        if warm:
            c=counts_of([n],pre); c=c/ max(1,len(pre[n][0])) * 8   # prefill history as soft prior
        m=topmask(c,N); toks=ntok[n]; acc=[np.array(dec[n][L]) for L in range(NL)]
        for t in range(toks):
            for L in range(NL):
                e=acc[L][t]; H+=m[L,e].sum(); T+=K
            c*=decay
            for L in range(NL): c[L,acc[L][t]]+=1
            if t%every==every-1: m=topmask(c,N)
    return H/T
def sim_static_oracle(f, scope):
    N=int(NL*NE*f); H=T=0
    for n in names:
        c=counts_of([n],dec)
        if scope=='layer':
            m=np.zeros((NL,NE),bool)
            for L in range(NL): m[L,np.argsort(-c[L])[:int(NE*f)]]=True
        else: m=topmask(c,N)
        h,t=eval_mask(m,n); H+=h; T+=t
    return H/T
def sim_profile(f):  # leave-one-out profile from other samples' decode+prefill routing
    N=int(NL*NE*f); H=T=0
    for n in names:
        c=counts_of([x for x in names if x!=n],dec)+0.1*counts_of([x for x in names if x!=n],pre)/np.maximum(1,1)
        h,t=eval_mask(topmask(c,N),n); H+=h; T+=t
    return H/T
fs=[0.10,0.25,0.40,0.55,0.60,0.78]
print("\nresident fraction -> decode hit rate")
print("%-6s %-9s %-10s %-10s %-10s %-10s %-10s"%("frac","LRU/layer","LRU cold","glob d.99","glob d.97","oracle/lay","oracle gl"))
rows={}
for f in fs:
    l,rm,rk=sim_lru(f); rows[f]=(rm,rk)
    print("%-6.2f %-9.3f %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f"%(f,l,sim_lru(f,False)[0],sim_global(f,0.99),sim_global(f,0.97),sim_static_oracle(f,'layer'),sim_static_oracle(f,'global')),flush=True)
print("\nleave-one-out global profile (no adaptation) hit rate:")
for f in fs: print("%.2f %.3f"%(f,sim_profile(f)))
print("\nshare of LRU misses by router rank position (0=top):")
for f in (0.25,0.55,0.78):
    rm,rk=rows[f]; tm=sum(rm); print("frac %.2f miss rate %.3f; ranks>=7 hold %.2f of misses; per-rank miss rate:"%(f,tm/sum(rk),sum(rm[7:])/tm)," ".join("%.2f"%(a/b) for a,b in zip(rm,rk)))
