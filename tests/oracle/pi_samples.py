# Sample real agent conversations from ~/.pi/agent/sessions for examples/trace_routing.uc:
# a history (<= ~22k chars) plus the next assistant message to teacher-force. Only the
# project directories listed in `projs` are read: extend deliberately, sessions can hold
# private material.   python3 pi_samples.py [OUT.json]
import json, glob, os, random, sys
random.seed(int(os.environ.get('PI_SEED', 7)))   # PI_SEED=11: another draw of sessions (held-out samples for the speed benchmarks)
root = os.path.expanduser('~/.pi/agent/sessions/')
projs = ['ampstein.git','ucode.git','ucode-llm.git','exllamav3','llama.cpp.git','luci.git','rpcd.git','uhttpd.git',
         'openwrt-minibuild.git','cgi-io.git','iwinfo.git','vllm','tabbyAPI','ucode-lang-manual.git','uwsd.git']
files = []
for p in projs:
    for f in glob.glob(root + '--home-jow-devel-%s--/*.jsonl' % p):
        if 150_000 < os.path.getsize(f) < 6_000_000: files.append((p, f))
random.shuffle(files)
def txt(c):
    if isinstance(c, str): return c
    out = []
    for b in c:
        t = b.get('type')
        if t == 'text': out.append(b['text'])
        elif t == 'thinking': out.append(b['thinking'])
        elif t == 'toolCall': out.append('[call %s %s]' % (b['name'], json.dumps(b.get('arguments'), ensure_ascii=False)))
    return '\n'.join(out)
CH_TOOL = 1800
samples = []; used = set()
for p, f in files:
    if p in used and len(used) < 8: continue
    msgs = []
    for l in open(f):
        try: d = json.loads(l)
        except: continue
        if d.get('type') == 'message': msgs.append(d['message'])
    # candidate cut points: an assistant message preceded by >= 12 messages
    cuts = [i for i, m in enumerate(msgs) if m['role'] == 'assistant' and i >= 12 and len(txt(m['content'])) > 600]
    if not cuts: continue
    i = random.choice(cuts)
    hist = []; size = 0
    for m in reversed(msgs[:i]):
        if m['role'] not in ('user', 'assistant', 'toolResult'): continue
        t = txt(m['content'])
        if m['role'] == 'toolResult': t = t[:CH_TOOL]
        else: t = t[:3000]
        size += len(t)
        hist.append({'role': {'toolResult': 'tool'}.get(m['role'], m['role']), 'content': t})
        if size > 22000: break
    hist.reverse()
    while hist and hist[0]['role'] != 'user': hist.pop(0)
    if not hist: continue
    cont = txt(msgs[i]['content'])[:1600]
    samples.append({'name': '%s:%d' % (p, i), 'history': hist, 'cont': cont})
    used.add(p)
    if len(samples) == 8: break
json.dump(samples, open(sys.argv[1] if len(sys.argv) > 1 else 'samples.json', 'w'), ensure_ascii=False)
for s in samples: print(s['name'], len(s['history']), sum(len(h['content']) for h in s['history']), len(s['cont']))
