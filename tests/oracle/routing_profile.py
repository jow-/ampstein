# Turn the output of examples/trace_routing_qwen4.uc into the routing profile that ampstein --expert-profile ranks
# the experts by: counts[layer][expert] = decode routings + 0.1 * prefill routings.
#   python3 routing_profile.py route.txt profile.json [layers experts]   (default 48 512: Qwen3.8-Flash-Next; DSV4.1: 40 384)
import json, sys
NL, NE = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (48, 512)
c = [[0.0] * NE for _ in range(NL)]
for line in open(sys.argv[1]):
    h = line.split()
    w = 1.0 if h[0] == 'D' else 0.1
    for e in h[3:]:
        c[int(h[2])][int(e)] += w
json.dump({'layers': NL, 'experts': NE, 'counts': [[round(x, 1) for x in r] for r in c],
           'note': 'decode count + 0.1 * prefill count of %s' % sys.argv[1]}, open(sys.argv[2], 'w'))
