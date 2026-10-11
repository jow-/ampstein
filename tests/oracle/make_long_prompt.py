# tests/oracle/make_long_prompt.py — NOT SHIPPING (test tooling).
# Builds an OpenAI chat request of roughly N tokens from the repo's own
# sources with a needle (a vault access code) at 40% and a retrieval
# question at the end:   python3 make_long_prompt.py N OUT.json [REPO]
# Used by the long-context checks of the server (needle retrieval at 256k).
import glob
import json
import sys

target = int(sys.argv[1])
out = sys.argv[2]
root = sys.argv[3] if len(sys.argv) > 3 else '.'
files = sorted(glob.glob(root + '/ucode/*.uc') + glob.glob(root + '/tests/*.uc') +
               glob.glob(root + '/src/*.c*') + glob.glob(root + '/docs/*.md') +
               glob.glob(root + '/examples/*.uc'))
parts, n = [], 0

for f in files * 200:
    s = open(f, errors='ignore').read()
    parts.append('### FILE %s (part %d)\n%s\n' % (f, len(parts), s))
    n += len(s)

    if n > target * 4:
        break

parts.insert(int(len(parts) * 0.4),
             '\n### NOTE FROM THE ARCHIVIST\nThe vault access code is 7391-ALPHA. Remember it.\n')
body = ''.join(parts) + ("\n\nQuestion: what is the vault access code mentioned in the "
                         "archivist's note? Answer with the code only.")

json.dump({'model': 'x', 'max_tokens': 400, 'temperature': 0,
           'messages': [{'role': 'user', 'content': body}]}, open(out, 'w'))
print(len(body), 'chars')
