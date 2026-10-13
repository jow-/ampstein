// examples/trace_routing.uc — teacher-forced GLM routing trace over recorded agent sessions,
// with cross-layer router peeks (what would the router of layer L pick from the hidden state
// d layers earlier). Output lines: PHASE sample layer a0..a7 [| dD p0..p31]...
// docs/EXPERT_STREAMING.md; input from tests/oracle/pi_samples.py.
//   ucode -L build/lib -L src/ucode -ltorch=ampstein.torch examples/trace_routing.uc \
//       samples.json OUT.txt [gpus]
let torch = require('ampstein.torch');
global.torch = torch;
import * as glm5 from 'ampstein.glm5';
import * as chat from 'ampstein.chat_glm5';
import { open, readfile } from 'fs';

let samples = json(readfile(ARGV[0]));
let out = open(ARGV[1], 'w');
let gpus = int(ARGV[2] ?? 6);
let dir = getenv('AMP_MODEL_GLM5') || '/data/models/GLM-5.3-Flash-exl3-3.05bpw';
let model = glm5.load(dir, { devices: gpus });
let tk = torch.tokenizer(dir);

let phase = 'P', name = '';
let Y = {};          // layer -> last router input (device tensor), decode only
const DS = [1, 2, 4];

glm5.set_trace((cfg, M, y, idx) => {
	let L = M.layer;
	let rows = idx.cpu().tolist();
	let peeks = {};

	if (phase == 'D') {
		for (let d in DS) {
			let src = Y[sprintf('%d', L - d)];
			if (src != null)
				peeks[sprintf('%d', d)] = glm5.route_peek(cfg, M, src.cuda(M.dev), 32).cpu().tolist();
		}
		Y[sprintf('%d', L)] = y;
	}
	for (let i = 0; i < length(rows); i++) {
		let line = `${phase} ${name} ${L} ${join(' ', rows[i])}`;

		for (let d in DS)
			if (peeks[sprintf('%d', d)] != null)
				line += ` | d${d} ${join(' ', peeks[sprintf('%d', d)][i])}`;

		out.write(line + '\n');
	}
});

for (let smp in samples) {
	name = replace(smp.name, ' ', '_');

	let msgs = [];
	for (let h in smp.history)
		push(msgs, {role: h.role == 'tool' ? 'user' : h.role, content: (h.role == 'tool' ? '[tool result]\n' : '') + h.content});

	let r = chat.render(msgs, null, {});
	let ids = tk.encode(r.prefix + r.gen, true);
	let cont = tk.encode(smp.cont, false);
	let cache = model.new_cache(length(ids) + length(cont) + 16);

	phase = 'P';
	Y = {};
	let lg = model.prefill(ids, cache, 512, 1);

	phase = 'D';
	for (let t in cont)
		model.forward_chunk([t], cache, 1);

	warn(sprintf('%s: %d prompt tokens, %d teacher-forced decode tokens\n', name, length(ids), length(cont)));
}
out.close();
