// examples/trace_routing_qwen4.uc — teacher-forced Qwen3.8-Flash-Next routing trace over recorded
// agent sessions (trace_routing.uc for the 512-expert, top-10 model). Output lines:
// PHASE sample layer e0..e9   (experts in router rank order, one line per token and layer)
// docs/EXPERT_STREAMING.md; input from tests/oracle/pi_samples.py.
//   ucode -L build/lib -L src/ucode -ltorch=ampstein.torch examples/trace_routing_qwen4.uc \
//       samples.json OUT.txt [gpus]
let torch = require('ampstein.torch');
global.torch = torch;
import * as glm5 from 'ampstein.glm5';
import * as qwen4 from 'ampstein.qwen4';
import * as chat from 'ampstein.chat_qwen35';
import { open, readfile } from 'fs';

let samples = json(readfile(ARGV[0]));
let out = open(ARGV[1], 'w');
let gpus = int(ARGV[2] ?? 2);
let dir = getenv('AMP_MODEL_QWEN4') || '/data/models/Qwen3.8-Flash-Next-exl3-2.05bpw';
let model = qwen4.load(dir, { devices: gpus });
let tk = torch.tokenizer(dir);

for (let i = 0; i < model.nl; i++)
	model.layers[i].moe.layer = i;

let phase = 'P', name = '';

glm5.set_trace((cfg, M, y, idx) => {
	let rows = idx.cpu().tolist();

	for (let i = 0; i < length(rows); i++)
		out.write(`${phase} ${name} ${M.layer} ${join(' ', rows[i])}\n`);
});

const CHUNK = 512;

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
	for (let p = 0; p < length(ids); p += CHUNK)
		model.forward_chunk(slice(ids, p, p + CHUNK), cache);

	phase = 'D';
	for (let t in cont)
		model.forward_chunk([t], cache);

	warn(sprintf('%s: %d prompt tokens, %d teacher-forced decode tokens\n', name, length(ids), length(cont)));
}
out.close();
