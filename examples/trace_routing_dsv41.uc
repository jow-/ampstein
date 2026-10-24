// examples/trace_routing_dsv41.uc — teacher-forced DeepSeek-V4.1 routing trace over recorded agent sessions
// (trace_routing_qwen4.uc for the 384-expert, top-6 model; run it on a pack that fits in VRAM). Output lines:
// PHASE sample layer e0..e5   (experts in router rank order, one line per token and layer)
// docs/EXPERT_STREAMING.md; input from tests/oracle/pi_samples.py, profile from tests/oracle/routing_profile.py
// (`routing_profile.py route.txt profile.json 40 384`).
//   ucode -L build/lib -L src/ucode -ltorch=ampstein.torch examples/trace_routing_dsv41.uc \
//       samples.json OUT.txt [gpus]
let torch = require('ampstein.torch');
global.torch = torch;
import * as glm5 from 'ampstein.glm5';
import * as dsv41 from 'ampstein.dsv41';
import { open, readfile } from 'fs';

let samples = json(readfile(ARGV[0]));
let out = open(ARGV[1], 'w');
let gpus = int(ARGV[2] ?? 6);
let dir = getenv('AMP_MODEL_DSV41') || '/data/models/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw';
const CHUNK = 512;

let model = dsv41.load(dir, { devices: gpus, max_seq: 16384, prefill_rows: CHUNK });
let tk = torch.tokenizer(dir);

for (let i = 0; i < length(model.layers); i++)
	model.layers[i].moe.layer = i;

let phase = 'P', name = '';

glm5.set_trace((cfg, M, y, idx) => {
	let rows = idx.cpu().tolist();

	for (let i = 0; i < length(rows); i++)
		out.write(`${phase} ${name} ${M.layer} ${join(' ', rows[i])}\n`);
});

for (let smp in samples) {
	name = replace(smp.name, ' ', '_');

	// (the chat template's turn markers, without its system prompt: the routing statistics need the content)
	let txt = '';
	for (let h in smp.history)
		txt += (h.role == 'assistant' ? '<｜Assistant｜>' : '<｜User｜>') + (h.role == 'tool' ? '[tool result]\n' : '') + h.content;

	let ids = tk.encode(txt + '<｜Assistant｜></think>', true);
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
