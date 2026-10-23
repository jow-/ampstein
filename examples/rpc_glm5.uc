// examples/rpc_glm5.uc — a GLM pipeline split over two hosts, driven by the coordinator
// (docs/RPC.md phase 4): the worker owns the first stages, this process the rest and the head.
//
//   worker (e.g. j7, 2 GPUs):
//     ampstein --worker --connect COORD:PORT --model DIR            # --gpus 2 by default: all
//   coordinator (e.g. jai, 6 GPUs → 3 local stages + the worker's 1):
//     ucode -L ... -ltorch=ampstein.torch examples/rpc_glm5.uc -- \
//         --bind ADDR --port 7700 --model DIR --gpus 6 --stages 3 [--steps 32] [--prompt TEXT]
//
// --gpus / --stages are the LOCAL GPUs and stages; the worker adds its GPUs / (gpus / stages) stages
// in front. The prompt is prefilled with forward_chunk and continued greedily; the token ids
// printed on the `tokens:` line must equal those of the same run on one host (--single: the
// whole model on the local GPUs, the stage count taken as the TOTAL then).
// Plain TCP, no TLS: bind to the address of a VPN / overlay or of a private link.

import * as coord from 'ampstein.rpc_coord';
import * as glm5_tp from 'ampstein.glm5_tp';

let bind = null, port = 7700, model_dir = null, gpus = null, stages = 3, steps = 32, single = false;
let max_seq = 2048, kv_bits = null, use_graphs = false, chunk = 512, repeat = 1, pipeline = false, prompt = 'What is the capital of France? Answer in one sentence.';
let argv = ARGV, st = index(argv, '--');

if (st >= 0)
	argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	let a = argv[i];

	if (a == '--bind') bind = argv[++i];
	else if (a == '--port') port = int(argv[++i]);
	else if (a == '--model') model_dir = argv[++i];
	else if (a == '--gpus') gpus = int(argv[++i]);
	else if (a == '--stages') stages = int(argv[++i]);
	else if (a == '--steps') steps = int(argv[++i]);
	else if (a == '--prompt') prompt = argv[++i];
	else if (a == '--max-seq') max_seq = int(argv[++i]);
	else if (a == '--kv-q8') kv_bits = 8;
	else if (a == '--single') single = true;
	else if (a == '--graphs') use_graphs = true;
	else if (a == '--chunk') chunk = int(argv[++i]);
	else if (a == '--pipeline') pipeline = true;
	else if (a == '--repeat') repeat = int(argv[++i]);
	else
		die(`unknown option ${a}\n`);
}

if (!model_dir || (!bind && !single))
	die('usage: --bind ADDR --port N --model DIR [--gpus N] [--stages N] [--steps N] [--prompt TEXT] [--graphs] [--chunk N] [--repeat N] [--pipeline] | --single --model DIR\n');

gpus ??= torch.device_count();

let model, K = null;

if (single) {
	model = glm5_tp.load(model_dir, { devices: gpus, stages: stages, max_seq: max_seq, kv_bits: kv_bits, mtp: false });
}
else {
	let ex = coord.attach({ listen: `${bind}:${port}`, model: model_dir, gpus: gpus, stages: stages, max_seq: max_seq, kv_bits: kv_bits });

	K = ex.client;
	model = glm5_tp.load(model_dir, { devices: gpus, max_seq: max_seq, kv_bits: kv_bits, mtp: false, ...ex.load });
}

function now() {
	let t = clock(true);

	return t[0] + t[1] / 1e9;
}

let tok = torch.tokenizer(model_dir);
// --repeat N: the prompt text N times (a long context, to take the sparse-attention graph)
let body = '';

for (let i = 0; i < repeat; i++)
	body += (repeat > 1 ? `(${i}) ` : '') + prompt + '\n';

let ids = tok.encode(`[gMASK]<sop><|user|>\n${body}<|assistant|>\n</think>`, true);
let cache = model.new_cache(max_seq);
let G = null;

if (use_graphs) {
	// the graphs of the step (the worker captures its own first): their warm-up clobbers the caches
	let tg = now();

	G = model.graphs(cache, [ 1 ]);
	model.reset_cache(cache);
	torch.synchronize();
	warn(sprintf('rpc_glm5: decode graphs captured in %.1f s\n', now() - tg));
}

warn(`rpc_glm5: prompt of ${length(ids)} tokens\n`);

let t0 = now();
let lg;

if (pipeline) {
	// the engine's prefill: a wavefront over the stages (the worker's stages run ahead on their host)
	let pp = model.prefill_pipeline(cache, ids, { chunk: chunk });

	while (pp.step(4))
		;

	lg = pp.logits();
}
else {
	for (let at = 0; at < length(ids); at += chunk) {
		let part = slice(ids, at, at + chunk);

		lg = model.forward_chunk(part, cache, (at + chunk >= length(ids)) ? 1 : 0);
	}
}

torch.synchronize();

let t_prefill = now() - t0;
let out = [];

t0 = now();

for (let k = 0; k < steps; k++) {
	let t = lg.reshape([-1]).argmax().item();

	push(out, t);
	lg = G ? G.run([ t ], cache.length).logits : model.forward_chunk([t], cache, 1);
}

torch.synchronize();

let t_dec = now() - t0;

printf('prefill %d tokens: %.2f s\n', length(ids), t_prefill);
printf('decode %d tokens%s: %.2f s (%.1f ms/token)\n', steps, G ? ' (graphs)' : '', t_dec, t_dec * 1000.0 / steps);
printf('tokens: %s\n', join(',', out));
printf('text: %s\n', tok.decode(out));

if (K) {
	warn(`rpc_glm5: worker ${sprintf('%J', K.stats())}\n`);
	model.free_cache(cache);
	K.bye();
}
