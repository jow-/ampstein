// tests/qwen35_vision.uc — image prompt through the whole model.
//
// Real 27B checkpoint, layer split over 2 GPUs. Checks, for a prompt
// carrying the oracle image (tests/fixtures/vision_test.png):
//   1. greedy continuation == exllamav3's (tests/oracle/
//      vision_generate_exllamav3.py, prompt "Describe this image
//      briefly.", thinking off): the first 18 generated tokens' text;
//   2. chunking invariance: prefill in 64-token chunks (boundaries
//      inside the image) vs one chunk gives the same next-token
//      argmax and logits within fp16 noise;
//   3. decode consistency after an image: prefill N tokens + one eager
//      decode step == prefill of the same N+1 tokens (MRoPE offset on
//      the decode path), and the same through a captured decode graph.

import * as qwen35 from 'ampstein.qwen35';
import * as qgraph from 'ampstein.qwen35_graph';
import * as vision from 'ampstein.qwen35_vision';
import { stat, readfile } from 'fs';

const M = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';
const PNG = `${sourcepath(0, true)}/fixtures/vision_test.png`;

if (!stat(M) || !stat(PNG)) {
	warn('skip: model dir or vision fixture missing\n');
	exit(125);
}

const EXPECT = 'This image is a simple digital graphic featuring three basic geometric shapes';

let cfg = qwen35.load_config(M);
let devs = [];

for (let i = 0; i < cfg.num_hidden_layers; i++)
	push(devs, (i < cfg.num_hidden_layers / 2) ? 0 : 1);

let model = qwen35.load(M, { devices: devs, mtp: false });
let v = vision.load(M, { device: 0 });
let r = v.embed(readfile(PNG));
let tok = torch.tokenizer(M);

let ids = [ ...tok.encode('<|im_start|>user\n', true), cfg.vision_start_id ];
let start = length(ids);

for (let i = 0; i < r.n_tokens; i++)
	push(ids, qwen35.MARK_BASE + 1);

push(ids, ...tok.encode('<|vision_end|>Describe this image briefly.<|im_end|>\n' +
	'<|im_start|>assistant\n<think>\n\n</think>\n\n', true));

assert(length(ids) == 89, 'prompt length matches exllamav3 (89)');

function fresh(max) {
	let cache = model.new_cache(max);

	cache.mm = qwen35.mm_build(cfg, [{ start: start, grid: r.grid, emb: r.emb }]);

	return cache;
}

function prefill(cache, seq, chunk) {
	let run = qwen35.prefill_pipeline(model, cache, seq, { chunk: chunk });

	while (run.step(1));

	return run.logits().reshape([-1]);
}

/* 1. greedy continuation vs exllamav3 */
let cache = fresh(512);
let lg = prefill(cache, ids, 1024);
let out = [];

for (let n = 0; n < 18; n++) {
	let t = lg.argmax().item();

	push(out, t);
	lg = model.forward_chunk([t], cache).reshape([-1]);
}

let text = tok.decode(out);

printf('continuation: %s\n', text);
assert(substr(text, 0, length(EXPECT)) == EXPECT,
	'greedy continuation matches exllamav3');

/* 2. chunking invariance */
let a = prefill(fresh(512), ids, 1024).to('float32');
let b = prefill(fresh(512), ids, 64).to('float32');
let dmax = a.sub(b).abs().max().item();

printf('chunk 64 vs 1024: argmax %d / %d, max |dlogit| %g\n',
	a.argmax().item(), b.argmax().item(), dmax);
assert(a.argmax().item() == b.argmax().item(), 'chunking changes the argmax');
assert(dmax < 0.5, 'chunking changes the logits');

/* 3. decode after an image == prefill of the longer sequence */
let c1 = fresh(512);
let l1 = prefill(c1, ids, 1024);
let nxt = l1.argmax().item();
let eager = model.forward_chunk([nxt], c1).reshape([-1]).to('float32');
let full = prefill(fresh(512), [ ...ids, nxt ], 1024).to('float32');

assert(eager.argmax().item() == full.argmax().item(), 'eager decode argmax');
assert(eager.sub(full).abs().max().item() < 0.5, 'eager decode logits');

/* graph: create() clobbers the cache with its warmup, so reset the
 * recurrent state and the length before the real prefill */
let cg = fresh(512);
let G2 = qgraph.create(model, cg, [1]);

cg.length = 0;
for (let i = 0; i < length(model.layers); i++) {
	let cl = cg.layers[i];

	if (cl.rec_state) {
		cl.rec_state.fill_(0);
		cl.conv_state.fill_(0);
	}
}

let lg2 = prefill(cg, ids, 1024);
let gr = G2.run([lg2.argmax().item()], cg.length).logits.reshape([-1]).to('float32');

assert(gr.argmax().item() == full.argmax().item(), 'graph decode argmax');
assert(gr.sub(full).abs().max().item() < 0.5, 'graph decode logits');

try { torch.synchronize(); } catch (e) {}

print('qwen35_vision: ok\n');
