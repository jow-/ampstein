// tests/qwen35_layers.uc — per-layer oracle test for the Qwen3.8 hybrid
// architecture (GDN linear-attention + full attention), against
// tests/fixtures/qwen35_layers.safetensors (produced by
// tests/oracle/make_qwen35_layer_oracle.py through exllamav3's module
// chain).
//
// Replays the same chain the oracle captured: embed -> block0 (GDN) ->
// block1 (GDN) -> block2 (GDN) -> block3 (full attention) -> norm ->
// lm_head, comparing every boundary. The reference graph runs fp32
// intermediates; ours is fp16 — expect rel-RMS ~1e-3..1e-2 noise, not
// bit-exactness (same contract as the llama KLD harness).

import * as qwen35 from 'ampstein.qwen35';
import * as models from 'ampstein.models';
import * as weights from 'ampstein.weights';
import * as math from 'math';
import { stat } from 'fs';

const MODEL_DIR = getenv('AMP_MODEL_QWEN35') ||
	'/data/models/Qwen3.8-27B-exl3-5.00bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL_DIR)) {
	warn('skip: model dir not found: ' + MODEL_DIR + '\n');
	exit(125);
}

const FIXTURE = `${sourcepath(0, true)}/fixtures/qwen35_layers.safetensors`;

function rel_rms(a, b) {
	let d = a.to('float32').sub(b.to('float32'));
	let num = d.mul(d).mean().item();
	let den = b.to('float32').mul(b.to('float32')).mean().item();

	return math.sqrt(num / den);
}

function maxdiff(a, b) {
	let d = a.to('float32').sub(b.to('float32'));

	return d.abs().max().item();
}

let fx = torch.st(FIXTURE);

let ids_t = fx.tensor('ids');
let ids = ids_t.tolist();
let S = length(ids);
printf('ids: %J  (S = %d)\n', ids, S);

let cfg = qwen35.load_config(MODEL_DIR);
printf('layers: %d, types 0-3: %J %J %J %J\n', cfg.num_hidden_layers,
	cfg.layer_types[0], cfg.layer_types[1], cfg.layer_types[2],
	cfg.layer_types[3]);

let ws = weights.open(MODEL_DIR);

// upload only what this test touches (embed + blocks 0-3 + norm + head)
ws.upload(function (key) {
	if (index(key, 'model.language_model.layers.') == 0) {
		let rest = substr(key, length('model.language_model.layers.'));
		let idx = +substr(rest, 0, index(rest, '.'));

		return (idx < 4) ? 0 : null;
	}

	if (index(key, 'model.language_model.embed_tokens') == 0 ||
	    index(key, 'model.language_model.norm') == 0 ||
	    index(key, 'lm_head') == 0)
		return 0;

	return null;
});

let eps = cfg.rms_norm_eps;

// --- embed ------------------------------------------------------------
let embed = ws.gpu16('model.language_model.embed_tokens.weight', 0);
let ids_gpu = torch.from_array(ids, 'int64').view([1, S]).cuda(0);
let h = embed.index(ids_gpu);

let want = fx.tensor('h_embed').cuda().view([1, S, cfg.hidden_size]);
printf('embed:   rel-rms %.6f  maxdiff %.4f\n',
	rel_rms(h, want), maxdiff(h, want));

// --- blocks 0..3 --------------------------------------------------------
let layers = [];
for (let i = 0; i < 4; i++)
	push(layers, qwen35.build_layer(ws, cfg, i, 0));

// per-layer caches: GDN state / attention KV + rope tables
let caches = [];
for (let i = 0; i < 4; i++) {
	let L = layers[i];
	let cl = { };

	if (L.type == 'linear_attention') {
		cl.conv_state = torch.zeros(
			[1, cfg.fdim_qkv, cfg.conv_kernel - 1],
			'float16', 'cuda:0');
		cl.rec_state = torch.zeros(
			[cfg.ln_num_v_heads, cfg.ln_k_head_dim, cfg.ln_v_head_dim],
			'float32', 'cuda:0');
	} else {
		cl.k_cache = torch.zeros(
			[1, cfg.num_kv_heads, S, cfg.head_dim], 'float16', 'cuda:0');
		cl.v_cache = torch.zeros(
			[1, cfg.num_kv_heads, S, cfg.head_dim], 'float16', 'cuda:0');

		let rc = qwen35.rope_cache(cfg, S, 0);
		cl.cos = rc.cos;
		cl.sin = rc.sin;
	}

	push(caches, cl);
}

for (let i = 0; i < 4; i++) {
	let L = layers[i];
	let cl = caches[i];

	let r = qwen35.rmsnorm_bias1(h, L.input_layernorm, eps);
	let attn_out = (L.type == 'linear_attention') ?
		qwen35.gdn_forward(cfg, L, r, cl) :
		qwen35.attn_forward(cfg, L, r, cl.cos.narrow(0, 0, S),
			cl.sin.narrow(0, 0, S), cl, 0);

	h = h.add(attn_out);

	r = qwen35.rmsnorm_bias1(h, L.post_attention_layernorm, eps);
	h = h.add(qwen35.mlp_forward(L, r));

	want = fx.tensor('h_block' + i).cuda().view([1, S, cfg.hidden_size]);
	printf('block%d (%s): rel-rms %.6f  maxdiff %.4f\n',
		i, L.type, rel_rms(h, want), maxdiff(h, want));
}

// --- final norm + head ---------------------------------------------------
let norm_w = ws.gpu16('model.language_model.norm.weight', 0).add(1.0);
h = qwen35.rmsnorm_bias1(h, norm_w, eps);

let head = models.LinearEXL3Seq(ws, 'lm_head', 0);
let logits = head.forward(h);

want = fx.tensor('logits').cuda().view([1, S, cfg.vocab_size]);
let md = maxdiff(logits, want);
printf('logits:  rel-rms %.6f  maxdiff %.4f\n',
	rel_rms(logits, want), md);

// argmax per position must match (the generation-relevant signal)
let am_got = logits.view([S, -1]).topk(1, 1).indices.view([S]).tolist();
let am_want = want.view([S, -1]).topk(1, 1).indices.view([S]).tolist();
let am_ok = true;

for (let i = 0; i < S; i++)
	if (am_got[i] != am_want[i])
		am_ok = false;

printf('argmax: %J\n', am_got);
printf('want:   %J\n', am_want);

if (!am_ok)
	die('argmax mismatch');

if (md > 1.0)
	die(`logits maxdiff ${md} exceeds bound 1.0`);

printf('PASS\n');