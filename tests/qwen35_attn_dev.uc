// tests/qwen35_attn_dev.uc — qwen35.attn_forward_dev's S <= 5 branch
// (the decode attention path) with the tensor-core mma selection wired
// in: a FAKE full-attention layer (raw matmul linears, real
// qk_norm_rope/attn_gate/KV writes) at the real Qwen3.8-27B shapes
// (nh 24, kvh 4, head_dim 256, hidden 5120) drives the exact engine
// call, and is compared against the same path run with the shuffle
// kernel (the pre-wiring code, oracle-verified separately).
//
//   - S = 1 and S = 3: |mma path - shuffle path| at the fp16 noise
//     floor (rel-rms <= 0.01; the kernels agree to ~1e-3)
//   - determinism: two mma runs bit-identical
//   - timing: the mma path must be visibly FASTER than the shuffle path
//     at 16k keys (min over 20 runs, <= 1.6x) — proves the mma kernel is
//     actually launched by the selection, not silently the shuffle
//
// Needs a GPU with ~0.5 GiB free (skips otherwise, exit 125). No model.

import * as math from 'math';
import * as qwen35 from 'ampstein.qwen35';

if (torch.device_count() < 1) {
	warn('skip: needs a GPU\n');
	exit(125);
}

let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

const NH = 24, KVH = 4, HD = 256, HID = 5120, CS = 16384;
const DEV = 'cuda:0';
const EPS = 1e-6;

/* Fake linear: x @ w, w (in, out) — same orientation as the engine's
 * LinearFP16 (x @ W.T with stored (out, in)). */
let lin = function (w) {
	let l = {};

	l.forward = function (t) {
		return torch.matmul(t, w);
	};

	return l;
};

let L = {
	q_proj: lin(torch.rand([HID, 2 * NH * HD], 'float16', DEV)),
	k_proj: lin(torch.rand([HID, KVH * HD], 'float16', DEV)),
	v_proj: lin(torch.rand([HID, KVH * HD], 'float16', DEV)),
	o_proj: lin(torch.rand([NH * HD, HID], 'float16', DEV)),
	q_norm_w: torch.rand([HD], 'float16', DEV),
	k_norm_w: torch.rand([HD], 'float16', DEV)
};

let cfg = {
	num_q_heads: NH,
	num_kv_heads: KVH,
	head_dim: HD,
	hidden_size: HID,
	rms_norm_eps: EPS,
	rotary_dim: HD
};

let cl = {
	k_cache: torch.zeros([1, KVH, CS, HD], 'float16', DEV),
	v_cache: torch.zeros([1, KVH, CS, HD], 'float16', DEV)
};

function make_st(S) {
	let freq = [];

	for (let j = 0; j < HD / 2; j++)
		push(freq, math.pow(10000.0, -2.0 * j / HD));

	let freq_t = torch.from_array(freq, 'float32', DEV);
	let pos = torch.arange(S, 'float32', DEV).view([S, 1]).mul(
		freq_t.view([1, HD / 2]));

	/* KV written at [0, S); the whole cache is rand-filled so the
	 * attention sees all CS keys (the kernels read [0, seqused)) */
	return {
		pos_idx: torch.arange(S, 'int64', DEV),
		seqused: torch.full([1], CS, 'int32', DEV),
		cos: pos.cos(),
		sin: pos.sin()
	};
}

function fill_cache() {
	cl.k_cache = torch.rand([1, KVH, CS, HD], 'float16', DEV);
	cl.v_cache = torch.rand([1, KVH, CS, HD], 'float16', DEV);
}

/* The S <= 5 attention path run by hand with the SHUFFLE kernel — the
 * exact body of qwen35.attn_forward_dev minus the mma selection. */
function attn_shuf(x, st) {
	let S = x.shape()[1];

	let qg = L.q_proj.forward(x).view([1, S, NH, 2 * HD]);
	let q = qg.narrow(-1, 0, HD);
	let g = qg.narrow(-1, HD, HD);
	let kk = L.k_proj.forward(x).view([1, S, KVH, HD]);
	let v = L.v_proj.forward(x).view([1, S, KVH, HD]);

	q = torch.qk_norm_rope(q, L.q_norm_w, st.cos, st.sin, HD, EPS);
	kk = torch.qk_norm_rope(kk, L.k_norm_w, st.cos, st.sin, HD, EPS);
	cl.k_cache.index_copy_(2, st.pos_idx, kk.transpose(1, 2));
	cl.v_cache.index_copy_(2, st.pos_idx, v.transpose(1, 2));

	let o = torch.attn_decode(q.contiguous().view([S, NH, HD]),
		cl.k_cache.view([KVH, CS, HD]), cl.v_cache.view([KVH, CS, HD]),
		st.seqused, 1.0 / math.sqrt(HD), null, 0);

	return L.o_proj.forward(
		torch.attn_gate(o.view([1, S, NH, HD]).transpose(1, 2), g));
}

function rel_rms(a, b) {
	let d = a.sub(b).to('float32');
	let m = a.to('float32');

	return math.sqrt(d.mul(d).mean().item()) / math.sqrt(m.mul(m).mean().item());
}

/* --- correctness: mma path (via the engine function) vs shuffle ----- */
for (let S = 1; S <= 3; S++) {
	// fresh KV; both paths read the same cache (the [0, S) KV write is
	// idempotent — same x both times)
	fill_cache();
	let x = torch.rand([1, S, HID], 'float16', DEV);
	let st = make_st(S);
	let ref = attn_shuf(x, st);
	let out = qwen35.attn_forward_dev(cfg, L, x, cl, st);
	let rr = rel_rms(out, ref);
	check('attn_forward_dev S=' + S + ': mma path vs shuffle path',
		rr <= 0.01, 'rel-rms ' + rr);
}

/* --- determinism: two mma runs bit-identical ------------------------- */
fill_cache();
let xd = torch.rand([1, 3, HID], 'float16', DEV);
let d1 = qwen35.attn_forward_dev(cfg, L, xd, cl, make_st(3));
let d2 = qwen35.attn_forward_dev(cfg, L, xd, cl, make_st(3));
check('attn_forward_dev determinism (S=3)',
	d1.sub(d2).abs().max().item() == 0);

/* --- timing: the mma path must actually be the fast kernel ----------- */
function min_time(f, n) {
	f();      // warmup (allocator, smem attribute)
	let best = 1e30;

	for (let i = 0; i < n; i++) {
		let t0 = now_ms();
		f();
		torch.synchronize();
		let dt = now_ms() - t0;

		if (dt < best)
			best = dt;
	}

	return best;
}

fill_cache();
let xt = torch.rand([1, 3, HID], 'float16', DEV);
let st3 = make_st(3);
let t_mma = min_time(() => {
	qwen35.attn_forward_dev(cfg, L, xt, cl, st3);
}, 20);
let t_shuf = min_time(() => {
	attn_shuf(xt, st3);
}, 20);

warn('timing @' + CS + ' keys S=3: mma path ' + t_mma +
	' ms vs shuffle ' + t_shuf + ' ms\n');
check('mma path is the fast kernel (<= 0.95x shuffle time at 16k keys)',
	t_mma <= t_shuf * 0.95,
	'' + t_mma + ' vs ' + t_shuf);

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('all green\n');
