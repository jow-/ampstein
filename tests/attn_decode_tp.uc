// tests/attn_decode_tp.uc — the TP decode attention scenario, model-free:
// per-rank head counts (12 q / 2 kv, rep 6, hd 256 — the N=2 TP split of
// Qwen3.8) through qwen35.attn_forward_dev, which must select the tensor-
// core mma kernel, on BOTH devices, including inside ONE cross-device CUDA
// graph (rooted at device 0 with device 1 on a side stream — the TP graph
// shape, unlike the layer split's per-device captures).
//
//   1. mma vs shuffle per device, S = 1..3 (correctness)
//   2. one-graph capture on cuda:0 + cuda:1, replay at two positions —
//      bit-identical to the eager runs (the replay re-reads the device
//      position), determinism.
//
// No model; ~40 MB per GPU. Needs 2 GPUs (skips otherwise).

import * as math from 'math';
import * as qwen35 from 'ampstein.qwen35';

if (torch.device_count() < 2) {
	warn('skip: needs 2 GPUs\n');
	exit(125);
}

const NH = 12, KVH = 2, HD = 256, HID = 1024, CS = 4096;
const EPS = 1e-6;
let fails = 0;

function check(name, ok, detail) {
	if (!ok)
		fails++;
	print((ok ? 'ok: ' : 'FAIL: ') + name + (detail ? ' — ' + detail : '') + '\n');
}

let lin = function (w) {
	let l = {};
	l.forward = function (t) {
		return torch.matmul(t, w);
	};
	return l;
};

let W = function (a, b, dev) {
	return torch.rand([a, b], 'float16', dev).mul(0.02);
};

/* the TP per-rank config: shard_cfg keeps rep = nh/kvh, so the mma
 * selection in attn_forward_dev sees the same 6*S rows as layer split */
let cfg = {
	num_q_heads: NH,
	num_kv_heads: KVH,
	head_dim: HD,
	rms_norm_eps: EPS,
	rotary_dim: HD
};

function make_rank(dev) {
	let r = {};
	r.dev = dev;
	r.L = {
		q_proj: lin(W(HID, 2 * NH * HD, dev)),
		k_proj: lin(W(HID, KVH * HD, dev)),
		v_proj: lin(W(HID, KVH * HD, dev)),
		o_proj: lin(W(NH * HD, HID, dev)),
		q_norm_w: torch.rand([HD], 'float16', dev),
		k_norm_w: torch.rand([HD], 'float16', dev)
	};
	r.cl = {
		k_cache: torch.rand([1, KVH, CS, HD], 'float16', dev),
		v_cache: torch.rand([1, KVH, CS, HD], 'float16', dev)
	};
	r.x = torch.rand([1, 3, HID], 'float16', dev);
	// any values work (qk_norm_rope just multiplies/adds), fixed once
	r.cos_tab = torch.rand([16, HD / 2], 'float32', dev);
	r.sin_tab = torch.rand([16, HD / 2], 'float32', dev);
	r.pos = torch.zeros([1], 'int64', dev);
	return r;
}

function make_st(r, S) {
	let pos_idx = torch.arange(S, 'int64', r.dev).add(r.pos);

	return {
		pos_idx: pos_idx,
		seqused: pos_idx.narrow(0, S - 1, 1).add(1).to('int32'),
		cos: r.cos_tab.index(pos_idx),
		sin: r.sin_tab.index(pos_idx)
	};
}

/* the S <= 5 attention path run by hand with the SHUFFLE kernel */
function attn_shuf(r) {
	let L = r.L, cl = r.cl, S = r.x.shape()[1];
	let qg = L.q_proj.forward(r.x).view([1, S, NH, 2 * HD]);
	let q = qg.narrow(-1, 0, HD);
	let g = qg.narrow(-1, HD, HD);
	let kk = L.k_proj.forward(r.x).view([1, S, KVH, HD]);
	let v = L.v_proj.forward(r.x).view([1, S, KVH, HD]);
	let st = make_st(r, S);

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

let r0 = make_rank('cuda:0');
let r1 = make_rank('cuda:1');

/* --- 1. mma (engine selection) vs shuffle, per device, S = 1..3 ---- */
for (let S = 1; S <= 3; S++) {
	r0.x = torch.rand([1, S, HID], 'float16', 'cuda:0');
	r1.x = torch.rand([1, S, HID], 'float16', 'cuda:1');
	for (let n = 0; n < 2; n++) {
		let r = n == 0 ? r0 : r1;

		r.cl.k_cache.fill_(0);
		r.cl.v_cache.fill_(0);
		r.pos.fill_(0);
		torch.synchronize();

		let ref = attn_shuf(r);
		r.pos.fill_(0);
		torch.synchronize();
		let out = qwen35.attn_forward_dev(cfg, r.L, r.x, r.cl, make_st(r, S));

		check('S=' + S + ' cuda:' + (n == 0 ? 0 : 1) +
			': mma path vs shuffle path', rel_rms(out, ref) <= 0.01,
			'rel-rms ' + rel_rms(out, ref));
	}
}

/* --- 2. one cross-device graph (root cuda:0, side cuda:1) ---------- */
r0.x = torch.rand([1, 3, HID], 'float16', 'cuda:0');
r1.x = torch.rand([1, 3, HID], 'float16', 'cuda:1');

// everything the graph reads must stay reachable (docs/LESSONS.md L38)
let g = { r0: r0, r1: r1, out: null };

let body = function () {
	for (let n = 0; n < 2; n++) {
		let r = n == 0 ? r0 : r1;
		let st = make_st(r, 3);

		r.out = qwen35.attn_forward_dev(cfg, r.L, r.x, r.cl, st);
	}

	g.out = [ r0.out, r1.out ];
};

function eager_run(pos) {
	r0.pos.fill_(pos);
	r1.pos.fill_(pos);
	torch.synchronize();
	body();
	torch.synchronize();
	return [ r0.out.cpu(), r1.out.cpu() ];
}

let e0 = eager_run(0);
let e4 = eager_run(4);

let graph = torch.graph();
graph.capture_begin('cuda:0', [ 1 ]);
body();
graph.capture_end();
torch.synchronize();

function replay_run(pos) {
	r0.pos.fill_(pos);
	r1.pos.fill_(pos);
	torch.synchronize();
	graph.replay();
	torch.synchronize();
	return [ r0.out.cpu(), r1.out.cpu() ];
}

let rep0 = replay_run(0);
let rep4 = replay_run(4);
let rep0b = replay_run(0);

check('graph replay @0 bit-identical to eager (both devices)',
	rep0[0].sub(e0[0]).abs().max().item() == 0 &&
	rep0[1].sub(e0[1]).abs().max().item() == 0, '');
check('graph replay @4 bit-identical to eager (both devices)',
	rep4[0].sub(e4[0]).abs().max().item() == 0 &&
	rep4[1].sub(e4[1]).abs().max().item() == 0,
	'read-back of the position buffer');
check('mma replay determinism',
	rep0b[0].sub(rep0[0]).abs().max().item() == 0 &&
	rep0b[1].sub(rep0[1]).abs().max().item() == 0, '');
check('replay @0 != replay @4 (values really re-read)',
	rep0[0].sub(rep4[0]).abs().max().item() > 0, '');

if (fails)
	die(fails + ' failures');
print('all green\n');
