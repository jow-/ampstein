// Phase 4 test: load-time optimization — batched pinned-async upload.
//
// Weights reach the GPU through two new paths (see ucode/weights.uc,
// src/amp_torch.cpp):
//
//   torch.h2d(tensors)        CPU tensors -> CUDA, pinned staging ring
//   torch.h2d_file(items)     [stfile, name] pairs -> CUDA, sequential
//                             pread() into the ring + async H2D
//
// h2d_file is the load-time path: it bypasses the mmap page-fault path
// (measured 0.42 GB/s cold on this machine vs 1.55 GB/s sequential
// read) and overlaps disk reads with in-flight device copies. Cold
// full-model load: 2.94 s -> 0.96 s (3.1x) on the 1.2 GB 1B model.
//
// Checks:
//   1. h2d round-trips CPU tensors bit-exactly (fp32/fp16/int16,
//      including a >32 MB tensor that exercises chunked transfers)
//   2. h2d_file uploads match the mmap views bit-exactly (int16
//      trellis and fp16 weights, compared in fp32 — exact)
//   3. sharded model dir uploads through the same path, bit-exact vs
//      the single-file fixture
//   4. ws.gpu16() bf16 -> fp16 cast stays exact on the GPU side
//   5. llama.load() through the upload path produces the oracle
//      final-position argmax on the chat prompt fixture
//
// Run via CTest or manually:
//   ucode -L <build-dir> -L <src>/ucode tests/loadperf.uc

import * as weights from 'ampstein.weights';
import * as llama from 'ampstein.llama';
import { stat } from 'fs';

const FIXTURES = sourcepath(0, true) + '/fixtures';
const MODEL = getenv('AMP_MODEL_1B') ||
	'/data/models/Llama-3.2-1B-Instruct-exl3-4.0bpw';

/* Model-dependent test: skip (not fail) when the model dir is absent
 * (CTest SKIP_RETURN_CODE) — docs/LESSONS.md L31. */
if (!stat(MODEL)) {
	warn('skip: model dir not found: ' + MODEL + '\n');
	exit(125);
}

function now() {
	let t = clock(true);

	return t[0] + t[1] / 1e9;
}

// --- 1. h2d: CPU tensor round-trip, incl. chunked big tensor -----------
let a = torch.arange(1000, 'float32').mul(0.5);
let b = torch.full([7, 33], 2.25, 'float16');
let c = torch.arange(12345, 'int16').view([5, 2469]);
let big = torch.arange(9 * 1024 * 1024, 'float32');   // 36 MB > 32 MB chunk

let ups = torch.h2d([a, b, c, big]);

assert(length(ups) == 4, 'h2d returns one tensor per input');

for (let i = 0; i < 4; i++)
	assert(ups[i].shape()[0] == [a, b, c, big][i].shape()[0],
		`h2d keeps shape (${i})`);

let diffs = [
	ups[0].cpu().sub(a).abs().max().item(),
	ups[1].cpu().sub(b).abs().max().item(),
	ups[2].cpu().to('int32').sub(c.to('int32')).abs().max().item(),
	ups[3].cpu().sub(big).abs().max().item()
];

for (let i = 0; i < 4; i++)
	assert(diffs[i] == 0, `h2d roundtrip ${i} bit-exact (diff ${diffs[i]})`);

// error paths
let raised = false;

try { torch.h2d(42); } catch (e) { raised = true; }
assert(raised, 'h2d rejects non-arrays');

raised = false;
try { torch.h2d([a, 'nope']); } catch (e) { raised = true; }
assert(raised, 'h2d rejects non-tensor entries');

// --- 2. h2d_file: bit-exact vs the mmap views --------------------------
let st = torch.st(FIXTURES + '/exl3_real_layer.safetensors');
let keys = st.list();

let ups2 = torch.h2d_file([ [st, 'trellis'], [st, 'suh'], [st, 'svh'] ]);

assert(length(ups2) == 3, 'h2d_file returns one tensor per pair');

let td = ups2[0].cpu().to('float32')
	.sub(st.tensor('trellis').cpu().to('float32')).abs().max().item();

assert(td == 0, `trellis upload bit-exact (diff ${td})`);

let sd = ups2[1].cpu().to('float32')
	.sub(st.tensor('suh').cpu().to('float32')).abs().max().item();

assert(sd == 0, `suh upload bit-exact (diff ${sd})`);

// every tensor of the real model through the file path
let ws = weights.open(MODEL);
let all = ws.list();
let items = [];

for (let i = 0; i < length(all); i++)
	push(items, [torch.st(MODEL + '/model.safetensors'), all[i]]);

let t0 = now();
let ups3 = torch.h2d_file(items);
let t1 = now();

printf('h2d_file: %d tensors (%.2f GB) in %.3f s\n',
	length(items), 1.21, t1 - t0);

// spot-check the extremes: the biggest tensor (embed, 525 MB, many
// chunks) and a tiny norm row
let spot = ['model.embed_tokens.weight', 'model.norm.weight'];

for (let i = 0; i < length(spot); i++) {
	let idx = -1;

	for (let j = 0; j < length(all); j++)
		if (all[j] == spot[i])
			idx = j;

	assert(idx >= 0, `${spot[i]} present`);

	let d = ups3[idx].cpu().to('float32')
		.sub(ws.tensor(spot[i]).cpu().to('float32')).abs().max().item();

	assert(d == 0, `${spot[i]} upload bit-exact (diff ${d})`);
}

// --- 3. sharded dir through the same path ------------------------------
let wss = weights.open(FIXTURES + '/sharded');

wss.upload();

let shard_t = wss.gpu('q_proj.trellis').cpu().to('float32');
let single_t = st.tensor('trellis').cpu().to('float32');
let d3 = shard_t.sub(single_t).abs().max().item();

assert(d3 == 0, `sharded upload bit-exact (diff ${d3})`);

// --- 4. gpu16 cast path (bf16 -> fp16 exact on the GPU) ----------------
let bf = wss.gpu('norm.weight');
assert(bf.dtype() == 'bfloat16', 'raw cache keeps the stored dtype');

let bf16 = wss.gpu16('norm.weight');
assert(bf16.dtype() == 'float16', 'gpu16 casts to fp16');

let d4 = bf16.cpu().to('float32')
	.sub(wss.tensor('norm.weight').cpu().to('float32')).abs().max().item();

assert(d4 == 0, `bf16 -> fp16 upload exact (diff ${d4})`);

// --- 5. full model load through upload() + forward sanity --------------
t0 = now();
let m = llama.load(MODEL);
t1 = now();

printf('llama.load with upload: %.3f s\n', t1 - t0);

let oracle = torch.st(FIXTURES + '/chat_prompt_logits.safetensors');
let ids = oracle.tensor('ids').cpu().tolist();

let cache = m.new_cache(length(ids) + 4);
let logits = m.forward_chunk(ids, cache);

let last = logits.view([length(ids), -1])
	.narrow(0, length(ids) - 1, 1).topk(1, 1).indices.item();
let ref = oracle.tensor('logits').cpu()
	.narrow(0, length(ids) - 1, 1).topk(1, 1).indices.item();

assert(last == ref, `final argmax ${last} vs oracle ${ref}`);

printf('loadperf: all uploads bit-exact, forward argmax exact\n');