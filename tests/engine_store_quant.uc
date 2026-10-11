// tests/engine_store_quant.uc — tiered quantization of KV entries in the
// host-RAM store, at long context.
//
// A ~36k-token prompt P (varied filler, a needle sentence near the START, a
// question at the end) is run, then an unrelated prompt B (P's state moves to
// RAM with its front tiers quantized), then P + a short tail from RAM. The
// continuation must still find the needle and agree with a fresh prefill of the
// same prompt. Variants: default tiers (fp16 tail, int8 beyond) and an int4
// front (q8 span 16k, so the needle sits in the int4 zone). Needs a Qwen3.5
// family checkpoint (AMP_MODEL_QWEN35) and as many GPUs as it takes to hold it
// (layer split over all of them, at most 6).
import * as uloop from 'uloop';
import * as engine from 'ampstein.engine';
import * as qwen35 from 'ampstein.qwen35';
import { stat, readfile } from 'fs';

const MODEL = getenv('AMP_MODEL_QWEN35') || '/data/models/Qwen3.8-27B-exl3-6.00bpw';
const WORDS = +(getenv('AMP_STORE_Q_WORDS') || 26000);
const N = 24;

if (!stat(MODEL) || torch.device_count() < 2) {
	warn('skip: needs the Qwen model and at least two GPUs\n');
	exit(125);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

function run_until(cond, timeout_ms) {
	let deadline = now_ms() + timeout_ms;

	while (!cond()) {
		if (now_ms() > deadline)
			return false;

		uloop.run(20);
	}

	return true;
}

uloop.init();

let tok = torch.tokenizer(MODEL);
let raw = json(readfile(MODEL + '/config.json'));
let nl = (raw.text_config || raw).num_hidden_layers;
let G = min(torch.device_count(), 6), devs = [];

for (let i = 0; i < nl; i++)
	push(devs, int(i * G / nl));

let model = qwen35.load(MODEL, { devices: devs, mtp: true });
let fails = 0;

function check(name, ok, detail) {
	if (!ok) {
		fails++;
		warn('FAIL: ' + name + (detail != null ? ' — ' + detail : '') + '\n');
	}
	else
		warn('ok: ' + name + '\n');
}

const VOCAB = [ 'river', 'copper', 'lantern', 'harbor', 'quiet', 'engine', 'meadow', 'ledger',
	'orchard', 'violet', 'granite', 'compass', 'winter', 'falcon', 'marble', 'signal',
	'timber', 'canyon', 'velvet', 'anchor', 'thunder', 'pebble', 'saddle', 'mirror' ];

function filler(seed, words) {
	let s = [], x = seed;

	for (let i = 0; i < words; i++) {
		x = (x * 1103515245 + 12345) % 2147483648;
		push(s, VOCAB[(x >> 8) % length(VOCAB)]);

		if (i % 17 == 16)
			push(s, '.\n');
	}

	return join(' ', s);
}

let NEEDLE = '\nNOTE: The vault access code is 7391-ALPHA, remember it.\n';
let P = tok.encode(filler(1, 600) + NEEDLE + filler(2, WORDS) + '\nQuestion: what is the vault access code?\nAnswer: The vault access code is',
	false);
let B = tok.encode(filler(3, WORDS / 2), false);
let TAIL = tok.encode(' ', false);

printf('P %d tokens, B %d tokens, %d GPUs\n', length(P), length(B), G);

function run_job(eng, ids, snap_at) {
	let toks = [], info = null;

	eng.submit({
		ids: ids, snap_at: snap_at, max_new: N, stop_ids: [],
		sampling: { temperature: 0 },
		alive: () => true,
		on_progress: () => true,
		on_token: function (t) { push(toks, t); return true; },
		on_done: function (i) { info = i; },
		on_error: function (m) { warn('engine error: ' + m + '\n'); info = { reason: 'error' }; }
	});

	if (!run_until(() => info != null, 3600000))
		die('job timed out');

	return { toks: toks, info: info, text: tok.decode(toks) };
}

function same_head(a, b) {
	let k = 0;

	while (k < length(a) && k < length(b) && a[k] == b[k])
		k++;

	return k;
}

let mk = (quant) => engine.create(model, { max_ctx: length(P) + 2048, prefill_chunk: 1024,
	mtp_window: 0, warmup: false, cache_ram: 24, cache_quant: quant });

// the reference: a fresh prefill of P + TAIL
let eng = mk('off');
let ref = run_job(eng, [ ...P, ...TAIL ], 0);

printf('fresh answer: %s\n', sprintf('%J', ref.text));
check('the unquantized prompt finds the needle', index(ref.text, '7391') >= 0, ref.text);

for (let v in [
	{ name: 'tiers fp16 tail 8k / int8 / (no int4 reached)', q: { min: 16384, tail: 8192, q8: 65536 } },
	{ name: 'tiers fp16 tail 8k / int8 8k..16k / int4 front', q: { min: 16384, tail: 8192, q8: 16384 } },
	{ name: 'off (lossless store)', q: 'off' } ]) {
	eng = mk(v.q);
	run_job(eng, P, length(P) - 20);

	let t0 = now_ms();
	let b = run_job(eng, B, 0);
	let c = eng.cached();

	printf('[%s] stored %d entries, %.2f GiB in RAM\n', v.name, c.stored, c.stored_bytes / 1073741824.0);

	let x = run_job(eng, [ ...P, ...TAIL ], 0);

	printf('[%s] restored: reuse %s cached %d of %d; answer %s; agreement with fresh: %d of %d tokens\n',
		v.name, x.info.reuse, x.info.cached_tokens, length(P) + 1, sprintf('%J', x.text),
		same_head(x.toks, ref.toks), length(ref.toks));
	check(`[${v.name}] the state came back from RAM`, x.info.reuse == 'ram' && x.info.cached_tokens >= length(P) - 20,
		`${x.info.reuse} ${x.info.cached_tokens}`);
	check(`[${v.name}] the needle survives`, index(x.text, '7391') >= 0, x.text);
}

if (fails)
	die(`${fails} check(s) failed`);

print('engine_store_quant: ok\n');
