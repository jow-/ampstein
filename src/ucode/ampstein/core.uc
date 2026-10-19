// ucode/ampstein_core.uc - Ampstein engine host logic, imported by the
// ampstein entry. The server/engine logic lives here as a module so the
// entry stays lean (arg parsing, --help/--version, REQUIRE_SEARCH_PATH bootstrap).
//
// Loaded via runtime import('ampstein.core'): the entry pushes
// <prefix>/share/*.uc onto REQUIRE_SEARCH_PATH first. The imports below are
// ADJACENT-RELATIVE ('./x.uc'), because a module's own static imports are
// resolved at compile time against the importing file's directory or -L,
// never against REQUIRE_SEARCH_PATH (L117) — this keeps the installed tree
// relocatable with no -L. Native modules (torch) are found via the entry's
// REQUIRE_SEARCH_PATH and exposed as the global `torch`.

import * as struct from 'struct';
import * as uloop from 'uloop';
import * as math from 'math';
import {readfile, writefile, unlink} from 'fs';

import * as httpd from './httpd.uc';
import * as engine from './engine.uc';
import * as utpl from './utpl.uc';
import * as chat_jinja from './chat_jinja.uc';
import * as chat_qwen35 from './chat_qwen35.uc';
import * as chat_glm5 from './chat_glm5.uc';
import * as chat_dsv4 from './chat_dsv4.uc';
import * as chat_dsv41 from './chat_dsv41.uc';
import * as chat_gemma4 from './chat_gemma4.uc';
import * as gemma4 from './gemma4.uc';
import * as dsv41 from './dsv41.uc';
import * as dsv41_spark from './dsv41_spark.uc';
import * as dsv41_tp from './dsv41_tp.uc';
import * as dsv41_spark_tp from './dsv41_spark_tp.uc';
import * as qwen35 from './qwen35.uc';
import * as qwen35_tp from './qwen35_tp.uc';
import * as gemma4_tp from './gemma4_tp.uc';
import * as qwen35_vision from './qwen35_vision.uc';
import * as glm5 from './glm5.uc';
import * as glm5_tp from './glm5_tp.uc';
import * as glm5_vision from './glm5_vision.uc';
import * as qwen4 from './qwen4.uc';
import * as qwen4_tp from './qwen4_tp.uc';
import * as dsv4 from './dsv4.uc';
import * as dsv4_tp from './dsv4_tp.uc';
import * as dsv4_vision from './dsv4_vision.uc';
import * as dspark_tp from './dspark_tp.uc';
import * as dflash from './dflash.uc';
import { tool_history } from './toolhist.uc';
import { repeated as repeated_calls, clean_messages } from './callguard.uc';

// digest.so is optional (not built on every host): images are then keyed
// by a cheap length+sample fingerprint instead of SHA-256
let sha256 = null;

try {
	sha256 = require('digest').sha256;
}
catch (e) {}

function now_s() {
	return time();
}

function rand_id(prefix) {
	let s = prefix;

	for (let i = 0; i < 24; i++)
		s += substr('0123456789abcdefghijklmnopqrstuvwxyz', int(math.rand(0, 36)), 1);

	return s;
}

function basename(p) {
	p = rtrim(p, '/');

	let i = rindex(p, '/');

	return (i >= 0) ? substr(p, i + 1) : p;
}

/* Number of leading bytes of s that form complete UTF-8 sequences. */
function utf8_complete(s) {
	let n = length(s);
	let i = n - 1, back = 0;

	while (i >= 0 && back < 4) {
		let c = ord(s, i);

		if ((c & 0xC0) != 0x80) {
			// lead byte: how long should the sequence be?
			let need = (c < 0x80) ? 1 : ((c & 0xE0) == 0xC0) ? 2 :
				((c & 0xF0) == 0xE0) ? 3 : ((c & 0xF8) == 0xF0) ? 4 : 1;

			return (n - i >= need) ? n : i;
		}

		i--;
		back++;
	}

	return n;
}

export function serve(args) {
	let served_name = args.name || basename(args.model);

	let gen_cfg = {};

	try {
		gen_cfg = json(readfile(args.model + '/generation_config.json')) || {};
	}
	catch (e) {
		gen_cfg = {};
	}

	// model family: GLM-5.x (Glm5NextForConditionalGeneration) or Qwen3.5/3.8
	if (args.arch == 'auto') {
		let arch = '';

		try {
			arch = '' + (json(readfile(args.model + '/config.json')).architectures || [''])[0];
		}
		catch (e) {}

		args.arch = (index(arch, 'Glm5Next') == 0) ? 'glm5' :
			(index(arch, 'DeepseekV41') == 0) ? 'dsv41' :
			(index(arch, 'DeepseekV4') == 0) ? 'dsv4' :
			(index(arch, 'Qwen4Exp') == 0) ? 'qwen4' :
			(index(arch, 'Gemma4') == 0) ? 'gemma4' : 'qwen35';
	}

	const GLM = (args.arch == 'glm5');
	const Q4 = (args.arch == 'qwen4');       // Qwen3.8-Flash-Next (qwen4.uc)
	const DS = (args.arch == 'dsv4');        // DeepSeek-V4-Flash (dsv4.uc): layer split, no MTP / TP / vision yet
	const D41 = (args.arch == 'dsv41');      // DeepSeek-V4.1-Flash (dsv41.uc): layer split, text only
	const GEM = (args.arch == 'gemma4');     // Gemma 4 (gemma4.uc, gemma4_tp.uc): text only

	// default device count: every GPU for GLM (117 GB) and Qwen-Flash-Next (100 GB),
	// two for the dense Qwen
	if (args.gpus == null)
		args.gpus = (GLM || Q4 || DS || D41) ? torch.device_count() : 2;

	if (GEM) {
		// layer split (prefill pipelines over the stages: the fastest prefill) or --tp (tensor parallel: faster decode)
		args.tp = !!args.tp && args.gpus >= 2;
		args.mtp = 0;
	}

	// Qwen3.8-Flash-Next: the verify window is the draft window, so a window of 1 verifies a
	// single row and cannot gain anything; 3 measured best (2: 1.25-1.45x, 3: 1.4-1.8x, 4: same)
	if (Q4 && args.mtp == 1)
		args.mtp = 2;

	// DFlash: the block size is 8, a round proposes up to 7 drafts (--mtp N lowers it)
	if (args.dflash) {
		if (!args.tp || args.arch != 'qwen35') {
			warn('ampstein: --dflash needs --tp on a Qwen3.5 / 3.8 model\n');
			exit(1);
		}

		if (!args.mtp_set)
			args.mtp = 7;
	}

	// tensor-parallel prefill is host-bound per chunk (thousands of launches across the ranks per
	// layer pass): bigger chunks amortise it (37k tokens: 4096 -> 657 tok/s)
	if (Q4 && args.tp && !args.chunk_set)
		args.chunk = (args.stages > 1) ? 1024 : 4096;

	// Qwen3.5 / 3.8 TP: the prefill all-reduces use a P2P communicator sized from the chunk and
	// the recon GEMM amortises better; 4096 measured +~15% over 1024 on two 3090s (the
	// communicator costs ~250 MB per rank)
	if (args.arch == 'qwen35' && args.tp && !args.chunk_set && !(args.stages > 1))
		args.chunk = 4096;


	const D41_MTP0 = (args.mtp_set && args.mtp == 0);                // --mtp 0: no DSpark drafter

	// DeepSeek-V4: pairs of GPUs (tensor-parallel) in a pipeline of stages, with the DSpark block
	// drafter (--mtp 0 turns it off, --dspark-window N verifies N drafts per round, default 4);
	// --layer-split runs the plain layer split instead (no drafter)
	if (DS && !args.layer_split && args.gpus >= 2) {
		args.tp = true;

		if (!args.stages)
			args.stages = int(args.gpus / 2);
	}
	else if (D41 && args.tp && args.gpus >= 2) {
		// DeepSeek-V4.1: pairs of GPUs (tensor-parallel) in a pipeline of stages (dsv41_tp.uc)
		if (!args.stages)
			args.stages = int(args.gpus / 2);

		args.mtp = 0;
	}
	else if (DS || D41) {
		args.tp = false;
		args.mtp = 0;
	}

	if (DS && !args.dspark_window)
		args.dspark_window = 4;

	let chat = GLM ? chat_glm5 : DS ? chat_dsv4 : D41 ? chat_dsv41 : GEM ? chat_gemma4 : chat_qwen35;
	let cfg = GLM ? glm5.load_config(args.model) : DS ? dsv4.load_config(args.model) :
		D41 ? dsv41.load_config(args.model) : GEM ? gemma4.load_config(args.model) :
		Q4 ? qwen4.load_config(args.model) : qwen35.load_config(args.model);

	// DeepSeek-V4.1: the DSpark block drafter (mtp.* weights) unless --mtp 0
	const D41_SPARK = D41 && !D41_MTP0 && cfg.dspark_block > 0 && cfg.num_mtp_layers > 0;

	if (D41_SPARK && !args.dspark_window)
		args.dspark_window = 4;

	// DeepSeek-V4.1 packs ship no generation_config.json: end of sentence is token 1 (config.json), and
	// DeepSeek recommends temperature 1.0 / top_p 0.95
	if (D41 && gen_cfg.eos_token_id == null)
		gen_cfg = {eos_token_id: cfg.eos_token_id, temperature: 1.0, top_p: 0.95, ...gen_cfg};

	/* The transpiled chat template: chat_template.jinja -> jinja2utpl ->
	   loadstring, compiled ONCE here; this closure is invoked for every
	   chat request. User content lands in tainted segments, so a user
	   string containing a special-token sequence can never tokenize as a
	   special token (ucode/chat_jinja.uc, ucode/utpl.uc). */
	let render_prompt = chat_jinja.make_renderer(D41 ? sourcepath(0, true) + '/chat_dsv41.jinja' :
		GEM ? sourcepath(0, true) + '/chat_gemma4.jinja' : args.model, chat.tpl_opts);

	let devs = [];
	let per = args.split || int((cfg.num_hidden_layers + args.gpus - 1) / args.gpus);

	for (let i = 0; i < cfg.num_hidden_layers; i++) {
		let d = int(i / per);

		push(devs, (d >= args.gpus) ? args.gpus - 1 : d);
	}

	warn(`ampstein: loading ${args.model} over ${args.gpus} GPU(s), ctx ${args.ctx}\n`);

	uloop.init();

	let t0 = time();
	let tok = torch.tokenizer(args.model);

	// --cache-mode FP16 (default) | Q8: the GLM latent KV cache as int8 codes
	// with group scales (544 instead of 1024 bytes per token and MLA layer)
	if (args.cache_mode && args.cache_mode != 'FP16' && args.cache_mode != 'Q8') {
		warn('ampstein: --cache-mode must be FP16 or Q8\n');
		exit(1);
	}

	let kv_bits = (args.cache_mode == 'Q8') ? 8 : null;

	if (kv_bits && !GLM && !Q4)
		warn('ampstein: --cache-mode Q8 is implemented for GLM and Qwen3.8-Flash-Next only (ignored)\n');

	if (kv_bits && Q4)
		qwen4.set_kv_q8(true);      // the QSA K / V caches as int8 codes + a scale per 32 values

	let model;

	if (GLM) {
		model = args.tp ? glm5_tp.load(args.model, {devices: args.gpus, kv_bits: kv_bits, mtp: args.mtp > 0,
			stages: args.stages, max_seq: args.ctx, prefill_rows: args.chunk}) :
			glm5.load(args.model, {devices: args.gpus, mtp: args.mtp > 0, kv_bits: kv_bits,
				max_seq: args.ctx});
	}
	else if (DS) {
		model = args.tp ? dsv4_tp.load(args.model, {devices: args.gpus, stages: args.stages,
			drafter: args.mtp > 0, prefill_rows: args.chunk}) :
			dsv4.load(args.model, {devices: args.gpus});
	}
	else if (D41 && args.tp) {
		model = dsv41_tp.load(args.model, {devices: args.gpus, stages: args.stages, max_seq: args.ctx,
			prefill_rows: args.chunk, spark: D41_SPARK && args.stages == cfg.num_mtp_layers});

		if (model.spark_stages) {
			model.dspark = dsv41_spark_tp.load(model, {stages: model.spark_stages});
			warn(`ampstein: DSpark drafter loaded on pipeline stages ${join(',', model.spark_stages)} (verify window ${args.dspark_window})\n`);
		}
	}
	else if (D41) {
		model = dsv41.load(args.model, {devices: args.gpus, max_seq: args.ctx, prefill_rows: args.chunk,
			spark: D41_SPARK});

		if (D41_SPARK) {
			model.dspark = dsv41_spark.load(model, {devs: model.spark_devs});
			warn(`ampstein: DSpark drafter loaded on cuda:${join(',cuda:', model.spark_devs)} (verify window ${args.dspark_window})\n`);
		}
	}
	else if (GEM) {
		model = gemma4.load(args.model, {devices: args.gpus, prefill_rows: args.chunk});

		if (args.tp) {
			let ranks = [];

			for (let g = 0; g < args.gpus; g++)
				push(ranks, g);

			model = gemma4_tp.shard(model, {devices: ranks, prefill_rows: args.chunk});
		}
	}
	else if (Q4) {
		model = args.tp ? qwen4_tp.load(args.model, {devices: args.gpus, mtp: args.mtp > 0,
			prefill_rows: args.chunk, stages: args.stages}) :
			qwen4.load(args.model, {devices: args.gpus, mtp: args.mtp > 0});
	}
	else {
		model = qwen35.load(args.model, {devices: devs, mtp: args.mtp > 0 && !args.dflash});
	}

	if (DS && args.tp && args.mtp > 0) {
		model.dspark = dspark_tp.load(model, {max_seq: args.ctx});
		warn(`ampstein: DSpark drafter loaded (verify window ${args.dspark_window})\n`);
	}

	if (args.tp && !GLM && !Q4 && !DS && !D41 && !GEM) {
		if (args.gpus < 2) {
			warn('ampstein: --tp needs --gpus >= 2\n');
			exit(1);
		}

		let ranks = [];

		for (let g = 0; g < args.gpus; g++)
			push(ranks, g);

		model = qwen35_tp.shard(model, {devices: ranks, f0: args.tp_share,
			prefill_rows: args.chunk});

		if (args.dflash) {
			// DFlash 2 block drafter: --mtp N is its window of drafts per round (default 7);
			// the target's taps are recorded by the forward and the verify graphs
			model.dflash = dflash.load(args.dflash, model, {window: args.mtp});
			model.set_taps(model.dflash.layer_ids);
			warn(`ampstein: DFlash drafter loaded (window ${args.mtp})\n`);
		}
	}
	let eng = engine.create(model, {
		max_ctx: args.ctx,
		prefill_chunk: args.chunk,
		mtp_window: (DS || D41) ? 0 : args.mtp,
		dspark_window: args.dspark_window,
		draft_vocab: args.draft_vocab,
		warmup: args.warmup,
		cache_ram: args.cache_ram,
		cache_quant: args.cache_quant,
		slots: args.slots,
		slot_margin: args.slot_margin,
		slice_ms: args.slice_ms,
		slice_tokens: args.slice_tokens,
		graphs: ((GLM || Q4) && getenv('AMP_GLM5_NO_GRAPHS')) ? false : null
	});

	let banner = `ampstein: model loaded in ${time() - t0}s`;

	if (args.tp)
		banner += `, tensor parallel over ${model.G} GPUs` +
			((GLM || Q4 || DS || D41) ? ' (expert parallel MoE)' : ` (${model.comm.info().mode} all-reduce)`);

	if (eng.mtp_window)
		banner += model.dflash ? `, speculative decoding (window ${eng.mtp_window})` :
			`, MTP speculative decoding (window ${eng.mtp_window})`;

	if (model.dflash)
		banner += ', DFlash drafter';

	if (model.dspark)
		banner += ', DSpark speculative decoding';

	warn(banner + '\n');

	/* --- hot reload: restore the cache state saved by a previous shutdown ---
	 *
	 * On SIGINT/SIGTERM the server writes its live generation state (token
	 * history, attention KV, GDN states, MTP draft cache) to a scratch
	 * safetensors file and restores it on the next start — so an agentic
	 * session survives server restarts without re-prefilling the context.
	 * The port-qualified path keeps concurrent servers apart. */
	const CACHE_ST = `/tmp/Ampstein-cache-${args.port}.safetensors`;
	const CACHE_META = `${CACHE_ST}.json`;
	const layout = (GLM || Q4) ? {gpus: args.gpus, tp: !!args.tp, stages: args.stages ?? 0, kv_bits: kv_bits ?? 0} : {};

	(function restore_cache() {
		let meta;

		if (DS || D41 || GEM)
			return;     // no cache-state file for these architectures yet

		try {
			meta = json(readfile(CACHE_META));
		}
		catch (e) {
			return;     // no saved state
		}

		if (meta.format != 1 || meta.model != args.model) {
			warn('ampstein: ignoring saved cache state ' +
				'(model or format changed)\n');
			return;
		}

		// the layout of a GLM entry follows the placement (stages, KV form)
		for (let k in ['gpus', 'tp', 'stages', 'kv_bits'])
			if (exists(meta, k) && meta[k] != layout[k]) {
				warn(`ampstein: ignoring saved cache state (${k} ${meta[k]} -> ${layout[k]})\n`);
				return;
			}

		try {
			let n = eng.restore_state(CACHE_ST);

			warn(`ampstein: restored ${n} cached tokens from ${CACHE_ST}\n`);
		}
		catch (e) {
			warn(`ampstein: cache restore failed: ${e.message}\n`);

			try {
				unlink(CACHE_ST);
				unlink(CACHE_META);
			} catch (e2) {}
		}
	})();

	// Prompt tokenization cache (one entry): agent sessions re-send the
	// whole conversation every turn, so the rendered prefix usually
	// extends the previous one (the new part starts where the old render's
	// generation tail meets the new assistant content). The extension
	// point is a segment boundary and each side encodes independently,
	// so cached ids + encoding of the NEW segments (each with its own
	// taint) equals a full encode — and the tainted user content in the
	// new segments is never atomized. 645 KB / 210k tokens: a full
	// re-encode costs 268 ms.
	let prefix_cache = utpl.prefix_cache();

	// positions of <|image_pad|> in ids[from..] (offset by from)
	function scan_pads(ids, from) {
		let out = [];
		let pad = cfg.image_token_id;

		for (let i = from; i < length(ids); i++)
			if (ids[i] == pad)
				push(out, i);

		return out;
	}

	function encode_prefix(r) {
		return prefix_cache.run(
			(segs) => utpl.encode_segments(tok, {segments: segs}),
			r, scan_pads);
	}

	/* Turn checkpoint boundary: the token index just after the last user
	 * message. Chat templates drop the reasoning of assistant turns that
	 * precede the last user message, so when a new user message arrives
	 * (typically while a tool loop was running) the prompt changes from the
	 * first assistant turn after the PREVIOUS user message on. The engine
	 * keeps a recurrent-state checkpoint there (job.turn_at). One entry is
	 * cached per user turn: the boundary's token count is computed once
	 * (a render and a segment encode of the part up to it), later requests
	 * of the same turn find the cached segments as a prefix of theirs. */
	let turn_cache = null;     // {lu, segs, n}

	function turn_boundary(r, messages, tools, ropts, prefix_len, has_images) {
		if (has_images || type(messages) != 'array')
			return 0;

		let lu = -1;

		for (let i = 0; i < length(messages); i++)
			if (type(messages[i]) == 'object' && messages[i].role == 'user')
				lu = i;

		if (lu < 0)
			return 0;

		if (lu == length(messages) - 1) {
			turn_cache = {lu: lu, segs: r.segments, n: prefix_len};

			return prefix_len;
		}

		if (turn_cache && turn_cache.lu == lu && utpl.seg_prefix(r.segments, turn_cache.segs) >= 0)
			return turn_cache.n;

		try {
			let r1 = render_prompt(slice(messages, 0, lu + 1), tools, ropts);

			if (utpl.seg_prefix(r.segments, r1.segments) < 0)
				return 0;

			let n = length(utpl.encode_segments(tok, {segments: r1.segments}));

			if (n <= 0 || n > prefix_len)
				return 0;

			turn_cache = {lu: lu, segs: r1.segments, n: n};

			return n;
		}
		catch (e) {
			return 0;
		}
	}

	// image pad positions of the ids encode_prefix() just returned
	function prefix_pads() {
		let c = prefix_cache.get();

		if (c.pads == null)
			c.pads = scan_pads(c.ids, 0);

		return c.pads;
	}

	/* --- vision -------------------------------------------------------------------- */

	let vis = null;                   // the tower, loaded on first use
	let img_cache = {map: {}, order: [ ]};

	function vision() {
		if (!args.vision)
			die('image input is disabled (--no-vision)');

		if ((GLM || DS) ? !cfg.has_vision : !cfg.vision_merge)
			die('this model has no (supported) vision tower');

		if (!vis) {
			let t = time();

			warn(`ampstein: loading the vision tower on cuda:${args.vision_device}\n`);
			vis = DS ? dsv4_vision.load(args.model, {device: args.vision_device}) : GLM ? glm5_vision.load(args.model, {
				device: args.vision_device,
				max_tokens: int(args.image_max_pixels / 784)       // 28 x 28 pixels per row
			}) : qwen35_vision.load(args.model, {
				device: args.vision_device,
				max_pixels: args.image_max_pixels
			});
			warn(`ampstein: vision tower loaded in ${time() - t}s\n`);
		}

		return vis;
	}

	// image bytes of an OpenAI image_url / Qwen image content part
	function image_bytes(part) {
		let u = part.image_url;

		if (type(u) == 'object')
			u = u.url;

		if (u == null)
			u = part.image;

		if (type(u) == 'object')
			u = u.url;

		if (type(u) != 'string')
			die('image part has no url');

		if (index(u, 'data:') != 0)
			die('only data: URLs are supported for images (base64-encode the image)');

		let comma = index(u, ',');

		if (comma < 0 || index(substr(u, 0, comma), ';base64') < 0)
			die('image data: URL must be base64-encoded');

		let bytes = b64dec(substr(u, comma + 1));

		if (bytes == null || length(bytes) == 0)
			die('image data: URL is not valid base64');

		return bytes;
	}

	// identity of an image: SHA-256 when digest.so exists, else length plus a
	// rolling hash of up to 4096 sampled bytes
	function image_key(bytes) {
		if (sha256)
			return sha256(bytes);

		let n = length(bytes), h = n % 2147483647;
		let step = (n > 4096) ? int(n / 4096) : 1;

		for (let i = 0; i < n; i += step)
			h = (h * 31 + ord(bytes, i)) % 2147483647;

		return sprintf('%08x%08x', n, h);
	}

	// tower output for one image part; recently seen images are cached
	function image_embedding(part) {
		let bytes = image_bytes(part);
		let key = image_key(bytes);
		let hit = img_cache.map[key];

		if (hit) {
			img_cache.order = [ ...filter(img_cache.order, (k) => k != key), key ];

			return hit;
		}

		let t = time();
		let r = vision().embed(bytes);

		// debug: AMP_VIS_EMB=<safetensors with 'emb'> replaces the tower output (A/B of the text side)
		let ovr = getenv('AMP_VIS_EMB');

		if (ovr)
			r.emb = torch.st(ovr).tensor('emb').cuda(args.vision_device).to('float16');

		if (DS) {
			let ed = {key: key, r: r, ds: true, marker: qwen35.mm_marker(key)};

			warn(`ampstein: image ${r.width}x${r.height} -> ${r.nvw * 14}x${r.nvh * 14}, ` +
				`${r.nh}x${r.nw} tokens (${sprintf('%.2f', time() - t)}s)\n`);

			img_cache.map[key] = ed;
			push(img_cache.order, key);

			while (length(img_cache.order) > args.image_cache) {
				let old = shift(img_cache.order);

				delete img_cache.map[old];
			}

			return ed;
		}

		let e = {
			key: key,
			emb: r.emb,
			grid: r.grid,
			n: r.n_tokens,
			marker: qwen35.mm_marker(key)
		};

		warn(`ampstein: image ${r.width}x${r.height} -> ${r.new_width}x${r.new_height}, ` +
			`${r.n_tokens} tokens (${sprintf('%.2f', time() - t)}s)\n`);

		img_cache.map[key] = e;
		push(img_cache.order, key);

		while (length(img_cache.order) > args.image_cache) {
			let old = shift(img_cache.order);

			delete img_cache.map[old];
		}

		return e;
	}

	// unexpanded prompt ids (one <|image_pad|> per image) -> ids with every
	// pad replaced by the image's marker ids; returns {ids, images} where
	// images = [{start, grid, emb}] for the engine
	function expand_images(ids, pads, embs) {
		if (length(pads) != length(embs))
			die('image placeholder count does not match the image parts ' +
			    '(literal "<|image_pad|>" in the text?)');

		let out = [], images = [], prev = 0;

		for (let k = 0; k < length(pads); k++) {
			for (let o = prev; o < pads[k]; o += 4096)
				push(out, ...slice(ids, o, (o + 4096 < pads[k]) ? o + 4096 : pads[k]));

			let e = embs[k];

			if (DS) {
				// the placeholder becomes the whole image block: lead pads (so IMAGE_START lands on a
				// compressor group boundary), START, the interleaved rows, newlines, END; every row is
				// an embedding with a per-image marker id (type in the low bits)
				let b = vis.block(e.r, length(out));
				let base = e.marker + (e.marker - qwen35.MARK_BASE) * 7;

				push(images, {start: length(out), nc0: length(out) + b.lead, emb: b.rows, key: e.key});

				for (let t in b.types)
					push(out, base + t);

				prev = pads[k] + 1;
				continue;
			}

			push(images, {start: length(out), grid: e.grid, emb: e.emb});

			for (let r = 0; r < e.n; r++)
				push(out, e.marker);

			prev = pads[k] + 1;
		}

		for (let o = prev; o < length(ids); o += 4096)
			push(out, ...slice(ids, o, (o + 4096 < length(ids)) ? o + 4096 : length(ids)));

		return {ids: out, images: images};
	}

	// stop tokens: generation_config's eos list, plus <|im_end|>
	let stop_ids = [];
	let eos = gen_cfg.eos_token_id;

	if (type(eos) == 'array')
		stop_ids = eos;
	else if (eos != null)
		stop_ids = [eos];

	let im_end = tok.encode('<|im_end|>', true);

	if (length(im_end) == 1 && index(stop_ids, im_end[0]) < 0)
		push(stop_ids, im_end[0]);

	// end-of-turn policy (--eos-hook): the compiled script, run per consultation
	// with the request state as its globals
	let eos_policy = null;

	if (args.eos_hook) {
		try {
			eos_policy = loadfile(args.eos_hook);
		}
		catch (e) {
			warn(`ampstein: cannot load --eos-hook ${args.eos_hook}: ${e.message}\n`);
			exit(1);
		}
	}

	// tool-call policy (--call-hook): consulted when the parser completes a tool call
	let call_policy = null;

	if (args.call_hook) {
		try {
			call_policy = loadfile(args.call_hook);
		}
		catch (e) {
			warn(`ampstein: cannot load --call-hook ${args.call_hook}: ${e.message}\n`);
			exit(1);
		}
	}

	// prompt policy (--prompt-hook): consulted for every chat request, once rendered
	let prompt_policy = null;

	if (args.prompt_hook) {
		try {
			prompt_policy = loadfile(args.prompt_hook);
		}
		catch (e) {
			warn(`ampstein: cannot load --prompt-hook ${args.prompt_hook}: ${e.message}\n`);
			exit(1);
		}
	}

	let created_at = now_s();

	/* --- helpers ------------------------------------------------------------------ */

	function api_error(res, status, message, type_, code, param) {
		res.json(status, {
			error: {
				message: message,
				type: type_ || 'invalid_request_error',
				param: param || null,
				code: code || null
			}
		});
	}

	function model_card() {
		return {
			id: served_name,
			object: 'model',
			created: created_at,
			owned_by: 'Ampstein',
			root: args.model,
			max_model_len: eng.max_ctx
		};
	}

	/* Sampling parameters: request values, defaults from
	 * generation_config.json (as vLLM does). */
	function sampling_of(body) {
		function pick(k, def) {
			return (body[k] != null) ? body[k] : ((gen_cfg[k] != null) ? gen_cfg[k] : def);
		}

		return {
			temperature: pick('temperature', 1.0),
			top_p: pick('top_p', 1.0),
			top_k: pick('top_k', 0),
			min_p: pick('min_p', 0),
			presence_penalty: body.presence_penalty || 0,
			frequency_penalty: body.frequency_penalty || 0,
			seed: body.seed
		};
	}

	function stops_of(body) {
		let s = body.stop;

		if (s == null)
			return [];

		if (type(s) == 'string')
			return [s];

		return (type(s) == 'array') ? filter(s, (x) => type(x) == 'string' && x != '') : [];
	}

	function usage_of(info) {
		return {
			prompt_tokens: info.prompt_tokens,
			completion_tokens: info.completion_tokens,
			total_tokens: info.prompt_tokens + info.completion_tokens,
			prompt_tokens_details: {cached_tokens: info.cached_tokens || 0}
		};
	}

	/* decode token ids for a log line: image marker ids (>= 2^30) stand for the
	 * placeholder token, and a failure must never break the request */
	function safe_decode(ids) {
		try {
			return tok.decode(map(ids, (x) => (x >= 1073741824) ? (cfg.image_token_id || 0) : x));
		}
		catch (e) {
			return '?';
		}
	}

	function log_done(kind, info) {
		let t = info.timings;

		if (!t) {
			warn(`${kind}: ${info.reason}\n`);
			return;
		}

		let pf = (t.prefill_ms > 0) ? t.prefill_tokens * 1000 / t.prefill_ms : 0;
		let dc = (t.decode_ms > 0) ? t.decode_tokens * 1000 / t.decode_ms : 0;

		let mtp = info.mtp ? sprintf(', MTP %d rounds, %.0f%% of drafts accepted, %.2f tok/round',
			info.mtp.rounds, 100.0 * info.mtp.accepted / info.mtp.drafted,
			info.completion_tokens * 1.0 / info.mtp.rounds) : '';

		let why = '';
		let d = info.reuse_detail;

		// not a plain continuation of the live state: say why (client-side
		// re-rendering of the previous turn shows as a short live match)
		if (d && info.reuse != 'live') {
			why = sprintf(' [live %d/%d matched, snapshot %d/%d matched', d.live_lcp, d.ids, d.snap_lcp, d.snap);

			if (d.div_old)
				why += `, diverges: ${sprintf('%J', safe_decode(d.div_old))} -> ${sprintf('%J', safe_decode(d.div_new))}`;

			if (d.stored)
				why += ', previous state moved to RAM';

			why += ']';
		}

		if (getenv('AMP_ENGINE_PROBE') != null) {
			let ta = clock(true);
			let bad = eng.scan();
			let tb = clock(true);

			warn(sprintf('scan: %s (%.0f ms)\n', length(bad) ? join('; ', bad) : 'clean',
				(tb[0] - ta[0]) * 1000 + (tb[1] - ta[1]) / 1e6));
		}

		if (getenv('AMP_DSV4_PROBE') != null) {
			let pr = dsv4.probe_report();

			warn(sprintf('probe: stream-peak %g collapse-peak %g%s\n', pr[0], pr[1], pr[2] ? ' NONFINITE-OR-OVERFLOW' : ''));
		}

		warn(sprintf('%s: prompt %d (cached %d via %s%s), prefill %d tok in %.0f ms (%.1f tok/s), ' +
			'decode %d tok in %.0f ms (%.1f tok/s)%s, finish %s\n',
			kind, info.prompt_tokens, info.cached_tokens, info.reuse, why,
			t.prefill_tokens, t.prefill_ms, pf, t.decode_tokens, t.decode_ms, dc,
			mtp, info.reason));
	}

	/* Intermediate prefill progress: the engine calls on_progress once per
	 * processed chunk; log a line whenever another ~1024 tokens completed
	 * (and the final one), but only for prompts long enough that
	 * intermediate state is interesting (more than one chunk). */
	function progress_logger(kind, id) {
		let logged = 0;
		let t0 = 0, done0 = 0;   // clock reference: the first callback

		return function (done, total) {
			let t = clock(true);
			let now = t[0] * 1000 + t[1] / 1e6;

			if (!t0) {
				t0 = now;
				done0 = done;
			}

			if (total >= 2048 && done > logged && now > t0 &&
			    (done - logged >= 1024 || done == total)) {
				// rate over the observed span (from the first callback:
				// the very first chunk's time is not included)
				let dt = now - t0;
				let rate = (dt > 0) ? (done - done0) * 1000 / dt : 0;

				warn(sprintf('%s %s: prefill %d/%d tok (%d%%), %.1f tok/s\n',
					kind, id, done, total, int(100 * done / total), rate));
			}

			logged = done;
		};
	}

	/* Incremental detokenizer: per-token byte pieces (byte-level BPE
	 * concatenates exactly), emitted at UTF-8 boundaries. */
	function detok() {
		let pending = '';

		return {
			push: function (id) {
				pending += tok.decode([id]);

				let n = utf8_complete(pending);
				let out = substr(pending, 0, n);

				pending = substr(pending, n);

				return out;
			},

			flush: function () {
				let out = pending;

				pending = '';

				return out;
			}
		};
	}

	/* Limit max_new to the remaining context. */
	function max_new_of(body, prompt_len) {
		let room = eng.max_ctx - prompt_len - 1;
		let want = body.max_completion_tokens || body.max_tokens;

		if (want == null || want > room)
			want = room;

		return want;
	}

	/* --- chat completions ------------------------------------------------------- */

	function chat_completions(req, res, body) {
		let kw = body.chat_template_kwargs || {};
		let effort = body.reasoning_effort;
		let enable_thinking = kw.enable_thinking;

		if (GEM && enable_thinking == null)
			enable_thinking = false;           // Gemma 4 thinks only when asked (<|think|> in the system turn)

		if (effort != null && (lc('' + effort) == 'none' || lc('' + effort) == 'off'))
			enable_thinking = false;

		let tools = body.tools;

		if (body.tool_choice == 'none')
			tools = null;

		let r;

		let ropts = {
			enable_thinking: enable_thinking,
			reasoning_effort: kw.reasoning_effort || effort,
			preserve_thinking: kw.preserve_thinking,
			clear_thinking: kw.clear_thinking,
			vision: (GLM || DS) && args.vision && cfg.has_vision
		};

		// --call-guard: earlier calls are shown without the arguments their tool does
		// not declare (the model copies a stray one until it crowds the real ones out)
		let msgs = args.call_guard ? clean_messages(body.messages, tools) : body.messages;

		// Prompt policy (--prompt-hook): every request is shown to the script once
		// its prompt is rendered (with its size and whether it fits). An array
		// verdict replaces the request's messages, {messages, tools} either of
		// them; the prompt is rendered again and the script asked again, up to
		// --prompt-hook-max rounds. Nothing but the prompt changes: the client sees
		// the reply of the request it sent.
		let want = body.max_completion_tokens || body.max_tokens;

		if (want == null || want > room)
			want = room;

		return want;
	}

	/* --- chat completions ------------------------------------------------------- */

	function chat_completions(req, res, body) {
		let kw = body.chat_template_kwargs || {};
		let effort = body.reasoning_effort;
		let enable_thinking = kw.enable_thinking;

		if (GEM && enable_thinking == null)
			enable_thinking = false;           // Gemma 4 thinks only when asked (<|think|> in the system turn)

		if (effort != null && (lc('' + effort) == 'none' || lc('' + effort) == 'off'))
			enable_thinking = false;

		let tools = body.tools;

		if (body.tool_choice == 'none')
			tools = null;

		let r;

		let ropts = {
			enable_thinking: enable_thinking,
			reasoning_effort: kw.reasoning_effort || effort,
			preserve_thinking: kw.preserve_thinking,
			clear_thinking: kw.clear_thinking,
			vision: (GLM || DS) && args.vision && cfg.has_vision
		};

		// --call-guard: earlier calls are shown without the arguments their tool does
		// not declare (the model copies a stray one until it crowds the real ones out)
		let msgs = args.call_guard ? clean_messages(body.messages, tools) : body.messages;

		// Prompt policy (--prompt-hook): when the rendered prompt plus the room the
		// request asks for does not fit the context, the script may answer with
		// replacement messages (a compaction request shrunk to fit, say); the prompt
		// is rendered again and the script asked again, up to --prompt-hook-max
		// rounds. Nothing but the messages changes: the client sees the reply of
		// the request it sent.
		let want = body.max_completion_tokens || body.max_tokens || 0;
		let prefix_ids, images, gen_ids, prefix_len, turn_at, ids;

		for (let round = 0; ; round++) {
			try {
				r = render_prompt(msgs, tools, ropts);
			}
			catch (e) {
				return api_error(res, 400, e.message, 'invalid_request_error', null, 'messages');
			}

			prefix_ids = encode_prefix(r);
			images = null;

			if (getenv('AMP_DEBUG_PROMPT'))
				warn(`prompt: ${length(prefix_ids)} prefix ids, text ${sprintf('%J', r.text)}, ${length(msgs)} messages\n`);

			if (length(r.images)) {
				try {
					let embs = map(r.images, (part) => image_embedding(part));
					let x = expand_images(prefix_ids, prefix_pads(), embs);

					prefix_ids = x.ids;
					images = x.images;
				}
				catch (e) {
					return api_error(res, 400, e.message, 'invalid_request_error', null, 'messages');
				}
			}

			gen_ids = tok.encode(r.gen, true);
			prefix_len = length(prefix_ids);   // BEFORE extending: ids aliases prefix_ids (L81)
			turn_at = turn_boundary(r, msgs, tools, ropts, prefix_len, length(r.images) > 0);
			ids = prefix_ids;     // fresh array (encode / expansion): extend in place

			push(ids, ...gen_ids);

			if (!prompt_policy || round >= args.prompt_hook_max)
				break;

			let v = null;

			try {
				v = call(prompt_policy, null, {
					messages: msgs,
					tools: tools || [],
					tokens: length(ids),
					max_tokens: want,
					context: eng.max_ctx,
					limit: eng.max_ctx - 1 - want,   // the prompt size that leaves the asked room
					fits: length(ids) + want < eng.max_ctx,
					round: round
				});
			}
			catch (e) {
				warn(`prompt hook failed: ${e.message}\n`);
			}

			let nm = (type(v) == 'object') ? v.messages : v;

			if (type(nm) != 'array' && !(type(v) == 'object' && type(v.tools) == 'array'))
				break;

			warn(`prompt hook: round ${round}, ${length(ids)} tokens, limit ${eng.max_ctx - 1 - want}: ` +
				`${length(msgs)} -> ${length(nm ?? msgs)} messages\n`);

			if (type(nm) == 'array')
				msgs = nm;

			if (type(v) == 'object' && type(v.tools) == 'array')
				tools = v.tools;
		}

		if (length(ids) >= eng.max_ctx)
			return api_error(res, 400,
				`This model's maximum context length is ${eng.max_ctx} tokens. ` +
				`However, your messages resulted in ${length(ids)} tokens. ` +
				`Please reduce the length of the messages.`,
				'invalid_request_error', 'context_length_exceeded', 'messages');

		let stream = !!body.stream;
		let include_usage = stream && body.stream_options && body.stream_options.include_usage;
		let id = rand_id('chatcmpl-');
		let progress = progress_logger('chat', id);
		let created = now_s();
		let p = chat.parser(r.thinking, stops_of(body));
		let dt = detok();

		// Constrained tool calls (--call-guard): the guard follows the emitted text and
		// the engine asks it about every token it is about to accept (job.veto)
		let ban = (args.call_guard && chat.call_guard) ? repeated_calls(tool_history(msgs)) : [];

		if (length(ban))
			warn(`chat ${id}: call guard: the conversation repeats its last ${join(', ', ban)} call, ` +
				`banning the name in this response\n`);

		let guard = (args.call_guard && chat.call_guard) ? chat.call_guard(body.tools, r.thinking, ban) : null;
		let pieces = {}, vetoes = 0;

		function piece_of(t) {
			let k = sprintf('%d', t), s = pieces[k];

			if (s == null)
				s = pieces[k] = tok.decode([t]);

			return s;
		}

		function gpush(t) {
			let s = dt.push(t);

			if (guard)
				guard.feed(s);

			return s;
		}

		// non-stream only: the final JSON needs the full text; the streamed
		// path never reads it. Buffer-backed so a long completion costs O(n)
		// total instead of O(n^2) string concatenation.
		let content = struct.buffer(), reasoning = struct.buffer();
		let calls = [];
		let stopped = false;

		// End-of-turn policy state: what the model produced so far (kept apart
		// from the streamed / final buffers, which a stream does not fill)
		let htext = (eos_policy || call_policy) ? struct.buffer() : null;
		let hreason = (eos_policy || call_policy) ? struct.buffer() : null;
		let hnum = 0;

		// Tool-call policy (--call-hook): a completed call is shown to the
		// script before the client sees it. A string verdict drops the call
		// and forces that text into the model's context instead (delivered
		// through force() at a step boundary or, when the end token comes
		// first, through the end-of-turn path); any other verdict lets it
		// through. Only the first --call-hook-max verdicts of a request are
		// consulted: a model that insists gets its call out.
		let call_hist = call_policy ? tool_history(body.messages) : null;
		let call_n = 0, call_note = null, call_raw = false;

		// Reasoning budget. After `nudge_at` reasoning tokens a soft nudge is
		// appended to the model's thinking (it carries on and usually wraps up);
		// after `limit_at` the block is closed by force (nudge_text / close_text
		// end in </think>) and the model answers. Both are injected through the
		// engine's force() hook, at a step boundary (an MTP round may overshoot
		// by up to the window). Defaults: --reasoning-nudge / --reasoning-limit;
		// per request: reasoning_nudge_tokens / reasoning_max_tokens or
		// reasoning_budget: {nudge, limit}.
		function budget_num(flat, nested, dflt) {
			let v = (flat != null) ? flat : nested;

			return (v != null && v > 0) ? int(v) : (v != null ? 0 : dflt);
		}

		let rb = body.reasoning_budget || {};
		let nudge_at = budget_num(body.reasoning_nudge_tokens, rb.nudge, args.reasoning_nudge);
		let limit_at = budget_num(body.reasoning_max_tokens, rb.limit, args.reasoning_limit);
		let budgeted = r.thinking && (nudge_at > 0 || limit_at > 0);
		let rtoks = 0, nudged = false, closed = false, pending = null, pending_kind = null;

		if (nudge_at > 0 && limit_at > 0 && nudge_at >= limit_at)
			nudge_at = 0;

		function budget_check() {
			if (!budgeted || pending != null || p.mode() != 'reason')
				return;

			if (limit_at > 0 && rtoks >= limit_at && !closed) {
				closed = true;
				pending_kind = 'limit';
				pending = tok.encode(args.reasoning_close_text != null ? args.reasoning_close_text : chat.REASONING_CLOSE, true);
			}
			else if (nudge_at > 0 && rtoks >= nudge_at && !nudged) {
				nudged = true;
				pending_kind = 'nudge';
				pending = tok.encode(args.reasoning_nudge_text != null ? args.reasoning_nudge_text : chat.REASONING_NUDGE, true);
			}
		}

		function chunk(delta, finish) {
			return {
				id: id, object: 'chat.completion.chunk', created: created,
				model: served_name,
				choices: [{index: 0, delta: delta, logprobs: null,
					finish_reason: finish || null}]
			};
		}

		function emit(events) {
			for (let i = 0; i < length(events); i++) {
				let ev = events[i];

				if (ev.type == 'reasoning') {
					if (hreason)
						hreason.put('*', ev.text);

					if (!stream)
						reasoning.put('*', ev.text);

					if (stream)
						res.sse(chunk({reasoning_content: ev.text}));
				}
				else if (ev.type == 'content') {
					if (htext)
						htext.put('*', ev.text);

					if (!stream)
						content.put('*', ev.text);

					if (stream)
						res.sse(chunk({content: ev.text}));
				}
				else if (ev.type == 'tool_call') {
					let tc = {
						id: rand_id('call_'),
						type: 'function',
						function: {
							name: ev.name,
							arguments: sprintf('%J', chat.typed_arguments(ev, body.tools))
						}
					};

					// (a call behind a dropped one: the note is pending, drop it too)
					if (call_note != null)
						continue;

					if (call_policy && call_n < args.call_hook_max) {
						let v = null;

						try {
							v = call(call_policy, null, {
								call: {name: tc.function.name, arguments: tc.function.arguments},
								calls: map(calls, (c) => ({name: c.function.name, arguments: c.function.arguments})),
								history: call_hist,
								text: htext.slice(),
								reasoning: hreason.slice(),
								tools: map(body.tools || [], (t) => (t.function || t).name),
								tool_defs: body.tools || [],
								tokens: hnum,
								overrides: call_n
							});
						}
						catch (e) {
							warn(`chat ${id}: call hook failed: ${e.message}\n`);
						}

						if (type(v) == 'string' && v != '') {
							call_n++;

							// the note as the call's tool result (the model answers it
							// with a new call or a change of plan, nothing of it is
							// shown to the client); a format without tool_feedback gets
							// it as text inside its own turn
							let fb = chat.tool_feedback ? chat.tool_feedback(v, r.thinking) : null;

							call_raw = (fb != null);
							call_note = tok.encode(fb ?? v, true);

							// the parser ignores everything after a call (the format has
							// no suffix): continue with a fresh one in the state the
							// model is in
							p = chat.parser(call_raw ? r.thinking : false, stops_of(body));
							warn(`chat ${id}: call hook: dropped ${tc.function.name}, forcing ${length(call_note)} tokens (after ${hnum} tokens)\n`);
							continue;
						}
					}

					if (stream)
						res.sse(chunk({tool_calls: [{index: length(calls), ...tc}]}));

					push(calls, tc);
				}
				else if (ev.type == 'stop') {
					stopped = true;
				}
			}
		}

		if (stream) {
			res.stream(200);
			res.sse(chunk({role: 'assistant', content: ''}));
		}

		eng.submit({
			ids: ids,
			images: images,
			snap_at: prefix_len,
			turn_at: turn_at,
			max_new: max_new_of(body, length(ids)),
			sampling: sampling_of(body),
			stop_ids: stop_ids,

			alive: function () {
				return res.alive();
			},

			// would this token (after the pending prefix tokens) leave the tool's schema?
			veto: !guard ? null : function (prefix, t) {
				return guard.veto(map(prefix, piece_of), piece_of(t));
			},

			// the text a refused position must continue with, when it is determined:
			// shown to the client like generated text, and forced
			veto_force: !guard ? null : function (prefix) {
				let fix = guard.fix(map(prefix, piece_of));
				let f = (fix != null) ? tok.encode(fix, true) : null;

				if (vetoes++ < 8)
					warn(`chat ${id}: call guard: refused a token after ${hnum} tokens` +
						(f ? `, forcing ${sprintf('%J', fix)}` : ', redrawing') + '\n');

				if (f == null || length(f) == 0)
					return null;

				for (let i = 0; i < length(f); i++)
					emit(p.feed(gpush(f[i])));

				hnum += length(f);

				return f;
			},

			on_progress: function (done, total) {
				progress(done, total);

				// keep the stream warm and notice disconnects during long
				// prefills
				if (stream)
					return res.ping(`prefill ${done}/${total}`);

				return res.alive();
			},

			on_token: function (t) {
				hnum++;

				if (budgeted && p.mode() == 'reason')
					rtoks++;

				emit(p.feed(gpush(t)));
				budget_check();

				if (stopped)
					return false;

				return stream ? !res.finished() && res.alive() : res.alive();
			},

			// the model sampled its end token (--eos-hook): see the policy docs
			eos_max: (eos_policy ? args.eos_hook_max : 0) + (call_policy ? args.call_hook_max : 0),
			on_eos: !(eos_policy || call_policy) ? null : function (n) {
				let v = true;

				// a call dropped by the call hook: its note goes in first
				if (call_note != null) {
					let f = call_note;

					call_note = null;

					for (let i = 0; i < length(f) && !call_raw; i++)
						emit(p.feed(gpush(f[i])));

					hnum += length(f);

					return f;
				}

				if (!eos_policy)
					return true;

				try {
					v = call(eos_policy, null, {
						text: htext.slice(),
						reasoning: hreason.slice(),
						calls: map(calls, (c) => ({name: c.function.name, arguments: c.function.arguments})),
						tools: map(body.tools || [], (t) => (t.function || t).name),
						mode: p.mode(),
						tokens: hnum,
						overrides: n,
						finish: 'stop'
					});
				}
				catch (e) {
					warn(`chat ${id}: eos hook failed: ${e.message}\n`);
					return true;
				}

				if (type(v) == 'int' || type(v) == 'double') {
					if (v > 0)
						warn(`chat ${id}: eos hook: redraw with the stop ${v} logits down (after ${hnum} tokens)\n`);

					return (v > 0) ? v : true;
				}

				if (type(v) != 'string' || v == '')
					return true;

				let f = tok.encode(v, true);

				warn(`chat ${id}: eos hook: forcing ${length(f)} tokens (after ${hnum} tokens)\n`);

				// show the forced text in the stream like generated text
				for (let i = 0; i < length(f); i++)
					emit(p.feed(gpush(f[i])));

				hnum += length(f);

				return f;
			},

			// the engine polls this at every step boundary
			force: !(budgeted || call_policy) ? null : function () {
				if (call_note != null) {
					let f = call_note;

					call_note = null;

					for (let i = 0; i < length(f) && !call_raw; i++)
						emit(p.feed(gpush(f[i])));

					hnum += length(f);

					return f;
				}

				if (pending == null)
					return null;

				let f = pending;

				pending = null;

				// the model may have closed the block itself since the check
				if (p.mode() != 'reason')
					return null;

				warn(`chat ${id}: reasoning ${pending_kind} ` +
					`after ${rtoks} tokens, forcing ${length(f)} tokens\n`);

				// show the forced text in the stream like generated text
				for (let i = 0; i < length(f); i++)
					emit(p.feed(gpush(f[i])));

				rtoks += length(f);

				return f;
			},

			on_done: function (info) {
				emit(p.feed(dt.flush()));
				emit(p.finish());
				log_done('chat', info);

				if (info.reason == 'cancelled')
					return res.end();

				let finish = (length(calls) > 0) ? 'tool_calls' :
					((info.reason == 'length') ? 'length' : 'stop');

				if (stream) {
					res.sse(chunk({}, finish));

					if (include_usage)
						res.sse({id: id, object: 'chat.completion.chunk', created: created,
							model: served_name, choices: [], usage: usage_of(info)});

					res.sse('[DONE]');
					res.end();
					return;
				}

				let msg = {role: 'assistant', content: (content.length() == 0 && length(calls)) ? null : content.pull()};

				// trim() copies unconditionally (amp_trim_common always allocates),
				// so only pay for it when an edge byte is actually whitespace —
				// the common case pulls the buffer out with zero copies
				if (reasoning.length() != 0) {
					let rs = reasoning.pull();

					if (index(' \t\r\n', substr(rs, 0, 1)) >= 0 ||
					    index(' \t\r\n', substr(rs, -1)) >= 0)
						rs = trim(rs);

					msg.reasoning_content = rs;
				}

				if (length(calls))
					msg.tool_calls = calls;

				res.json(200, {
					id: id, object: 'chat.completion', created: created, model: served_name,
					choices: [{index: 0, message: msg, logprobs: null, finish_reason: finish}],
					usage: usage_of(info)
				});
			},

			on_error: function (msg) {
				warn(`chat: error: ${msg}\n`);

				if (stream) {
					res.sse({error: {message: msg, type: 'server_error', code: null}});
					res.end();
				}
				else {
					api_error(res, 500, msg, 'server_error');
				}
			}
		});
	}

	/* --- completions (raw prompt) ------------------------------------------------ */

	function completions(req, res, body) {
		let prompt = body.prompt;

		if (type(prompt) == 'array')
			prompt = (type(prompt[0]) == 'string') ? prompt[0] : null;

		if (type(prompt) != 'string')
			return api_error(res, 400, 'prompt must be a string', 'invalid_request_error', null, 'prompt');

		let ids = tok.encode(prompt, true);

		if (length(ids) >= eng.max_ctx)
			return api_error(res, 400,
				`This model's maximum context length is ${eng.max_ctx} tokens, ` +
				`your prompt has ${length(ids)} tokens.`,
				'invalid_request_error', 'context_length_exceeded', 'prompt');

		let stream = !!body.stream;
		let id = rand_id('cmpl-');
		let progress = progress_logger('completion', id);
		let created = now_s();
		let dt = detok();
		let stops = stops_of(body);
		// buffer-backed accumulation + incremental stop search (O(n) total,
		// not the O(n^2) of per-piece string concatenation + full re-scans)
		let text = chat.make_text(stops);
		let stopped = false;

		function piece(t) {
			stopped = text.stopped();

			return text.feed(t);
		}

		function chunk(t, finish) {
			return {id: id, object: 'text_completion', created: created, model: served_name,
				choices: [{index: 0, text: t, logprobs: null, finish_reason: finish || null}]};
		}

		if (stream)
			res.stream(200);

		eng.submit({
			ids: ids,
			snap_at: 0,
			max_new: max_new_of(body, length(ids)),
			sampling: sampling_of(body),
			stop_ids: stop_ids,

			alive: function () {return res.alive();},

			on_progress: function (done, total) {
				progress(done, total);

				return stream ? res.ping(`prefill ${done}/${total}`) : res.alive();
			},

			on_token: function (t) {
				let s = piece(dt.push(t));

				if (stream && s != '')
					res.sse(chunk(s));

				return !stopped && res.alive();
			},

			on_done: function (info) {
				if (!stopped) {
					let s = piece(dt.flush());

					if (stream && s != '')
						res.sse(chunk(s));
				}

				log_done('completion', info);

				if (info.reason == 'cancelled')
					return res.end();

				let finish = (info.reason == 'length') ? 'length' : 'stop';

				if (stream) {
					res.sse(chunk('', finish));
					res.sse('[DONE]');
					res.end();
					return;
				}

				res.json(200, {
					id: id, object: 'text_completion', created: created, model: served_name,
					choices: [{index: 0, text: text.done(), logprobs: null, finish_reason: finish}],
					usage: usage_of(info)
				});
			},

			on_error: function (msg) {
				if (stream) {
					res.sse({error: {message: msg, type: 'server_error', code: null}});
					res.end();
				}
				else {
					api_error(res, 500, msg, 'server_error');
				}
			}
		});
	}

	/* --- routing --------------------------------------------------------------------- */

	function route(req, res) {
		let path = req.path;

		if (index(path, '/v1/') == 0)
			path = substr(path, 3);

		// request length: Content-Length when present (chunked bodies have
		// none), human-readable
		let cl = req.headers['content-length'];
		let len = '';

		if (cl != null) {
			let n = int(cl);

			len = (n >= 1048576) ? sprintf(' (%.1f MiB body)', n / 1048576) :
				(n >= 1024) ? sprintf(' (%.1f KiB body)', n / 1024) :
				` (${n} B body)`;
		}

		warn(`${req.method} ${req.target}${len}\n`);

		if (req.method == 'OPTIONS')
			return res.send(204, {
				'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
				'Access-Control-Allow-Headers': '*'
			}, '');

		if (req.method == 'GET' && (path == '/health' || path == '/'))
			return res.json(200, {status: 'ok', model: served_name,
				busy: eng.busy(), queued: eng.queued(), cache: eng.cached()});

		if (req.method == 'GET' && path == '/models')
			return res.json(200, {object: 'list', data: [model_card()]});

		if (req.method == 'GET' && index(path, '/models/') == 0) {
			// lenient: any id resolves to the loaded model
			return res.json(200, model_card());
		}

		if (req.method == 'POST' && (path == '/chat/completions' || path == '/completions')) {
			let body;

			try {
				body = json(req.body);
			}
			catch (e) {
				return api_error(res, 400, 'invalid JSON body: ' + e.message);
			}

			if (type(body) != 'object')
				return api_error(res, 400, 'request body must be a JSON object');

			if (body.n != null && body.n > 1)
				return api_error(res, 400, 'n > 1 is not supported', 'invalid_request_error', null, 'n');

			if (path == '/chat/completions')
				return chat_completions(req, res, body);

			return completions(req, res, body);
		}

		return api_error(res, 404, `unknown endpoint ${req.method} ${req.path}`, 'invalid_request_error', 'not_found');
	}

	httpd.listen(args.host, args.port, route);
	warn(`ampstein: serving '${served_name}' on http://${args.host}:${args.port}/v1\n`);

	/* --- shutdown: save the cache state on SIGINT/SIGTERM ---------------------- */

	let shutting_down = false;

	function shutdown_cache() {
		if (shutting_down) {
			exit(0);    // second signal: force
			return;
		}

		shutting_down = true;

		try {
			if (DS || D41 || GEM)
				die('not supported for this architecture yet');

			let n = eng.save_state(CACHE_ST);

			writefile(CACHE_META, sprintf('%J', {
				format: 1, model: args.model, tokens: n,
				max_ctx: eng.max_ctx, mtp: eng.mtp_window, ...layout}));

			warn(`ampstein: saved ${n} cached tokens to ${CACHE_ST}\n`);
		}
		catch (e) {
			warn(`ampstein: cache state not saved (${e.message})\n`);
		}

		// let in-flight device work (and host-func callbacks) drain before
		// tearing down — exiting mid-callback corrupts the heap
		try {
			torch.synchronize();
		}
		catch (e2) {}

		uloop.end();
	}

	// keep the references: a collected signal object unregisters its handler
	let sig_int = uloop.signal('SIGINT', shutdown_cache);
	let sig_term = uloop.signal('SIGTERM', shutdown_cache);

	uloop.run();
}
