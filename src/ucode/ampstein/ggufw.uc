// ucode/ggufw.uc — weight resolution over GGUF files (ggml-quantized checkpoints) for DeepSeek-V4.1.
//
// The same interface as weights.uc (has / tensor / info / list / gpu / gpu16 / upload / release / drop / file), addressed
// by the HF checkpoint names the engine uses (layers.N.attn.wq_a.weight, ...): each is mapped to the GGUF tensor of the
// llama.cpp-style naming (blk.N.attn_q_a.weight, ...). A model directory holds the backbone shards
// (<name>-0000i-of-0000N.gguf) and, for DeepSeek-V4.1, the shared Engram file (<name>-Engram-<type>.gguf), plus the
// config.json / tokenizer.json of the original checkpoint (a GGUF carries no config.json the engine can read).
//
// What differs from a safetensors checkpoint:
//   - block-quantized tensors (Q8_0, Q5_K, IQ2_XXS, ...): tensor() / gpu() give the raw uint8 rows (rows, row bytes),
//     info() says {dtype: "IQ2_XXS", shape: the logical shape, nbytes}, qtype(key) names the type (null for plain
//     tensors); gpu16() dequantizes. The linear layers are built from them by ggufq.uc.
//   - the routed experts of a layer are one stacked tensor per matrix: layers.N.ffn.exps.w1 / w3 / w2 (gate / up / down),
//     shape (experts, rows, columns), instead of 384 x 3 separate ones.
//   - the Engram tables are IQ4_NL (layers.N.engram.embed.weight, rows of 144 bytes, no scale tensor) in the separate
//     Engram GGUF, or FP8 + scales (.weight / .scale) from safetensors files of the directory (an EXL3 pack's shards);
//     the hash tables of engram_meta.safetensors come from the GGUF's own engram_* tensors (engram_meta()).

import {lsdir, stat} from 'fs';

// blk.N.<gguf name> -> layers.N.<hf name>
const BLK = {
	'attn_q_a.weight': 'attn.wq_a.weight', 'attn_q_a_norm.weight': 'attn.q_norm.weight', 'attn_q_b.weight': 'attn.wq_b.weight',
	'attn_kv.weight': 'attn.wkv.weight', 'attn_kv_a_norm.weight': 'attn.kv_norm.weight',
	'attn_output_a.weight': 'attn.wo_a.weight', 'attn_output_b.weight': 'attn.wo_b.weight', 'attn_sinks.weight': 'attn.attn_sink',
	'attn_norm.weight': 'attn_norm.weight', 'ffn_norm.weight': 'ffn_norm.weight',
	'attn_compressor_kv.weight': 'attn.compressor.wkv.weight', 'attn_compressor_gate.weight': 'attn.compressor.wgate.weight',
	'attn_compressor_norm.weight': 'attn.compressor.norm.weight',
	'indexer.attn_k.weight': 'attn.indexer.wk.weight', 'indexer.attn_q_b.weight': 'attn.indexer.wq_b.weight',
	'indexer.k_norm.weight': 'attn.indexer.k_norm.weight', 'indexer.proj.weight': 'attn.indexer.weights_proj.weight',
	'engram_k.weight': 'engram.k_weight', 'engram_q.weight': 'engram.q_weight', 'engram_wkv.weight': 'engram.wkv.weight',
	'exp_probs_b.bias': 'ffn.gate.bias', 'ffn_gate_inp.weight': 'ffn.gate.weight',
	'ffn_gate_shexp.weight': 'ffn.shared_experts.w1.weight', 'ffn_up_shexp.weight': 'ffn.shared_experts.w3.weight',
	'ffn_down_shexp.weight': 'ffn.shared_experts.w2.weight',
	'ffn_gate_exps.weight': 'ffn.exps.w1', 'ffn_up_exps.weight': 'ffn.exps.w3', 'ffn_down_exps.weight': 'ffn.exps.w2',
	'hc_attn_fn.weight': 'hc_attn_fn', 'hc_attn_base.weight': 'hc_attn_base', 'hc_attn_scale.weight': 'hc_attn_scale',
	'hc_ffn_fn.weight': 'hc_ffn_fn', 'hc_ffn_base.weight': 'hc_ffn_base', 'hc_ffn_scale.weight': 'hc_ffn_scale',
	'engram_embd.weight': 'engram.embed.weight'
};

const TOP = {'token_embd.weight': 'embed.weight', 'output.weight': 'head.weight', 'output_norm.weight': 'norm.weight'};

/* Qwen3.8-Flash-Next (qwen4exp): blk.N.<gguf name> -> layers.N.<hf name> */
const Q4BLK = {
	'attn_qkv.weight': 'linear_attn.in_proj_qkv.weight',
	'attn_gate.weight': 'linear_attn.in_proj_z.weight',
	'ssm_out.weight': 'linear_attn.out_proj.weight',
	'ssm_beta.weight': 'linear_attn.in_proj_b.weight',
	'ssm_alpha.weight': 'linear_attn.in_proj_a.weight',
	'ssm_dt.bias': 'linear_attn.dt_bias',
	'ssm_a': 'linear_attn.A_log',
	'ssm_conv1d.weight': 'linear_attn.conv1d.weight',
	'ssm_norm.weight': 'linear_attn.norm.weight',
	'attn_q.weight': 'self_attn.q_proj.weight', 'attn_k.weight': 'self_attn.k_proj.weight',
	'attn_v.weight': 'self_attn.v_proj.weight', 'attn_output.weight': 'self_attn.o_proj.weight',
	'attn_q_norm.weight': 'self_attn.q_norm.weight', 'attn_k_norm.weight': 'self_attn.k_norm.weight',
	'indexer.q_proj.weight': 'self_attn.indexer.q_proj.weight',
	'indexer.k_proj.weight': 'self_attn.indexer.k_proj.weight',
	'indexer.q_norm.weight': 'self_attn.indexer.q_norm.weight',
	'indexer.k_norm.weight': 'self_attn.indexer.k_norm.weight',
	'ffn_gate_inp.weight': 'mlp.gate.weight',
	'ffn_gate_inp_shexp.weight': 'mlp.shared_expert_gate.weight',
	'ffn_gate_exps.weight': 'mlp.exps.w1', 'ffn_up_exps.weight': 'mlp.exps.w3',
	'ffn_down_exps.weight': 'mlp.exps.w2',
	'ffn_gate_shexp.weight': 'mlp.shared_expert.w1.weight',
	'ffn_up_shexp.weight': 'mlp.shared_expert.w3.weight',
	'ffn_down_shexp.weight': 'mlp.shared_expert.w2.weight',
	'hc_attn_down.weight': 'attn_hyper_connection.input_mix_weight_down.weight',
	'hc_attn_up.weight': 'attn_hyper_connection.input_mix_weight_up.weight',
	'hc_attn_inject.weight': 'attn_hyper_connection.block_inject_weight.weight',
	'hc_attn_norm.weight': 'attn_hyper_connection.hc_norm.weight',
	'hc_ffn_down.weight': 'mlp_hyper_connection.input_mix_weight_down.weight',
	'hc_ffn_up.weight': 'mlp_hyper_connection.input_mix_weight_up.weight',
	'hc_ffn_inject.weight': 'mlp_hyper_connection.block_inject_weight.weight',
	'hc_ffn_norm.weight': 'mlp_hyper_connection.hc_norm.weight',
	'ple_key.weight': 'ple.key_proj.weight', 'ple_value.weight': 'ple.value_proj.weight',
	'ple_conv1d.weight': 'ple.conv1d.weight',
	'ple_norm_conv.weight': 'ple.norm_conv.weight', 'ple_norm_key.weight': 'ple.norm_key.weight',
	'ple_norm_query.weight': 'ple.norm_query.weight'
};

/* qwen4exp MTP block (blk.48.nextn.*) -> the engine's mtp.* keys */
const Q4NEXT = {
	'nextn.eh_proj.weight': 'mtp.eh_proj.weight',
	'nextn.enorm.weight': 'mtp.pre_fc_norm_embedding.weight',
	'nextn.hnorm.weight': 'mtp.pre_fc_norm_hidden.weight',
	'nextn.hc_head_down.weight': 'mtp.hyper_connection_mixer.input_mix_weight_down.weight',
	'nextn.hc_head_up.weight': 'mtp.hyper_connection_mixer.input_mix_weight_up.weight',
	'nextn.hc_head_norm.weight': 'mtp.hyper_connection_mixer.hc_norm.weight'
};

const Q4TOP = {
	'token_embd.weight': 'model.language_model.embed_tokens.weight',
	'output.weight': 'lm_head.weight',
	'per_layer_token_embd.weight': 'ple_table.weight',
	'output_hc_down.weight': 'model.language_model.hyper_connection_mixer.input_mix_weight_down.weight',
	'output_hc_up.weight': 'model.language_model.hyper_connection_mixer.input_mix_weight_up.weight',
	'output_hc_norm.weight': 'model.language_model.hyper_connection_mixer.hc_norm.weight'
};

function hf_name(g, arch) {
	if (arch == 'qwen4exp') {
		if (Q4TOP[g])
			return Q4TOP[g];

		let m = match(g, /^blk\.([0-9]+)\.(.*)$/);

		if (m) {
			if (+m[1] == 48) {
				if (Q4NEXT[m[2]])
					return Q4NEXT[m[2]];

				if (Q4BLK[m[2]])
					return `mtp.layers.0.${Q4BLK[m[2]]}`;

				return null;
			}

			if (Q4BLK[m[2]])
				return `model.language_model.layers.${m[1]}.${Q4BLK[m[2]]}`;
		}

		return null;
	}

	if (TOP[g])
		return TOP[g];

	let m = match(g, /^blk\.([0-9]+)\.(.*)$/);

	if (m && BLK[m[2]])
		return `layers.${m[1]}.${BLK[m[2]]}`;

	return null;
}

/* the GGUF files of a model directory: [backbone shards (sorted), engram file or null] */
export function files(model_dir) {
	let shards = [], engram = null;

	for (let f in sort(lsdir(model_dir) ?? [])) {
		if (!match(f, /\.gguf$/))
			continue;

		if (index(f, 'Engram') >= 0)
			engram = f;
		else
			push(shards, f);
	}

	return [ shards, engram ];
}

/* true when a safetensors file of the directory carries the DSpark drafter (mtp.* tensors, symlinked from an EXL3 pack) */
export function has_drafter(model_dir) {
	for (let f in sort(lsdir(model_dir) ?? [])) {
		if (!match(f, /\.safetensors$/))
			continue;

		for (let k in torch.st(model_dir + '/' + f).list())
			if (k == 'mtp.0.main_proj.weight' || k == 'mtp.0.main_proj.trellis')
				return true;
	}

	return false;
}

/* true when the directory holds a GGUF checkpoint (and no safetensors index / file) */
export function is_gguf_dir(model_dir) {
	let f = files(model_dir);

	return length(f[0]) > 0 && !stat(model_dir + '/model.safetensors.index.json') && !stat(model_dir + '/model.safetensors');
}

export function open(model_dir) {
	let fl = files(model_dir);
	let kmap = {};             // hf key -> [stfile, gguf name]
	let gmap = {};             // gguf name -> [stfile, gguf name] (the tensors that have no hf key: engram_* hash tables)
	let keylist = [];
	let gpu_raw = {}, gpu_cast = {};
	let meta = null, arch = null;

	if (!length(fl[0]))
		die(`ggufw: no .gguf files in ${model_dir}`);

	function add(path, stf) {
		for (let g in stf.list()) {
			let k = hf_name(g, arch);

			gmap[g] = [ stf, g ];

			if (k != null && !kmap[k]) {
				kmap[k] = [ stf, g ];
				push(keylist, k);
			}
		}
	}

	for (let f in fl[0]) {
		let stf = torch.gguf(model_dir + '/' + f);

		if (meta == null) {
			meta = json(stf.meta());
			arch = meta['general.architecture'];
		}

		add(f, stf);
	}

	if (fl[1])
		add(fl[1], torch.gguf(model_dir + '/' + fl[1]));

	// Engram tables and the DSpark drafter (mtp.*) from safetensors files in the directory (the EXL3 / SAGE packs' shards, symlinked
	// here: the tables at the original FP8 + e8m0 scales, so the 110 GB IQ4_NL GGUF is not needed; the drafter in the pack's EXL3 / FP8
	// format, 3.5 GiB); the GGUF's own tensors win if both are present
	for (let f in sort(lsdir(model_dir) ?? [])) {
		if (!match(f, /\.safetensors$/))
			continue;

		let stf = torch.st(model_dir + '/' + f);

		for (let k in stf.list()) {
			if ((index(k, '.engram.embed.') > 0 || index(k, 'mtp.') == 0) && !kmap[k]) {
				kmap[k] = [ stf, k ];
				push(keylist, k);
			}
		}
	}

	function ent(key) {
		let e = kmap[key];

		if (!e)
			die(`ggufw: unknown key '${key}'`);

		return e;
	}

	function dev_of(d) {
		return (d == null) ? 0 : d;
	}

	function qtype(key) {
		let e = kmap[key];

		if (!e)
			return null;

		let t = e[0].info(e[1]).dtype;

		return match(t, /^I?Q[0-9]/) ? t : null;          // (Q8_0, Q5_K, IQ2_XXS, ...; F32 / BF16 / I32 are plain)
	}

	function gpu_for(key, dtype, dev) {
		dev = dev_of(dev);

		if (!dtype) {
			let ck = key + '\x02' + dev;

			if (!gpu_raw[ck])
				gpu_raw[ck] = ent(key)[0].tensor(ent(key)[1]).cuda(dev);

			return gpu_raw[ck];
		}

		let ck = key + '\x01' + dtype + '\x02' + dev;

		if (!gpu_cast[ck]) {
			let raw = gpu_for(key, null, dev);
			let q = qtype(key);

			gpu_cast[ck] = q ? torch.gguf_dequant(raw, q, dtype) : ((raw.dtype() == dtype) ? raw : raw.to(dtype));
		}

		return gpu_cast[ck];
	}

	function upload_all(devfn) {
		let by_dev = {}, devlist = [];

		for (let key in keylist) {
			let dev = devfn ? devfn(key) : 0;

			if (dev == null)
				continue;

			dev = dev_of(dev);

			if (!by_dev[dev]) {
				by_dev[dev] = [];
				push(devlist, dev);
			}

			push(by_dev[dev], [ kmap[key][0], kmap[key][1], key ]);
		}

		for (let dev in devlist) {
			let items = by_dev[dev];
			let ups = torch.h2d_file(map(items, (it) => [ it[0], it[1] ]), dev);

			for (let i = 0; i < length(items); i++)
				gpu_raw[items[i][2] + '\x02' + dev] = ups[i];
		}
	}

	return {
		gguf: true,
		meta: meta,

		has: function (key) {
			return kmap[key] != null;
		},

		file: function (key) {
			return ent(key)[0];
		},

		/* the GGUF tensor name behind a key (and its file) */
		gguf_name: function (key) {
			return ent(key)[1];
		},

		/* the block-quantized type ("IQ2_XXS", "Q5_K", ...) of the tensor under `key`, null for a plain one */
		qtype: qtype,

		tensor: function (key) {
			return ent(key)[0].tensor(ent(key)[1]);
		},

		fp16: function (key) {
			return this.tensor(key).to('float16');
		},

		info: function (key) {
			let e = ent(key);

			return e[0].info(e[1]);
		},

		list: function () {
			return keylist;
		},

		gpu: function (key, dev) {
			return gpu_for(key, null, dev);
		},

		gpu16: function (key, dev) {
			return gpu_for(key, 'float16', dev);
		},

		release: function (key, dev) {
			delete gpu_raw[key + '\x02' + dev_of(dev)];
		},

		drop: function (key, dev) {
			dev = dev_of(dev);

			delete gpu_raw[key + '\x02' + dev];

			for (let dt in [ 'float16', 'float32', 'bfloat16' ])
				delete gpu_cast[key + '\x01' + dt + '\x02' + dev];
		},

		upload: function (devfn) {
			upload_all(devfn);
		},

		/* a tensor without an hf name (engram_primes.weight, ...) from the backbone files, on the CPU */
		raw_tensor: function (name) {
			if (!gmap[name])
				die(`ggufw: no tensor '${name}'`);

			return gmap[name][0].tensor(name);
		},

		/* The Engram hash tables as dsv41.build_engram_meta wants them (the engram_meta.safetensors tensors):
		 * token_map (vocab) int32, multipliers (layers, ngram) int64, primes (layers, heads * (ngram - 1)) int64, offsets the
		 * same, pad: the compressed id of the pad token (a 1-element int64 tensor). */
		engram_meta: function (pad_token_id) {
			let tm = this.raw_tensor('engram_token_map.weight');

			return {
				token_map: tm,
				multipliers: this.raw_tensor('engram_multipliers.weight'),
				primes: this.raw_tensor('engram_primes.weight'),
				offsets: this.raw_tensor('engram_offsets.weight'),
				pad: tm.to('int64').view([-1, 1]).index(torch.from_array([ pad_token_id ], 'int64')).view([-1])
			};
		}
	};
}
