// ucode/ggufmeta.uc — the HF side files of a GGUF checkpoint, made from the GGUF's own metadata.
//
// A GGUF holds the hyperparameters, the tokenizer (BPE tokens, merges, token types) and the chat template, but the loaders read
// config.json, generation_config.json, chat_template.jinja and tokenizer.json from the model directory (the layout of the
// original Hugging Face model). When any of them is missing, ensure() writes it from the metadata of the first shard, so a
// directory with only the .gguf files starts. Files that exist are never touched (a checkpoint's own files win).
//
// Qwen3.8-Flash-Next (general.architecture "qwen4exp") only; the values the GGUF does not carry (vision tower, MTP block
// layout, n-gram hash split) are the family's constants. tests/ggufmeta.uc compares the result with the original files.

import {writefile, stat, rename} from 'fs';
import * as ggufw from './ggufw.uc';

// the split regex the GGUF names with tokenizer.ggml.pre
const PRETOKENIZE = {
	qwen35: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
};

// added tokens of the Qwen vocabulary that the original tokenizer.json does not mark special (the GGUF calls every one of them
// "control"): code-completion and repository markers, which stay atomic in the input but are not hidden when decoding
const NOT_SPECIAL = [ '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>' ];

// llama.cpp token types
const T_NORMAL = 1, T_CONTROL = 3, T_USER = 4;

function put(path, text) {
	writefile(path + '.tmp', text);
	rename(path + '.tmp', path);
}

function round7(v) {
	return +sprintf('%.6g', v) + 0.0;
}

function qwen4_config(m) {
	let a = 'qwen4exp.';
	let cr = m[a + 'attention.compress_ratios'];
	let layers = [], ratio = 0;

	for (let i = 0; i < m[a + 'block_count']; i++) {
		push(layers, cr[i] > 0 ? 'full_attention' : 'linear_attention');

		if (cr[i] > 0)
			ratio = cr[i];
	}

	let img = m[a + 'ple.image_token_id'];
	let rope = m[a + 'rope.freq_base'];
	let hd = m[a + 'attention.key_length'];

	return {
		architectures: [ 'Qwen4ExpForConditionalGeneration' ],
		image_token_id: img,
		language_model_only: true,
		model_type: 'qwen4_exp',
		tie_word_embeddings: false,
		transformers_version: '5.8.0.dev0',
		video_token_id: img + 1,
		vision_end_token_id: img - 2,
		vision_start_token_id: img - 3,
		text_config: {
			attention_bias: false,
			attention_dropout: 0.0,
			bos_token_id: m['tokenizer.ggml.bos_token_id'],
			dtype: 'bfloat16',
			eos_token_id: m['tokenizer.ggml.bos_token_id'],
			full_attention_interval: m[a + 'full_attention_interval'],
			hc_count: m[a + 'hyper_connection.count'],
			hc_lowrank: m[a + 'hyper_connection.low_rank'],
			head_dim: hd,
			heads_per_ngram: m[a + 'ple.heads_per_ngram'],
			hidden_act: 'silu',
			hidden_size: m[a + 'embedding_length'],
			indexer_budget: m[a + 'attention.indexer.top_k'],
			indexer_compress_ratio: ratio,
			indexer_head_dim: m[a + 'attention.indexer.key_length'],
			indexer_kv_heads: 1,
			indexer_n_heads: m[a + 'attention.indexer.head_count'],
			initializer_range: 0.02,
			layer_types: layers,
			linear_conv_kernel_dim: m[a + 'ssm.conv_kernel'],
			linear_key_head_dim: m[a + 'ssm.state_size'],
			linear_num_key_heads: m[a + 'ssm.group_count'],
			linear_num_value_heads: m[a + 'ssm.time_step_rank'],
			linear_value_head_dim: m[a + 'ssm.inner_size'] / m[a + 'ssm.time_step_rank'],
			make_ngram_vocab_size_divisible_by: 128,
			mamba_ssm_dtype: 'float32',
			max_position_embeddings: m[a + 'context_length'],
			model_type: 'qwen4_exp_text',
			moe_intermediate_size: m[a + 'expert_feed_forward_length'],
			mtp: {hybrid: true, layer_types: [ 'full_attention' ], mtp_use_hidden_state_from_layer: null, num_hidden_layers: 1, rope_theta: int(rope)},
			mtp_num_hidden_layers: 1,
			mtp_use_dedicated_embeddings: false,
			ngram_size: m[a + 'ple.ngram_size'],
			ngram_vocab_size_base: int(m[a + 'ple.head_vocab_sizes'][0] / 1000000) * 1000000,
			num_attention_heads: m[a + 'attention.head_count'],
			num_experts: m[a + 'expert_count'],
			num_experts_per_tok: m[a + 'expert_used_count'],
			num_hidden_layers: m[a + 'block_count'],
			num_key_value_heads: m[a + 'attention.head_count_kv'],
			output_gate_type: 'sigmoid',
			output_router_logits: false,
			pad_token_id: null,
			partial_rotary_factor: m[a + 'rope.dimension_count'] / (hd + 0.0),
			ple_conv_kernel_size: m[a + 'ple.conv_kernel'],
			ple_embed_dim: m[a + 'embedding_length'],
			ple_layer_ids: map(m[a + 'ple.layers'], (l) => l + 1),
			rms_norm_eps: round7(m[a + 'attention.layer_norm_rms_epsilon']),
			rope_parameters: {mrope_interleaved: true, mrope_section: slice(m[a + 'rope.dimension_sections'], 0, 3),
				partial_rotary_factor: m[a + 'rope.dimension_count'] / (hd + 0.0), rope_theta: int(rope), rope_type: 'default'},
			router_aux_loss_coef: 0.001,
			shared_expert_intermediate_size: m[a + 'expert_shared_feed_forward_length'],
			split_ngram_parts: 128,
			tie_word_embeddings: false,
			use_cache: true,
			vocab_size: length(m['tokenizer.ggml.tokens'])
		}
	};
}

function generation_config(m) {
	let g = {bos_token_id: m['tokenizer.ggml.bos_token_id'], do_sample: true,
		eos_token_id: [ m['tokenizer.ggml.eos_token_id'], m['tokenizer.ggml.bos_token_id'] ],
		pad_token_id: m['tokenizer.ggml.padding_token_id']};

	if (m['general.sampling.temp'] != null)
		g.temperature = round7(m['general.sampling.temp']);

	if (m['general.sampling.top_k'] != null)
		g.top_k = m['general.sampling.top_k'];

	if (m['general.sampling.top_p'] != null)
		g.top_p = round7(m['general.sampling.top_p']);

	return g;
}

/* tokenizer.json: the byte-level BPE of the GGUF (tokenizer.ggml.model "gpt2"). Ids from the first token that is not a plain one
 * on are "added tokens" (control tokens special, user-defined ones not); unused slots (padding of the embedding) are left out. */
function tokenizer_json(m) {
	let toks = m['tokenizer.ggml.tokens'], types = m['tokenizer.ggml.token_type'];
	let rx = PRETOKENIZE[m['tokenizer.ggml.pre']];

	if (m['tokenizer.ggml.model'] != 'gpt2' || !rx)
		die(`ggufmeta: tokenizer ${m['tokenizer.ggml.model']} / ${m['tokenizer.ggml.pre']} is not supported`);

	let base = 0, vocab = {}, added = [];

	while (base < length(toks) && types[base] == T_NORMAL)
		base++;

	for (let i = 0; i < base; i++)
		vocab[toks[i]] = i;

	for (let i = base; i < length(toks); i++)
		if (types[i] == T_CONTROL || types[i] == T_USER)
			push(added, {id: i, content: toks[i], single_word: false, lstrip: false, rstrip: false, normalized: false,
				special: types[i] == T_CONTROL && index(NOT_SPECIAL, toks[i]) < 0});

	let bl = {type: 'ByteLevel', add_prefix_space: false, trim_offsets: false, use_regex: false};

	return {
		version: '1.0', truncation: null, padding: null, added_tokens: added,
		normalizer: {type: 'NFC'},
		pre_tokenizer: {type: 'Sequence', pretokenizers: [
			{type: 'Split', pattern: {Regex: rx}, behavior: 'Isolated', invert: false}, bl ]},
		post_processor: bl, decoder: bl,
		model: {type: 'BPE', dropout: null, unk_token: null, continuing_subword_prefix: '', end_of_word_suffix: '',
			fuse_unk: false, byte_fallback: false, ignore_merges: false, vocab: vocab, merges: m['tokenizer.ggml.merges']}
	};
}

/* The side files `model_dir` lacks, written from the GGUF metadata. Returns the names written. */
export function ensure(model_dir) {
	const files = {
		'config.json': (m) => qwen4_config(m),
		'generation_config.json': (m) => generation_config(m),
		'tokenizer.json': (m) => tokenizer_json(m),
		'chat_template.jinja': (m) => m['tokenizer.chat_template']
	};
	let todo = [];

	for (let f in keys(files))
		if (!stat(`${model_dir}/${f}`))
			push(todo, f);

	if (!length(todo))
		return [];

	let m = ggufw.open(model_dir).meta;

	if (m['general.architecture'] != 'qwen4exp')
		return [];

	// (a lone drafter shard or a truncated download has the architecture but not the model)
	for (let k in [ 'qwen4exp.block_count', 'qwen4exp.expert_count', 'qwen4exp.embedding_length', 'qwen4exp.attention.compress_ratios',
	    'qwen4exp.ple.head_vocab_sizes', 'tokenizer.ggml.tokens', 'tokenizer.ggml.merges', 'tokenizer.ggml.token_type' ]) {
		if (m[k] == null) {
			warn(`ampstein: ${model_dir}: the GGUF metadata has no ${k}, so the side files cannot be made from it ` +
				'(are all shards of the model present? a directory with only the mtp-*.gguf cannot be served)\n');
			exit(1);
		}
	}

	for (let f in todo) {
		let v = files[f](m);

		if (v == null)
			continue;

		put(`${model_dir}/${f}`, (type(v) == 'string') ? v : sprintf('%J', v));
	}

	warn(`ampstein: ${model_dir} had no ${join(', ', todo)}: written from the GGUF metadata\n`);

	return todo;
}
