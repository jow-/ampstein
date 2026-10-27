// ggufmeta.ensure (src/ucode/ampstein/ggufmeta.uc): the side files a GGUF directory lacks are written from the GGUF metadata and
// equal the original Hugging Face files of the checkpoint (config text_config, generation_config, tokenizer.json: vocabulary, merges,
// added tokens and token ids on real text); the chat template is the GGUF's own (it may be newer than the repository's file).
// Needs a Qwen3.8-Flash-Next GGUF directory with the original files next to the shards (AMP_MODEL_QWEN4_GGUF, default
// /data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS).

import { stat, readfile, mkdir, symlink, glob, rmdir, unlink, basename } from 'fs';
import * as ggufmeta from 'ampstein.ggufmeta';

const MODEL = getenv('AMP_MODEL_QWEN4_GGUF') || '/data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS';

if (!stat(MODEL + '/config.json') || !length(glob(MODEL + '/*.gguf'))) {
	warn(`skip: ${MODEL} has no GGUF shards with the original config.json (AMP_MODEL_QWEN4_GGUF)\n`);
	exit(125);
}

let tmp = `${getenv('TMPDIR') || '/tmp'}/ampstein-ggufmeta-${time()}-${getpid ? getpid() : 0}`;

mkdir(tmp);

for (let f in glob(MODEL + '/*.gguf'))
	symlink(f, `${tmp}/${basename(f)}`);

let written = ggufmeta.ensure(tmp);

assert(join(',', sort(written)) == 'chat_template.jinja,config.json,generation_config.json,tokenizer.json', `written ${written}`);
assert(length(ggufmeta.ensure(tmp)) == 0, 'a second call writes nothing');

function J(path) {
	return json(readfile(path));
}

let o = J(MODEL + '/config.json').text_config, g = J(tmp + '/config.json').text_config;

for (let k, v in o)
	assert(sprintf('%J', v) == sprintf('%J', g[k]), `config text_config.${k}: ${sprintf('%J', v)} vs ${sprintf('%J', g[k])}`);

let og = J(MODEL + '/generation_config.json'), gg = J(tmp + '/generation_config.json');

assert(length(keys(og)) == length(keys(gg)), `generation_config keys ${keys(gg)}`);

for (let k, v in og)
	assert(sprintf('%J', v) == sprintf('%J', gg[k]), `generation_config.${k}`);

let a = torch.tokenizer(MODEL), b = torch.tokenizer(tmp);
let texts = [ 'The quick brown fox', '<|im_start|>user\nhi<|im_end|>', '<|file_sep|>x<|fim_prefix|>', '<think>\n</think>',
	'  x\t\n\n  y 123456 ü 日本語 🙂 ', '', 'if (a[i] >= 0) { return "\\n"; }' ];
let n = 0;

for (let t in texts) {
	let x = a.encode(t, false), y = b.encode(t, false);

	assert(sprintf('%J', x) == sprintf('%J', y), `tokens of ${t}`);
	assert(a.decode(x) == b.decode(y), `decoding of ${t}`);
	n += length(x);
}

assert(index(readfile(tmp + '/chat_template.jinja'), 'add_generation_prompt') > 0, 'chat template');

printf('ok: %d files, %d tokens compared\n', length(written), n);

for (let f in glob(tmp + '/*'))
	unlink(f);

rmdir(tmp);
