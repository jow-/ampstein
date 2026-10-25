// GGUF reader (torch.gguf, src/torch/gguf.c): metadata, tensor inventory, shapes (row-major), block types as raw uint8
// rows, plain types as themselves, and refusal of malformed files. CPU only.
// Fixture: tests/fixtures/gguf_small.gguf (tests/oracle/make_gguf_dequant.py).

import { stat, readfile, writefile, unlink } from 'fs';

const FIXTURE = `${sourcepath(0, true)}/fixtures/gguf_small.gguf`;

if (!stat(FIXTURE)) {
	warn('skip: tests/fixtures/gguf_small.gguf missing (tests/oracle/make_gguf_dequant.py)\n');
	exit(125);
}

let f = torch.gguf(FIXTURE);
let m = json(f.meta());

assert(m['general.architecture'] == 'deepseek41', 'architecture');
assert(m['test.u32'] == 4000000000 && m['test.i32'] == -7 && m['test.u8'] == 200, 'integer metadata');
assert(m['test.i64'] == -(1 << 40), 'int64 metadata');
assert(m['test.f32'] == 1.5 && m['test.f64'] == 2.25 && m['test.bool'] === true, 'float / bool metadata');
assert(m['test.str'] == 'ampstein ü', 'string metadata');
assert(join(',', m['test.arr_i']) == '1,2,3' && join('|', m['test.arr_s']) == 'a|bc|' && m['test.arr_f'][1] == 0.25, 'arrays');

let names = sort(f.list());

assert(join(',', names) == 'iq1_m,iq1_s,iq2_xs,iq2_xxs,iq4_nl,plain_bf16,plain_f32,plain_i32,q5_k,q8_0,stack', `inventory ${names}`);

// shape (row-major), logical dtype, byte count
function check_info(n, dtype, shape, nbytes) {
	let i = f.info(n);

	assert(i.dtype == dtype, `${n} dtype ${i.dtype}`);
	assert(join('x', i.shape) == shape, `${n} shape ${i.shape}`);
	assert(i.nbytes == nbytes, `${n} nbytes ${i.nbytes}`);
}

check_info('q8_0', 'Q8_0', '24x768', 24 * 24 * 34);
check_info('iq4_nl', 'IQ4_NL', '24x768', 24 * 24 * 18);
check_info('q5_k', 'Q5_K', '24x768', 24 * 3 * 176);
check_info('iq2_xxs', 'IQ2_XXS', '24x768', 24 * 3 * 66);
check_info('iq2_xs', 'IQ2_XS', '24x768', 24 * 3 * 74);
check_info('iq1_m', 'IQ1_M', '24x768', 24 * 3 * 56);
check_info('iq1_s', 'IQ1_S', '24x768', 24 * 3 * 50);
check_info('stack', 'IQ2_XXS', '5x6x512', 5 * 6 * 2 * 66);
check_info('plain_f32', 'F32', '3x8', 96);
check_info('plain_bf16', 'BF16', '4x6', 48);
check_info('plain_i32', 'I32', '10', 40);

// block types come as raw uint8 rows (row bytes last), plain types as themselves
let t = f.tensor('iq2_xxs');

assert(t.dtype() == 'uint8' && join('x', t.shape()) == '24x198', `iq2_xxs tensor ${t.dtype()} ${t.shape()}`);
t = f.tensor('stack');
assert(t.dtype() == 'uint8' && join('x', t.shape()) == '5x6x132', `stack tensor ${t.shape()}`);
t = f.tensor('plain_i32');
assert(t.dtype() == 'int32' && join(',', t.tolist()) == '0,1,2,3,4,5,6,7,8,9', 'plain i32 values');
t = f.tensor('plain_bf16');
assert(t.dtype() == 'bfloat16' && join('x', t.shape()) == '4x6', 'plain bf16');

// the file bytes are what the view shows: a row of q8_0 (its first 32 bytes) is found in the file, once
let raw = readfile(FIXTURE);
let q = f.tensor('q8_0').tolist()[0];

assert(length(q) == 24 * 34, 'q8_0 row bytes');

let needle = '';

for (let i = 0; i < 32; i++)
	needle += chr(q[i]);

let at = index(raw, needle);

assert(at > 0 && index(raw, needle, at + 1) < 0, 'q8_0 row bytes found once in the file');

// refusals: a missing tensor, wrong argument types, other files
let raised = (fn) => { try { fn(); } catch (e) { return true; } return false; };

assert(raised(() => f.info('nope')), 'unknown tensor raises');
assert(raised(() => f.tensor('nope')), 'unknown tensor raises (tensor)');
assert(raised(() => torch.gguf()), 'no path raises');
assert(raised(() => torch.gguf(42)), 'non-string path raises');
assert(raised(() => torch.gguf('/nonexistent/x.gguf')), 'missing file raises');

const TMP = '/tmp/ampstein_gguf_test.gguf';

// truncated: header cut inside the metadata, cut inside the tensor data, a bad magic, an absurd tensor count
writefile(TMP, substr(raw, 0, 200));
assert(raised(() => torch.gguf(TMP)), 'truncated header raises');
writefile(TMP, substr(raw, 0, length(raw) - 100));
assert(raised(() => torch.gguf(TMP)), 'truncated data raises');
writefile(TMP, 'GGUX' + substr(raw, 4));
assert(raised(() => torch.gguf(TMP)), 'bad magic raises');
writefile(TMP, substr(raw, 0, 8) + '\xff\xff\xff\xff\xff\xff\xff\x7f' + substr(raw, 16));
assert(raised(() => torch.gguf(TMP)), 'absurd tensor count raises');
writefile(TMP, '');
assert(raised(() => torch.gguf(TMP)), 'empty file raises');
unlink(TMP);

// a safetensors file is not a GGUF file, and the other way round
assert(raised(() => torch.st(FIXTURE)), 'st() refuses a GGUF file');

print('gguf_reader: all ok\n');
