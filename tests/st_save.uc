// tests/st_save.uc — torch.st_save (write a safetensors file from
// [name, tensor] pairs) round-tripped through torch.st:
//   - fp16 CUDA, fp32 pinned, int64 CPU and bf16 tensors come back
//     bit-exact (values, shapes, dtypes)
//   - dotted/rank-suffixed names (the engine's cache-state layout)
//   - re-saving overwrites a previous file cleanly
//   - duplicate names and bad name characters raise
//
// Needs >= 1 GPU (skips otherwise); no model.

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

const PATH = '/tmp/Ampstein-test-st_save.st';

function names_eq(a, b) {
	if (length(a) != length(b))
		return false;

	for (let i = 0; i < length(a); i++)
		if (a[i] != b[i])
			return false;

	return true;
}

function shape_str(t) {
	let s = t.shape(), out = '';

	for (let i = 0; i < length(s); i++)
		out += (i ? ',' : '') + s[i];

	return out;
}

/* --- 1. round-trip across dtypes and devices ------------------------------ */

{
	let f16 = torch.arange(24, 'float16', 'cuda:0').mul(3.0).reshape([2, 3, 4]);
	let f32 = torch.zeros([5], 'float32').pin();
	f32.fill_(2.5);
	let i64 = torch.from_array([7, 8, 9, 1 << 40], 'int64');
	let bf16 = torch.arange(6, 'float32', 'cuda:0').to('bfloat16');

	torch.st_save(PATH, [
		[ 'l0.r0.k', f16 ],
		[ 'l0.r0.v', f16 ],
		[ 'h', f32 ],
		[ 'ids', i64 ],
		[ 'rec', bf16 ]
	]);

	let st = torch.st(PATH);

	check('roundtrip: names', names_eq(st.list(),
		[ 'l0.r0.k', 'l0.r0.v', 'h', 'ids', 'rec' ]),
		st.list());

	let kf = st.tensor('l0.r0.k').cuda(0);

	check('roundtrip: fp16 exact', kf.sub(f16).abs().max().item() == 0);
	check('roundtrip: fp16 shape', shape_str(kf) == '2,3,4',
		shape_str(kf));

	let hf = st.tensor('h');

	check('roundtrip: fp32 pinned exact', hf.sub(f32).abs().max().item() == 0);

	let idf = st.tensor('ids').tolist();

	check('roundtrip: int64 exact', idf[0] == 7 && idf[1] == 8 &&
		idf[2] == 9 && idf[3] == 1099511627776, idf);

	let bf = st.tensor('rec').cuda(0);

	check('roundtrip: bf16 exact', bf.sub(bf16).abs().max().item() == 0);

	let info = st.info('l0.r0.k');

	check('roundtrip: dtype string', info.dtype == 'F16', info.dtype);
	check('roundtrip: nbytes', info.nbytes == 48, info.nbytes);
}

/* --- 2. overwrite ---------------------------------------------------------- */

{
	let a = torch.full([3], 1.5, 'float16', 'cuda:0');
	let b = torch.full([4], 9.0, 'float16', 'cuda:0');

	torch.st_save(PATH, [ [ 'only', a ] ]);
	torch.st_save(PATH, [ [ 'x', b ] ]);

	let st = torch.st(PATH);

	check('overwrite: fresh content', names_eq(st.list(), [ 'x' ]),
		st.list());
	check('overwrite: values', st.tensor('x').cuda(0).sub(b)
		.abs().max().item() == 0);
}

/* --- 3. rejections ---------------------------------------------------------- */

function raises(name, fn) {
	let ok = false;

	try {
		fn();
	}
	catch (e) {
		ok = true;
	}

	check(name, ok);
}

{
	let t = torch.zeros([2], 'float16', 'cuda:0');

	raises('reject: duplicate name', function () {
		torch.st_save(PATH, [ [ 'a', t ], [ 'a', t ] ]);
	});

	raises('reject: bad name character', function () {
		torch.st_save(PATH, [ [ 'a b', t ] ]);
	});

	raises('reject: entry not a pair', function () {
		torch.st_save(PATH, [ [ 'a', t, t ] ]);
	});

	raises('reject: not a tensor', function () {
		torch.st_save(PATH, [ [ 'a', 42 ] ]);
	});
}

if (fails) {
	warn(fails + ' failures\n');
	exit(1);
}

warn('all green\n');