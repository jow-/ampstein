// Negative tests: every binding must RAISE on wrong argument types and
// counts — never segfault, never silently return null (docs/LESSONS.md
// L54). Table-driven: each case is a
// closure; an exception is a pass, anything else (including a crash,
// which fails the whole test process) is a failure.
//
// Run via CTest or manually:
//   ucode -L <build-dir> tests/bindings_errors.uc

let n = 0;

function expect_raise(what, fn) {
	let raised = false;
	let msg = '';

	try {
		fn();
	} catch (e) {
		raised = true;
		msg = (type(e) == 'object' && e.message != null) ?
			e.message : `${e}`;
	}

	assert(raised, `${what}: expected an exception, got none`);
	assert(length(msg) > 0, `${what}: exception carries no message`);

	n++;
}

/* --- constructors ------------------------------------------------------- */

/* --- host/device copies ---------------------------------------------------- */

expect_raise('copy_h2d_async: no args', function () { torch.copy_h2d_async(); });
expect_raise('copy_h2d_async: one arg', function () { torch.copy_h2d_async(torch.zeros([2])); });
expect_raise('copy_h2d_async: not tensors', function () { torch.copy_h2d_async(1, 2); });
expect_raise('copy_h2d_async: CPU dst', function () {
	torch.copy_h2d_async(torch.zeros([2]), torch.pinned_empty([2]));
});
expect_raise('copy_h2d_async: unpinned src', function () {
	torch.copy_h2d_async(torch.zeros([2], 'float32', 'cuda:0'), torch.zeros([2]));
});
expect_raise('copy_h2d_async: size mismatch', function () {
	torch.copy_h2d_async(torch.zeros([2], 'float32', 'cuda:0'), torch.pinned_empty([3]));
});

expect_raise('zeros: no args', function () { torch.zeros(); });
expect_raise('zeros: negative dim', function () { torch.zeros([-1]); });
expect_raise('zeros: NaN dim', function () { torch.zeros([NaN]); });
expect_raise('zeros: bad dtype', function () { torch.zeros([2], 'nope'); });
expect_raise('zeros: bad device', function () {
	torch.zeros([2], 'float16', 'nope');
});
expect_raise('ones: no args', function () { torch.ones(); });
expect_raise('full: missing value', function () { torch.full([2]); });
expect_raise('full: bad dtype', function () {
	torch.full([2], 1.0, 'nope');
});
expect_raise('arange: no args', function () { torch.arange(); });
expect_raise('arange: bad dtype', function () { torch.arange(4, 'nope'); });

/* from_array: 3-D input used to pass the width check and overflow the
 * destination buffer (heap overflow on 3-D input) */
expect_raise('from_array: 3-D input', function () {
	torch.from_array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
});
expect_raise('from_array: ragged', function () {
	torch.from_array([[1, 2], [3]]);
});
expect_raise('from_array: non-number', function () {
	torch.from_array(['x']);
});
expect_raise('from_array: bad dtype', function () {
	torch.from_array([1], 'nope');
});

/* --- tensor methods ------------------------------------------------------ */

let t = torch.from_array([1, 2, 3]);

expect_raise('item: multi-element', function () { t.item(); });
expect_raise('fill_: no args', function () { t.fill_(); });
expect_raise('fill_: non-number', function () { t.fill_('x'); });
expect_raise('add: wrong type', function () { t.add({}); });
expect_raise('view: no args', function () { t.view(); });

/* --- module-level tensor ops --------------------------------------------- */

expect_raise('matmul: one arg (must raise, not return null)',
	function () { torch.matmul(t); });
expect_raise('where: two args', function () { torch.where(t, t); });
expect_raise('where: non-bool cond', function () {
	torch.where(t, t, t);
});
expect_raise('cat: empty array', function () { torch.cat([]); });
expect_raise('sdpa: two args', function () { torch.sdpa(t, t); });

/* --- stfile ---------------------------------------------------------------- */

expect_raise('st: no args', function () { torch.st(); });
expect_raise('st: bad path', function () {
	torch.st('/nonexistent/path.safetensors');
});

/* --- fused kernels (CUDA) --------------------------------------------------- */

let xc = torch.zeros([2, 8], 'float16', 'cuda');
let wc = torch.zeros([8], 'float16', 'cuda');
let cpu = torch.zeros([2, 8], 'float16');

expect_raise('rms_norm: eps not a number', function () {
	torch.rms_norm(xc, wc, 'x');
});
expect_raise('rms_norm: cpu tensor', function () {
	torch.rms_norm(cpu, wc.cuda(), 1e-5);
});
expect_raise('rms_norm: dim mismatch', function () {
	torch.rms_norm(xc, torch.zeros([4], 'float16', 'cuda'), 1e-5);
});
expect_raise('gdn_ab_gates: wrong arg count', function () {
	torch.gdn_ab_gates(xc, xc, xc);
});
expect_raise('gdn_ab_gates: cpu tensor', function () {
	let w = torch.zeros([2, 8], 'float16', 'cuda');
	let f = torch.zeros([2], 'float32', 'cuda');

	torch.gdn_ab_gates(cpu, w, w, f, f);
});
expect_raise('gdn_ab_gates: K mismatch', function () {
	let w = torch.zeros([2, 8], 'float16', 'cuda');
	let x = torch.zeros([1, 1, 6], 'float16', 'cuda');
	let f = torch.zeros([2], 'float32', 'cuda');

	torch.gdn_ab_gates(x, w, w, f, f);
});
expect_raise('gdn_ab_gates: too many rows', function () {
	let w = torch.zeros([2, 8], 'float16', 'cuda');
	let x = torch.zeros([1, 9, 8], 'float16', 'cuda');
	let f = torch.zeros([2], 'float32', 'cuda');

	torch.gdn_ab_gates(x, w, w, f, f);
});
/* DFlash kernels */
expect_raise('dflash_conv: wrong arg count', function () {
	torch.dflash_conv(xc, xc, xc, 0);
});
expect_raise('dflash_conv: base shape', function () {
	let d = torch.zeros([2, 4 * 8 / 4], 'float16', 'cuda');

	torch.dflash_conv(xc, d, torch.zeros([2, 8], 'float16', 'cuda'), 0, 4);
});
expect_raise('dflash_conv: which out of range', function () {
	let d = torch.zeros([2, 8], 'float16', 'cuda');
	let b = torch.zeros([2, 2, 8], 'float16', 'cuda');

	torch.dflash_conv(xc, d, b, 2, 4);
});
expect_raise('dflash_conv_res: h not fp32', function () {
	let d = torch.zeros([2, 8], 'float16', 'cuda');
	let b = torch.zeros([2, 2, 8], 'float16', 'cuda');

	torch.dflash_conv_res(xc, xc, d, b, 0, 4, 1.0);
});
expect_raise('rms_norm_f32: fp16 input', function () {
	torch.rms_norm_f32(xc, wc, 1e-6);
});
expect_raise('rms_norm_f32: weight size', function () {
	torch.rms_norm_f32(torch.zeros([2, 8], 'float32', 'cuda'),
		torch.zeros([4], 'float16', 'cuda'), 1e-6);
});
expect_raise('dflash_select: wrong arg count', function () {
	torch.dflash_select(xc, xc);
});
expect_raise('dflash_select: cand not int64', function () {
	let u = torch.zeros([2, 4], 'float32', 'cuda');
	let hp = torch.zeros([2, 8], 'float16', 'cuda');
	let cb = torch.zeros([16, 8], 'float16', 'cuda');
	let an = torch.zeros([1], 'int64', 'cuda');

	torch.dflash_select(u, torch.zeros([2, 4], 'float32', 'cuda'), hp, an, cb, cb);
});
expect_raise('rms_norm_res: dim %% 4 != 0', function () {
	torch.rms_norm_res(xc, xc, torch.zeros([6], 'float16', 'cuda'), 1e-5);
});
expect_raise('rms_norm_res: short residual', function () {
	torch.rms_norm_res(xc, torch.zeros([1, 1, 4], 'float16', 'cuda'),
		wc, 1e-5);
});
expect_raise('rope: cos/sin shape mismatch', function () {
	torch.rope(xc, torch.zeros([4, 8], 'float16', 'cuda'),
		torch.zeros([4, 8], 'float16', 'cuda'));
});
expect_raise('silu_mul: size mismatch', function () {
	torch.silu_mul(xc, torch.zeros([3, 8], 'float16', 'cuda'));
});

/* decode_attn with wrong types/counts — these paths used to deref NULL
 * and kill the VM (NULL derefs on bad input) */
expect_raise('decode_attn: no args', function () { torch.decode_attn(); });
expect_raise('decode_attn: non-tensor arg', function () {
	torch.decode_attn(xc, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1.0);
});
expect_raise('decode_attn: pos not int64', function () {
	torch.decode_attn(xc, xc, xc, torch.zeros([2, 4, 8], 'float16', 'cuda'),
		torch.zeros([2, 4, 8], 'float16', 'cuda'), xc, xc, xc, xc, xc, 1.0);
});

expect_raise('argmax_advance: two args', function () {
	torch.argmax_advance(xc, xc);
});
expect_raise('argmax_advance: fp16 state', function () {
	torch.argmax_advance(xc, xc, xc, xc);
});
expect_raise('argmax_advance: undersized ws', function () {
	torch.argmax_advance(xc, xc, xc, xc, torch.zeros([4], 'float32', 'cuda'));
});

expect_raise('mgemm: K not a number', function () {
	torch.mgemm(xc, xc, xc, xc, xc, xc, 'K', false, false);
});
expect_raise('mgemm: nine args (opts cut off)', function () {
	torch.mgemm(xc, xc, xc, xc, xc, xc, 4, false, false);
});

expect_raise('ptr_array: no args', function () { torch.ptr_array(); });
expect_raise('ptr_array: non-integer entry', function () {
	torch.ptr_array([1.5]);
});
expect_raise('ptr_array: string entry', function () {
	torch.ptr_array(['x']);
});

expect_raise('h2d: non-tensor entry', function () { torch.h2d([1]); });
expect_raise('h2d: cuda tensor entry', function () { torch.h2d([xc]); });

expect_raise('exl3_gemm: two args', function () {
	torch.exl3_gemm(xc, xc);
});
expect_raise('exl3_reconstruct: one arg', function () {
	torch.exl3_reconstruct(xc);
});

expect_raise('device_props: no args', function () { torch.device_props(); });
expect_raise('device_props: not an index', function () { torch.device_props('0'); });
expect_raise('device_props: no such device', function () { torch.device_props(999); });

expect_raise('view_dtype: no args', function () { torch.zeros([4], 'uint8').view_dtype(); });
expect_raise('view_dtype: bad dtype', function () { torch.zeros([4], 'uint8').view_dtype('nope'); });
expect_raise('view_dtype: size not a multiple', function () { torch.zeros([3], 'uint8').view_dtype('int64'); });

/* --- fds ---------------------------------------------------------------------- */

expect_raise('fd_write: no args', function () { torch.fd_write(); });
expect_raise('fd_write: fd not an integer', function () { torch.fd_write('3', torch.zeros([2]), 0, 1); });
expect_raise('fd_write: negative fd', function () { torch.fd_write(-1, torch.zeros([2]), 0, 1); });
expect_raise('fd_write: not a tensor', function () { torch.fd_write(1, 'x', 0, 1); });
expect_raise('fd_write: cuda tensor', function () { torch.fd_write(1, xc, 0, 1); });
expect_raise('fd_write: range outside', function () { torch.fd_write(1, torch.zeros([2]), 4, 8); });
expect_raise('fd_write: negative length', function () { torch.fd_write(1, torch.zeros([2]), 0, -1); });
expect_raise('fd_write: not a socket', function () {
	torch.fd_write(require('fs').open('/dev/null', 'w').fileno(), torch.zeros([2]), 0, 4);
});
expect_raise('fd_write: closed fd', function () { torch.fd_write(987, torch.zeros([2]), 0, 4); });
expect_raise('fd_read: three args', function () { torch.fd_read(1, torch.zeros([2]), 0); });
expect_raise('fd_read: bad fd', function () { torch.fd_read(987, torch.zeros([2]), 0, 4); });
expect_raise('fd_send_all: no args', function () { torch.fd_send_all(); });
expect_raise('fd_send_all: bad timeout', function () { torch.fd_send_all(1, torch.zeros([2]), -5); });
expect_raise('fd_send_all: non-contiguous', function () {
	torch.fd_send_all(1, torch.zeros([4, 4]).transpose(0, 1));
});
expect_raise('fd_recv_all: bad fd', function () { torch.fd_recv_all(987, torch.zeros([2])); });
expect_raise('fd_recv_all: too many args', function () { torch.fd_recv_all(1, torch.zeros([2]), 1, 2); });

/* --- graphs ----------------------------------------------------------------- */

expect_raise('graph.replay: zero count', function () {
	let g = torch.graph();

	g.replay(0);
});

/* --- host-resident experts ------------------------------------------------------ */

expect_raise('read_into: not an array', function () { torch.read_into(1); });
expect_raise('read_into: bad item', function () { torch.read_into([ [ 1, 2 ] ]); });
expect_raise('moe_cpu_make_layer: no args', function () { torch.moe_cpu_make_layer(); });
expect_raise('moe_cpu_make_layer: not arrays', function () {
	torch.moe_cpu_make_layer(1, 2, 3, 4, 5, 6, 7, 8, 9, 1.0);
});
expect_raise('moe_cpu_make_layer: no experts', function () {
	torch.moe_cpu_make_layer([], [], [], [], [], [], [], [], [], 1.0);
});
expect_raise('moe_cpu_forward: no args', function () { torch.moe_cpu_forward(); });
expect_raise('moe_cpu_forward: bad tensors', function () {
	torch.moe_cpu_forward(0, torch.zeros([1, 8]), torch.zeros([1, 2]), torch.zeros([1, 2]), torch.zeros([1, 8]));
});
expect_raise('moe_cpu_enqueue: no args', function () { torch.moe_cpu_enqueue(); });
expect_raise('moe_cpu_threads: no args', function () { torch.moe_cpu_threads(); });
expect_raise('moe_cpu_threads: too many', function () { torch.moe_cpu_threads(65); });
expect_raise('moe_cpu_prof: negative', function () { torch.moe_cpu_prof(-1); });
expect_raise('moe_cpu_prof: phases not boolean', function () { torch.moe_cpu_prof(10, 1); });
expect_raise('expert_exchange: no args', function () { torch.expert_exchange(); });
expect_raise('expert_exchange: bad slot', function () { torch.expert_exchange(0, 9, [], [], [], [], [], [], []); });
expect_raise('expert_exchange: unpaired copy', function () {
	torch.expert_exchange(0, 2, [ torch.zeros([2], 'float32', 'cuda:0') ], [], [], [], [], [], []);
});
expect_raise('expert_exchange: unpinned source', function () {
	torch.expert_exchange(0, 2, [ torch.zeros([2], 'float32', 'cuda:0') ], [ torch.zeros([2]) ], [], [], [], [], []);
});
expect_raise('expert_exchange: bad poke', function () { torch.expert_exchange(0, 2, [], [], [], [], [], [], [ [ 3, 0, 8 ] ]); });
expect_raise('expert_exchange: bad poke size', function () {
	torch.expert_exchange(0, 2, [], [], [], [], [], [], [ [ 4096, 0, 2 ] ]);
});

printf('bindings_errors: %d wrong-argument cases all raised\n', n);