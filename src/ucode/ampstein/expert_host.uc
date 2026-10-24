// ucode/expert_host.uc — MoE experts kept in pinned host memory (qwen4_tp host_experts).
//
// Decode reads a cold expert's trellis straight from host memory through the pointer tables of the fused
// kernels (a miss costs its transfer, a hit nothing). A prefill chunk reads every expert, and each row tile
// of each launch reads its weights again, so the same zero-copy reads cost several times the bytes of one
// pass over the link. For chunks of STAGE_MIN rows and more the cold experts of a layer are therefore
// copied once, in bulk, into a per-device double buffer on a side stream — one layer ahead, behind the
// previous layer's compute — and the fused kernels run on a second set of pointer tables (`M.stage.pf`)
// that point at the buffer. M.stage is null for a layer without cold experts, and nothing here runs for a
// model without host experts.
//
// M.stage = {dev, arena: [gate, up, down] (nc, a, b, c) pinned trellis of the cold experts, nc, par (buffer
//            parity), nst (slots [0, nst) are staged, the rest computed on the CPU: CPU experts with a split; default nc),
//            bufs (set by the loader), bytes (per expert and matrix), next (the next layer's M on the
//            same device), copied (a copy into bufs is queued), on (this layer runs on the staged tables),
//            pf / dec (the staged and the decode tables of M.coop), hv (staged MLPs of heavy experts, by index)}

const SLOT = 0;       // stream_side slot: glm5 forks the shared expert on 1, the all-reduces use 2 and 3

const NO_PREFETCH = getenv('AMP_EXPERT_NO_PREFETCH') != null;     // debugging: every layer copies its own experts synchronously

export const STAGE_MIN = +(getenv('AMP_EXPERT_STAGE_MIN') || 128);

// AMP_EXPERT_STAGE_BUFS=1: one staging buffer per device instead of two (half the VRAM, for more resident experts): a
// layer's copy then starts at the layer itself, behind the previous layer's MoE, and overlaps only its attention
export const BUFS = (getenv('AMP_EXPERT_STAGE_BUFS') == '1') ? 1 : 2;

// queue the copy of the cold experts of st's layer into its buffer, behind the work already queued on the
// device (the layer that last read this buffer has been launched by then)
function issue(st) {
	if (!(st.nst ?? st.nc)) {
		st.copied = true;

		return;
	}

	torch.stream_side(st.dev, SLOT);

	for (let m = 0; m < 3; m++)
		torch.copy_h2d_async(st.bufs[m].narrow(0, 0, st.nst ?? st.nc), st.arena[m].narrow(0, 0, st.nst ?? st.nc));

	torch.stream_main(st.dev, SLOT);
	st.copied = true;
}

// the start of a layer of a prefill-sized chunk: its cold experts are staged (queued now unless the previous
// layer already did), the compute stream waits for them, and the next layer's copy is queued behind this one.
// A group of chunks that runs layer by layer (gid: its number) stages the layer once: a later chunk of the same
// group only switches the layer's tables to the buffer again.
export function begin(M, gid) {
	let st = M.stage;

	if (gid != null && st.gid === gid) {
		st.on = true;

		return;
	}

	st.gid = gid;

	if (!st.copied)
		issue(st);

	torch.stream_wait(st.dev, SLOT);
	st.copied = false;
	st.on = true;

	if (st.next && !NO_PREFETCH && BUFS == 2 && !st.next.stage.copied)
		issue(st.next.stage);
}

// the start of a staged prefill: a stage's GPUs idle until the first chunk reaches them, so the first layers of every
// stage (both buffers) are copied ahead — M: the stage's first layer on a device
export function prestage(M) {
	let st = M.stage;

	if (!st || st.copied)
		return;

	issue(st);

	if (st.next && BUFS == 2 && !NO_PREFETCH && !st.next.stage.copied)
		issue(st.next.stage);
}

// the end of a prefill: copies queued ahead that no chunk used are dropped (the arena may change before the next
// prefill: expert_dyn exchanges)
export function unstage(M) {
	let st = M.stage;

	if (st)
		st.copied = false;
}

// the pointer tables of M.coop with the cold experts' trellis repointed into the staging buffer
export function tables(M) {
	let st = M.stage, dv = 'cuda:' + M.dev;

	function tbl(name, m) {
		let a = [];
		let base = st.bufs[m].data_ptr();

		for (let ex in M.experts)
			// (a slot past nst is not staged: the CPU computes that expert's picks, expert_dyn / glm5.cpu_experts_begin)
			push(a, (ex.slot == null || ex.slot >= (st.nst ?? st.nc)) ? ex[name].trellis.data_ptr() : base + ex.slot * st.bytes[m]);

		return torch.from_array(a, 'int64', dv);
	}

	return { ...M.coop, g_trellis: tbl('gate', 0), u_trellis: tbl('up', 1), d_trellis: tbl('down', 2) };
}

// expert e of M as a gated MLP that reads the staged copy of its trellis (the composite path for experts with
// many rows); a resident expert is returned as is
export function mlp(M, e) {
	let st = M.stage, ex = M.experts[e];

	// (a slot that is not staged, past nst: the expert's own forward reads the arena)
	if (ex.slot == null || ex.slot >= (st.nst ?? st.nc))
		return ex;

	if (!st.hv)
		st.hv = {};

	if (!st.hv[e]) {
		let o = {};
		let names = ['gate', 'up', 'down'];

		for (let m = 0; m < 3; m++) {
			let l = ex[names[m]];

			o[names[m]] = { ...l, dev_trellis: st.bufs[m].narrow(0, ex.slot, 1).view(l.trellis.shape()) };
		}

		st.hv[e] = o;
	}

	return st.hv[e];
}

// the loader: one pair of staging buffers per device (sized for its largest layer), buffers handed to the
// layers by parity, each layer linked to the next one on the same device
export function setup(layers, nl) {
	let cap = {};

	for (let i = 0; i < nl; i++)
		for (let R in layers[i]) {
			let st = R.moe.stage;

			if (st && (st.nst ?? st.nc) > (cap[st.dev] || 0))
				cap[st.dev] = st.nst ?? st.nc;
		}

	let pool = {}, total = 0;

	for (let i = 0; i < nl; i++)
		for (let R in layers[i]) {
			let st = R.moe.stage;

			if (!st)
				continue;

			if (!pool[st.dev]) {
				pool[st.dev] = [];

				for (let par = 0; par < BUFS; par++) {
					let b = [];

					for (let m = 0; m < 3; m++) {
						let sh = st.arena[m].shape();

						push(b, torch.empty([cap[st.dev], sh[1], sh[2], sh[3]], 'int16', 'cuda:' + st.dev));
						total += cap[st.dev] * sh[1] * sh[2] * sh[3] * 2;
					}

					push(pool[st.dev], b);
				}
			}

			st.bufs = pool[st.dev][st.par % BUFS];
			st.bytes = map(st.arena, (t) => t.numel() * 2 / st.nc);

			if (i + 1 < nl)
				for (let R2 in layers[i + 1])
					if (R2.moe.stage && R2.moe.dev == R.moe.dev)
						st.next = R2.moe;
		}

	return total;
}
