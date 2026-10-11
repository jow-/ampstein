// perf_server.uc — reproduce the ampstein decode path (layer split
// or --tp, MTP on by default) outside the server, with the AMP_PERF
// counters (ucode/engine.uc, ucode/qwen35_tp.uc) printing host-side
// us per op family every few tokens.
//
//   ucode -L build -L ucode -L ucode examples/perf_server.uc --
//       [--model <dir>] [--tp] [--tp-share 0.5] [--mtp 3]
//       [--n 256] [--greedy]

let torch = require('ampstein.torch');
global.torch = torch;
import * as qwen35 from 'ampstein.qwen35';
import * as qwen35_tp from 'ampstein.qwen35_tp';
import * as engine from 'ampstein.engine';
import * as math from 'math';

let args = { model: '/data/models/Qwen3.8-27B-exl3-5.00bpw', tp: false,
	tp_share: 0.5, mtp: 3, n: 256, greedy: false };

let argv = slice(ARGV, index(ARGV, '--') + 1);

for (let i = 0; i < length(argv); i++) {
	switch (argv[i]) {
	case '--model': args.model = argv[i + 1]; i++; break;
	case '--tp': args.tp = true; break;
	case '--tp-share': args.tp_share = +argv[i + 1]; i++; break;
	case '--mtp': args.mtp = int(argv[i + 1]); i++; break;
	case '--n': args.n = int(argv[i + 1]); i++; break;
	case '--greedy': args.greedy = true; break;
	}
}

let cfg = qwen35.load_config(args.model);
let devs = [];
let per = int((cfg.num_hidden_layers + 1) / 2);

for (let i = 0; i < cfg.num_hidden_layers; i++)
	push(devs, (i < per) ? 0 : 1);

warn(`loading ${args.model} (${args.tp ? 'tp' : 'split'})\n`);
let model = qwen35.load(args.model, { devices: devs, mtp: args.mtp > 0 });

if (args.tp)
	model = qwen35_tp.shard(model, { devices: [ 0, 1 ], f0: args.tp_share });

let eng = engine.create(model, { max_ctx: 32768, prefill_chunk: 1024,
	mtp_window: args.mtp });

let ids = [ 151644, 872, 198, 108517, 97377, 25, 100184, 37141, 54071, 323, 96250, 97237, 323, 29, 22502, 5153, 96673, 220, 97237, 37899, 16, 13, 220, 271, 29, 151645, 198, 151644, 872, 198 ];
let t0 = clock(true);
eng.submit({
	ids: ids,
	snap_at: 0,
	max_new: args.n,
	sampling: args.greedy ? { temperature: 0 } : { temperature: 1, top_k: 20, top_p: 0.95 },
	stop_ids: [ 151645, 198 ],

	on_token: function (t) {
		if ((t && (t % 32 == 0)) || t == false)
			warn(`tok ${t ? t : 'END'} @ ${(clock(true)[0] - t0)}s\n`);
		return true;
	},

	on_done: function (info) {
		let t = info.timings;

		if (t)
			warn(sprintf('done: %s %d tok prefill %.0f ms, decode %.0f ms (%.1f tok/s) mtp %s\n',
				info.reason, info.prompt_tokens, t.prefill_ms, t.decode_ms,
				(t.decode_tokens > 0) ? 1000.0 * t.decode_tokens / t.decode_ms : 0,
				info.mtp ? `${info.mtp.rounds}r ${info.mtp.accepted}/${info.mtp.drafted}` : '-'));
	}
});

uloop.init();
uloop.run();
