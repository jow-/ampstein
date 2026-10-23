// examples/rpc_demo.uc — the cross-host transport (rpc.uc) end to end: a coordinator
// that waits for a worker, then times pings and tensor echoes through the framed connection.
//
//   coordinator:  ucode -L ... -ltorch=ampstein.torch examples/rpc_demo.uc -- \
//                     --listen ADDR:PORT --model DIR [--sizes 40000,100000] [--iters 500]
//   worker:       ucode -L ... -ltorch=ampstein.torch examples/rpc_demo.uc -- \
//                     --connect ADDR:PORT --model DIR [--name j7]
//
// --model is the model directory (the same copy on both hosts; its manifest is
// compared in the handshake). Bind to the address of the link, not 0.0.0.0: there is
// no TLS or authentication (docs/RPC.md).

import * as rpc from 'ampstein.rpc';

let listen = null, connect = null, model = null, name = null;
let sizes = [ 64, 40000, 100000, 2000000 ], iters = 500;
let argv = ARGV, st = index(argv, '--');

if (st >= 0)
	argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	let a = argv[i];

	if (a == '--listen') listen = argv[++i];
	else if (a == '--connect') connect = argv[++i];
	else if (a == '--model') model = argv[++i];
	else if (a == '--name') name = argv[++i];
	else if (a == '--sizes') sizes = map(split(argv[++i], ','), (x) => int(x));
	else if (a == '--iters') iters = int(argv[++i]);
	else
		die(`unknown option ${a}\n`);
}

if (!model || (!listen && !connect))
	die('usage: --listen ADDR:PORT --model DIR | --connect ADDR:PORT --model DIR [--name N]\n');

let man = rpc.manifest(model);

warn(`manifest: ${length(man.files)} files, strong ${man.strong ? substr(man.strong, 0, 16) : 'n/a (no digest module)'}\n`);

function now() {
	let t = clock(true);

	return t[0] * 1000000.0 + t[1] / 1000.0;
}

function pct(v, p) {
	sort(v, (a, b) => a - b);

	return v[int((length(v) - 1) * p)];
}

if (listen) {
	let m = match(listen, /^(.*):([0-9]+)$/);
	let L = rpc.listen({ bind: m[1], port: int(m[2]), manifest: man });

	warn(`waiting for a worker on ${m[1]}:${L.port}\n`);

	let w = L.wait(1, 600)[0];
	let c = w.conn;

	warn(`worker ${w.name}: ${join(', ', map(w.hello.gpus, (g) => `${g.name} ${g.cc} ${sprintf('%.0f', g.total / 1073741824)} GiB`))}\n`);

	let rt = [];

	for (let i = 0; i < 200; i++)
		push(rt, c.ping().us);

	printf('%-10s %9s %9s %9s\n', 'frame', 'p50 us', 'p99 us', 'max us');
	printf('%-10s %9.0f %9.0f %9.0f\n', 'ping', pct(rt, 0.5), pct(rt, 0.99), rt[length(rt) - 1]);

	for (let size in sizes) {
		let tx = torch.pinned_empty([ size ], 'uint8');
		let n = (size >= 2000000) ? 100 : iters;
		let rt = [];

		for (let i = 0; i < n + 10; i++) {
			let t0 = now();

			c.send({ t: 'echo', n: i }, tx);

			let f = c.recv(10000);

			if (!f || f.meta.t != 'echo' || f.tensor.numel() != size)
				die('echo failed\n');

			if (i >= 10)
				push(rt, now() - t0);
		}

		printf('%-10d %9.0f %9.0f %9.0f\n', size, pct(rt, 0.5), pct(rt, 0.99), rt[length(rt) - 1]);
	}

	c.send({ t: 'bye' });
	L.close();
}
else {
	rpc.worker({
		connect: connect, name: name ?? 'worker', manifest: man, once: true,
		serve: function (c, w) {
			for (;;) {
				let f = c.recv(-1);

				if (f.meta.t == 'bye')
					return 'bye';

				if (f.meta.t == 'echo')
					c.send(f.meta, f.tensor);
			}
		}
	});
}
