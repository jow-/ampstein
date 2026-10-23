// examples/rpc_pingpong.uc — phase 0 of docs/RPC.md: round trips over the link
// that would carry a pipeline-stage hand-off.
//
//   host B:  ucode -L ... examples/rpc_pingpong.uc -- --serve 7700 [--bind ADDR]
//   host A:  ucode -L ... examples/rpc_pingpong.uc -- --connect HOST:7700 \
//                [--sizes 64,40000,100000,20000000] [--iters 2000] [--gpu 0]
//
// The client sends N bytes, the server echoes them; the report is the round
// trip (p50 / p99 / max) per size, one way being about half. Run it between
// the two real hosts, at the quiet and the busy time of the day: the p99
// decides (go if it stays under ~2 ms at the activation size).
//
// --gpu D adds the device side: a D2H copy into a pinned buffer plus a
// synchronize, and the H2D back, of each size on device D, so the copy cost
// that the hop adds (separate from the wire) is measured on that machine.
// The wire part uses plain strings; the native send / receive bindings of
// phase 1 will not be slower than this.
//
// --native moves the bytes with torch.fd_send_all / fd_recv_all between pinned
// tensors (phase 1 bindings) instead of ucode strings; give it to both sides.
//
// No TLS, no authentication: bind to the link's address (--bind), not 0.0.0.0.

import * as socket from 'socket';

let mode = null, bind = '127.0.0.1', port = 7700, host = null;
let sizes = [ 64, 40000, 100000, 2000000, 20000000 ];
let iters = 2000, gpu = null, native = false;
let argv = ARGV, st = index(argv, '--');

if (st >= 0)
	argv = slice(argv, st + 1);

for (let i = 0; i < length(argv); i++) {
	let a = argv[i];

	if (a == '--serve') {
		mode = 'serve';
		port = int(argv[++i]);
	}
	else if (a == '--connect') {
		mode = 'connect';

		let m = match(argv[++i], /^(.*):([0-9]+)$/);

		if (!m)
			die('--connect HOST:PORT expected\n');

		host = m[1];
		port = int(m[2]);
	}
	else if (a == '--bind') bind = argv[++i];
	else if (a == '--sizes') sizes = map(split(argv[++i], ','), (x) => int(x));
	else if (a == '--iters') iters = int(argv[++i]);
	else if (a == '--gpu') gpu = int(argv[++i]);
	else if (a == '--native') native = true;
	else
		die(`unknown option ${a}\n`);
}

if (!mode)
	die('usage: --serve PORT [--bind ADDR] | --connect HOST:PORT [--sizes a,b] [--iters N] [--gpu D] [--native]\n');

function now() {
	let t = clock(true);

	return t[0] * 1000000.0 + t[1] / 1000.0;      // us
}

function read_n(s, n) {
	let parts = [], got = 0;

	while (got < n) {
		let c = s.recv(n - got);

		if (c == null || length(c) == 0)
			return null;

		push(parts, c);
		got += length(c);
	}

	return join('', parts);
}

function write_all(s, data) {
	let off = 0, n = length(data);

	while (off < n) {
		let w = s.send(off ? substr(data, off) : data);

		if (w == null || w <= 0)
			return false;

		off += w;
	}

	return true;
}

function nodelay(s) {
	s.setopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1);
}

let torch = null;

if (gpu != null || native) {
	torch = require('ampstein.torch');
	torch.synchronize();
}

// native framing: an int64 length, then that many bytes; echoed whole
function native_echo(fd) {
	let hdr = torch.zeros([1], 'int64'), buf = null, cap = 0;

	for (;;) {
		try {
			torch.fd_recv_all(fd, hdr);
		}
		catch (e) {
			return;
		}

		let n = hdr.item();

		if (n > cap) {
			buf = torch.pinned_empty([n], 'uint8');
			cap = n;
		}

		let v = buf.narrow(0, 0, n);

		torch.fd_recv_all(fd, v);
		torch.fd_send_all(fd, hdr);
		torch.fd_send_all(fd, v);
	}
}

if (mode == 'serve') {
	let l = socket.listen(bind, port, null, 16, true);

	if (!l)
		die(`listen ${bind}:${port}: ${socket.error()}\n`);

	warn(`rpc_pingpong: listening on ${bind}:${port}\n`);

	for (;;) {
		let c = l.accept();

		if (!c)
			continue;

		nodelay(c);

		if (native) {
			native_echo(c.fileno());
			c.close();
			warn('rpc_pingpong: client done\n');
			continue;
		}

		// each message: 8-byte decimal length, then the payload; echoed whole
		for (;;) {
			let hdr = read_n(c, 8);

			if (hdr == null)
				break;

			let n = int(hdr);
			let body = n ? read_n(c, n) : '';

			if (body == null || !write_all(c, hdr + body))
				break;
		}

		c.close();
		warn('rpc_pingpong: client done\n');
	}
}

let s = socket.connect(host, port);

if (!s)
	die(`connect ${host}:${port}: ${socket.error()}\n`);

nodelay(s);

function pct(v, p) {
	sort(v, (a, b) => a - b);

	return v[int((length(v) - 1) * p)];
}

printf('%-10s %-4s %9s %9s %9s %9s %10s\n', 'bytes', '', 'p50 us', 'p99 us', 'max us', 'mean us', 'MB/s');

for (let size in sizes) {
	if (native) {
		let hdr = torch.from_array([size], 'int64'), back = torch.zeros([1], 'int64');
		let tx = torch.pinned_empty([size], 'uint8'), rx = torch.pinned_empty([size], 'uint8');
		let n = (size >= 2000000) ? (iters > 100 ? 100 : iters) : iters;
		let rt = [];

		for (let i = 0; i < n + 10; i++) {
			let t0 = now();

			torch.fd_send_all(s.fileno(), hdr);
			torch.fd_send_all(s.fileno(), tx);
			torch.fd_recv_all(s.fileno(), back);
			torch.fd_recv_all(s.fileno(), rx);

			if (i >= 10)
				push(rt, now() - t0);
		}

		let sum = 0.0;

		for (let v in rt)
			sum += v;

		printf('%-10d %-4s %9.0f %9.0f %9.0f %9.0f %10.0f\n', size, 'nat', pct(rt, 0.5), pct(rt, 0.99),
			rt[length(rt) - 1], sum / length(rt), 2.0 * size / (sum / length(rt)));

		continue;
	}

	let payload = '0123456789abcdef';

	while (length(payload) < size)
		payload += payload;

	payload = substr(payload, 0, size);

	let hdr = sprintf('%08d', size);
	let n = (size >= 2000000) ? (iters > 100 ? 100 : iters) : iters;
	let rt = [];

	// warm-up
	for (let i = 0; i < 10; i++) {
		write_all(s, hdr + payload);
		read_n(s, 8 + size);
	}

	for (let i = 0; i < n; i++) {
		let t0 = now();

		if (!write_all(s, hdr + payload) || read_n(s, 8 + size) == null)
			die('connection lost\n');

		push(rt, now() - t0);
	}

	let sum = 0.0;

	for (let v in rt)
		sum += v;

	let mean = sum / length(rt);
	let p50 = pct(rt, 0.5), p99 = pct(rt, 0.99);

	printf('%-10d %-4s %9.0f %9.0f %9.0f %9.0f %10.0f\n', size, 'rtt', p50, p99, rt[length(rt) - 1], mean,
		2.0 * size / mean);
}

if (gpu != null) {
	print('\ndevice copy through a pinned buffer (D2H + sync, H2D + sync), cuda:' + gpu + '\n');
	printf('%-10s %-4s %9s %9s %9s\n', 'bytes', '', 'p50 us', 'p99 us', 'max us');

	for (let size in sizes) {
		let n = (size >= 2000000) ? 100 : iters;
		let dev = torch.zeros([size], 'uint8', 'cuda:' + gpu);
		let host = torch.pinned_empty([size], 'uint8');
		let d2h = [], h2d = [];

		for (let i = 0; i < n + 10; i++) {
			let t0 = now();

			torch.copy_async(host, dev);
			torch.synchronize();

			let t1 = now();

			dev.copy_(host);
			torch.synchronize();

			let t2 = now();

			if (i >= 10) {
				push(d2h, t1 - t0);
				push(h2d, t2 - t1);
			}
		}

		printf('%-10d %-4s %9.0f %9.0f %9.0f\n', size, 'D2H', pct(d2h, 0.5), pct(d2h, 0.99), d2h[length(d2h) - 1]);
		printf('%-10d %-4s %9.0f %9.0f %9.0f\n', size, 'H2D', pct(h2d, 0.5), pct(h2d, 0.99), h2d[length(h2d) - 1]);
	}
}

s.close();
