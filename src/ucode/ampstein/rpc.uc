// ucode/rpc.uc — transport for pipeline stages on other hosts (docs/RPC.md, phase 2).
//
// One persistent TCP connection per worker, carrying frames
//
//   header  4 x int64 (little endian): magic, seq, json length, tensor length
//   json    a metadata object, `t` names the message type
//   tensor  raw bytes of a contiguous CPU tensor (the activations of a hop),
//           received into a buffer from opts.alloc(meta, nbytes). By default that is
//           a view of one pinned buffer of the connection, reused: the tensor of a
//           frame is valid until the next frame with a tensor arrives (pass `alloc`
//           to keep frames; allocating pinned memory per frame costs ~0.5 ms)
//
// The coordinator LISTENS and the workers dial in. Everything here is plain TCP:
// no TLS, no authentication — put a VPN / overlay under it and bind to its
// address (docs/RPC.md).
//
//   coordinator                                   worker
//   let L = rpc.listen({bind, port, manifest});   rpc.worker({connect: 'host:port', name, manifest, serve});
//   L.wait(1, 60);            // start-up          // retries with back-off, hello, welcome, serve(conn, welcome)
//   L.workers[0].conn.send({t: 'step', ...}, tensor);
//   let f = conn.recv(1000);  // {meta, tensor, seq} or null on timeout
//   L.accept_one(0);          // a rejoin, from a uloop handle on L.fd()
//
// A conn is blocking for its sends and non-blocking for its receives: recv(ms) waits,
// pump() takes what has arrived (for a uloop handler) and returns a frame or null.
// Every error (peer closed, reset, bad frame, timeout of a send) raises.

import * as socket from 'socket';
import * as fs from 'fs';

export const PROTO = 1;

const MAGIC = 0x414d5043;                 // "AMPC"
const HDR_BYTES = 32;
const MAX_JSON = 1 << 20;
const MAX_TENSOR = 1 << 32;               // 4 GiB; a frame beyond it is a protocol error
const SEND_TIMEOUT = 30000;               // ms, a send that makes no progress that long raises

const ELEM = { bool: 1, uint8: 1, int8: 1, int16: 2, float16: 2, bfloat16: 2, int32: 4, float32: 4, int64: 8, float64: 8 };

function nbytes(t) {
	let e = ELEM[t.dtype()];

	if (!e)
		die(`rpc: unsupported tensor dtype ${t.dtype()}`);

	return t.numel() * e;
}

let sha256 = null;

try {
	sha256 = require('digest').sha256;
}
catch (e) {
}

function tune(sock) {
	sock.setopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1);
	sock.setopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1);
	sock.setopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 5);
	sock.setopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 2);
	sock.setopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3);
	sock.setopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 15000);
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1000 + int(t[1] / 1000000);
}

/* Send a string completely: poll while the socket is full. */
function send_str(sock, s, timeout_ms) {
	let off = 0, n = length(s), deadline = now_ms() + timeout_ms;

	while (off < n) {
		let w = sock.send(off ? substr(s, off) : s, socket.MSG_NOSIGNAL | socket.MSG_DONTWAIT);

		if (w != null && w > 0) {
			off += w;
			continue;
		}

		let err = socket.error();

		if (w == null && index(err, 'Resource temporarily unavailable') < 0)
			die(`rpc: send: ${err}`);

		let left = deadline - now_ms();

		if (left <= 0)
			die(`rpc: send: no progress for ${timeout_ms} ms`);

		socket.poll(left, [ sock, socket.POLLOUT ]);
	}
}

/* A framed connection around a connected socket. */
export function conn(sock, opts) {
	opts = opts ?? {};

	let max_tensor = opts.max_tensor ?? MAX_TENSOR;
	let seq = 0;
	let rx = {
		hdr: torch.zeros([4], 'int64'), got: 0, phase: 0,  // 0 header, 1 json, 2 body
		jlen: 0, tlen: 0, parts: [], jgot: 0, meta: null, body: null, bgot: 0, seq: 0
	};
	let c = null;
	let scratch = null;            // the default receive buffer (pinned memory is slow to allocate)

	function reset() {
		rx.got = 0;
		rx.phase = 0;
		rx.parts = [];
		rx.jgot = 0;
		rx.meta = null;
		rx.body = null;
		rx.bgot = 0;
	}

	function closed() {
		die('rpc: connection closed by the peer');
	}

	/* the frame in progress finished: apply the internal messages */
	function done() {
		let f = { meta: rx.meta, tensor: rx.body, seq: rx.seq };

		reset();

		if (f.meta.t == 'ping') {
			c.send({ t: 'pong', ts: f.meta.ts });

			return null;
		}

		return f;
	}

	c = {
		sock: sock,
		peer: opts.peer,

		fd: function () {
			return sock.fileno();
		},

		close: function () {
			sock.close();
		},

		/* send(meta, tensor?, timeout_ms?): the whole frame, or raise */
		send: function (meta, tensor, timeout_ms) {
			timeout_ms ??= SEND_TIMEOUT;

			let js = sprintf('%J', meta);
			let tlen = 0;

			if (tensor) {
				if (!tensor.is_contiguous())
					die('rpc: send: the tensor must be contiguous');

				tlen = nbytes(tensor);
			}

			if (length(js) > MAX_JSON)
				die('rpc: send: metadata too large');

			let fd = sock.fileno();

			torch.fd_send_all(fd, torch.from_array([ MAGIC, ++seq, length(js), tlen ], 'int64'), timeout_ms);
			send_str(sock, js, timeout_ms);

			if (tlen > 0)
				torch.fd_send_all(fd, tensor, timeout_ms);

			return seq;
		},

		/* pump(): consume what has arrived; a complete frame, or null when there is none yet */
		pump: function () {
			let fd = sock.fileno();

			for (;;) {
				if (rx.phase == 0) {
					let r = torch.fd_read(fd, rx.hdr, rx.got, HDR_BYTES - rx.got);

					if (r < 0)
						closed();

					if (r == 0)
						return null;

					rx.got += r;

					if (rx.got < HDR_BYTES)
						continue;

					let h = rx.hdr.tolist();

					if (h[0] != MAGIC)
						die('rpc: bad frame (magic)');

					if (h[2] < 2 || h[2] > MAX_JSON || h[3] < 0 || h[3] > max_tensor)
						die(`rpc: bad frame (json ${h[2]}, tensor ${h[3]} bytes)`);

					rx.seq = h[1];
					rx.jlen = h[2];
					rx.tlen = h[3];
					rx.phase = 1;
				}

				if (rx.phase == 1) {
					while (rx.jgot < rx.jlen) {
						let s = sock.recv(rx.jlen - rx.jgot, socket.MSG_DONTWAIT);

						if (s == null) {
							if (index(socket.error(), 'Resource temporarily unavailable') >= 0)
								return null;

							die(`rpc: recv: ${socket.error()}`);
						}

						if (length(s) == 0)
							closed();

						push(rx.parts, s);
						rx.jgot += length(s);
					}

					rx.meta = json(join('', rx.parts));

					if (type(rx.meta) != 'object' || type(rx.meta.t) != 'string')
						die('rpc: bad frame (metadata)');

					if (rx.tlen == 0) {
						let f = done();

						if (f)
							return f;

						continue;
					}

					rx.body = opts.alloc ? opts.alloc(rx.meta, rx.tlen) : null;

					if (!rx.body) {
						if (!scratch || scratch.numel() < rx.tlen)
							scratch = torch.pinned_empty([ (rx.tlen + 1048575) & ~1048575 ], 'uint8');

						rx.body = scratch.narrow(0, 0, rx.tlen);
					}

					if (nbytes(rx.body) != rx.tlen)
						die(`rpc: alloc returned ${nbytes(rx.body)} bytes for a ${rx.tlen} bytes tensor`);

					rx.phase = 2;
				}

				while (rx.bgot < rx.tlen) {
					let r = torch.fd_read(fd, rx.body, rx.bgot, rx.tlen - rx.bgot);

					if (r < 0)
						closed();

					if (r == 0)
						return null;

					rx.bgot += r;
				}

				let f = done();

				if (f)
					return f;
			}
		},

		/* recv(timeout_ms): the next frame; null after the timeout (-1: wait forever) */
		recv: function (timeout_ms) {
			let deadline = (timeout_ms == null || timeout_ms < 0) ? null : now_ms() + timeout_ms;

			for (;;) {
				let f = this.pump();

				if (f)
					return f;

				let left = -1;

				if (deadline != null) {
					left = deadline - now_ms();

					if (left <= 0)
						return null;
				}

				socket.poll(left, [ sock, socket.POLLIN ]);
			}
		},

		/* ping(timeout_ms): round trip in microseconds; other frames received meanwhile are returned in .other */
		ping: function (timeout_ms) {
			let t0 = clock(true);
			let other = [];

			this.send({ t: 'ping', ts: t0[0] * 1000000 + int(t0[1] / 1000) });

			for (;;) {
				let f = this.recv(timeout_ms ?? 5000);

				if (!f)
					die('rpc: ping: timeout');

				if (f.meta.t != 'pong') {
					push(other, f);
					continue;
				}

				let t1 = clock(true);

				return { us: (t1[0] - t0[0]) * 1000000 + int((t1[1] - t0[1]) / 1000), other: other };
			}
		}
	};

	return c;
}

/* The files of a model directory as an identity.
 *   id      names and sizes of every file and the header length of each .safetensors, as text
 *           ("name:size:headerlen" per line): computable on every host (no digest module needed),
 *           identical for copies of one model directory
 *   strong  sha256 of the above plus the hash of every .safetensors header (names, dtypes, shapes,
 *           offsets) and small .json; null when this host has no digest.so
 * The handshake refuses a worker whose id differs, and one whose strong differs when both have one. */
export function manifest(dir) {
	let names = sort(filter(fs.lsdir(dir) ?? [], (n) => substr(n, 0, 1) != '.'));
	let files = [];
	let weak = '', strong = '';

	for (let n in names) {
		let path = `${dir}/${n}`;
		let st = fs.stat(path);

		if (!st || st.type != 'file')
			continue;

		let len = '', h = '';

		if (match(n, /\.safetensors$/)) {
			let f = fs.open(path, 'r');

			if (f) {
				let l = require('struct').unpack('<Q', f.read(8))[0];

				len = `${l}`;

				if (sha256 && l > 0 && l < (1 << 28))
					h = sha256(f.read(l));

				f.close();
			}
		}
		else if (sha256 && match(n, /\.json$/) && st.size < (1 << 24))
			h = sha256(fs.readfile(path));

		push(files, { name: n, size: st.size, header: len, hash: h });
		weak += `${n}:${st.size}:${len}\n`;
		strong += `${n}:${st.size}:${len}:${h}\n`;
	}

	return { id: weak, strong: sha256 ? sha256(strong) : null, files: files };
}

/* The GPUs of this host: name, compute capability, total memory. With `mem` also the free memory,
 * which creates a CUDA context on every device (a few hundred MB each): not for the hello. */
export function gpus(mem) {
	let out = [];

	for (let i = 0; i < torch.device_count(); i++) {
		let p = torch.device_props(i);
		let g = { index: i, name: p.name, cc: `${p.major}.${p.minor}`, total: p.total };

		if (mem)
			g.free = torch.mem_info(i)[2];

		push(out, g);
	}

	return out;
}

function split_addr(addr) {
	let m = match(addr, /^(.*):([0-9]+)$/);

	if (!m)
		die(`rpc: "${addr}": expected HOST:PORT`);

	return { host: m[1], port: int(m[2]) };
}

/* the first line where two manifest texts differ, for a refusal message */
function first_difference(mine, theirs) {
	let a = split(`${mine}`, '\n'), b = split(`${theirs}`, '\n');

	for (let i = 0; i < (length(a) > length(b) ? length(a) : length(b)); i++)
		if (a[i] != b[i])
			return `coordinator ${a[i] ?? 'no such file'}, worker ${b[i] ?? 'no such file'}`;

	return 'unknown';
}

/* ---- coordinator side ------------------------------------------------------------------------ */

/* listen({bind, port, manifest, check?, welcome?, alloc?}): a listener for workers.
 *   bind     the address to listen on (required: there is no default, see the header)
 *   manifest rpc.manifest(dir) of the coordinator's model; a worker with another id is refused
 *   check    function (hello) -> null or a reason string to refuse the worker
 *   welcome  extra fields for the welcome message: an object, or a function (hello) -> object (e.g. the
 *            assignment of stages, made from what the worker announced; raising refuses it)
 * Returns {port, fd(), workers (array of {name, conn, hello}), accept_one(ms), wait(n, seconds), close()}. */
export function listen(opts) {
	if (!opts?.bind)
		die('rpc: listen: bind address required');

	let sock = socket.listen(opts.bind, opts.port ?? 0, null, 16);

	if (!sock)
		die(`rpc: listen ${opts.bind}:${opts.port}: ${socket.error()}`);

	let L = {
		workers: [],
		port: sock.sockname().port
	};

	function refuse(c, reason) {
		warn(`rpc: refusing a worker: ${reason}\n`);

		try {
			c.send({ t: 'reject', reason: reason });
		}
		catch (e) {
		}

		c.close();
	}

	L.fd = function () {
		return sock.fileno();
	};

	/* take one pending connection, run the handshake; the worker, or null (nothing pending / refused) */
	L.accept_one = function (timeout_ms) {
		if (!length(socket.poll(timeout_ms ?? 0, [ sock, socket.POLLIN ])))
			return null;

		let s = sock.accept();

		if (!s)
			return null;

		let peer = s.peername();

		tune(s);

		let c = conn(s, { alloc: opts.alloc, peer: peer });
		let hello = null;

		try {
			let f = c.recv(5000);

			hello = f?.meta;
		}
		catch (e) {
			warn(`rpc: handshake with ${peer?.address}: ${e}\n`);
			c.close();

			return null;
		}

		if (!hello || hello.t != 'hello')
			return refuse(c, 'expected a hello'), null;

		if (hello.proto != PROTO)
			return refuse(c, `protocol ${hello.proto}, this coordinator speaks ${PROTO}`), null;

		if (opts.manifest && hello.manifest != opts.manifest.id)
			return refuse(c, `the model directory differs (${first_difference(opts.manifest.id, hello.manifest)})`), null;

		if (opts.manifest?.strong && hello.strong && hello.strong != opts.manifest.strong)
			return refuse(c, 'the model directory differs (same files and sizes, other contents of a header or json file)'), null;

		let reason = opts.check ? opts.check(hello) : null;

		if (reason)
			return refuse(c, reason), null;

		let name = hello.name ?? `${peer?.address}:${peer?.port}`;
		let old = null;

		for (let i = 0; i < length(L.workers); i++)
			if (L.workers[i].name == name) {
				old = i;
				break;
			}

		let w = { name: name, conn: c, hello: hello, peer: peer };

		try {
			let extra = (type(opts.welcome) == 'function') ? opts.welcome(hello) : opts.welcome;

			c.send({ t: 'welcome', name: name, ...(extra ?? {}) });
		}
		catch (e) {
			c.close();

			return null;
		}

		if (old != null) {
			try {
				L.workers[old].conn.close();
			}
			catch (e) {
			}

			L.workers[old] = w;
		}
		else
			push(L.workers, w);

		warn(`rpc: worker ${name} joined from ${peer?.address} (${length(hello.gpus ?? [])} GPUs)\n`);

		return w;
	};

	/* block until n workers have joined; raise after `seconds` */
	L.wait = function (n, seconds) {
		let deadline = now_ms() + (seconds ?? 60) * 1000;

		while (length(L.workers) < n) {
			let left = deadline - now_ms();

			if (left <= 0)
				die(`rpc: ${length(L.workers)} of ${n} workers joined after ${seconds} s`);

			L.accept_one(left > 500 ? 500 : left);
		}

		return L.workers;
	};

	L.close = function () {
		for (let w in L.workers) {
			try {
				w.conn.close();
			}
			catch (e) {
			}
		}

		sock.close();
	};

	return L;
}

/* ---- worker side ----------------------------------------------------------------------------- */

/* dial(addr, opts): a connected conn, or raise */
export function dial(addr, opts) {
	let a = split_addr(addr);
	let s = socket.connect(a.host, a.port);

	if (!s)
		die(`rpc: connect ${addr}: ${socket.error()}`);

	tune(s);

	return conn(s, { alloc: opts?.alloc, peer: a });
}

/* introduce a worker: send the hello */
export function hello(c, opts) {
	c.send({
		t: 'hello', proto: PROTO, name: opts.name, manifest: opts.manifest?.id, strong: opts.manifest?.strong,
		gpus: opts.gpus ?? (torch.device_count ? gpus() : []), ...(opts.hello ?? {})
	});
}

/* wait for the welcome; raise on a refusal */
export function welcome(c, timeout_ms) {
	let f = c.recv(timeout_ms ?? 10000);

	if (!f)
		die('rpc: no answer to the hello');

	if (f.meta.t == 'reject')
		die(`rpc: refused by the coordinator: ${f.meta.reason}`);

	if (f.meta.t != 'welcome')
		die(`rpc: unexpected ${f.meta.t} instead of the welcome`);

	return f.meta;
}

/* worker({connect, name, manifest, serve, alloc?, retry?, stop?, once?}): dial in, serve, and
 * dial in again when the connection drops (a refusal is fatal: it will not get better).
 *   serve(conn, welcome) runs the worker's loop; it returns when the connection ends, or
 *   with the string 'bye' to leave for good.
 * With opts.once it returns after the first session. */
export function worker(opts) {
	let lo = opts.retry?.min ?? 500, hi = opts.retry?.max ?? 5000, wait = lo;

	for (;;) {
		if (opts.stop?.())
			return;

		let c = null, w = null;

		try {
			c = dial(opts.connect, opts);
			hello(c, opts);
			w = welcome(c, 10000);
		}
		catch (e) {
			if (c)
				c.close();

			if (index(`${e}`, 'refused by') >= 0)
				die(`${e}`);

			if (opts.once)
				die(`${e}`);

			warn(`${e}; retrying in ${wait} ms\n`);
			sleep(wait);
			wait = (wait * 2 > hi) ? hi : wait * 2;

			continue;
		}

		wait = lo;
		warn(`rpc: joined ${opts.connect} as ${w.name}\n`);

		let rv = null;

		try {
			rv = opts.serve(c, w);
		}
		catch (e) {
			warn(`rpc: session ended: ${e}\n`);
		}

		c.close();

		if (rv == 'bye' || opts.once)
			return rv;
	}
}
