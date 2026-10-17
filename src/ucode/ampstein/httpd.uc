// ucode/httpd.uc — minimal HTTP/1.1 server on uloop + socket.
//
// A deliberately small subset, enough for OpenAI-style API clients:
//
//   - one request per connection (responses carry Connection: close)
//   - request bodies by Content-Length or chunked transfer coding
//   - Expect: 100-continue
//   - responses with Content-Length, or streamed with chunked coding
//     (Server-Sent Events)
//   - disconnect detection for long-running streamed responses
//
// Usage:
//   import * as httpd from 'ampstein.httpd';
//   httpd.listen('127.0.0.1', 8000, function (req, res) {
//       res.json(200, {ok: true});
//  });
//   uloop.run();
//
// The handler may finish the response later (from a uloop timer): the
// connection stays open until res.end()/res.json()/res.send() is called.

import * as socket from 'socket';
import * as uloop from 'uloop';
import * as struct from 'struct';

const MAX_HEADER = 64 * 1024;
const MAX_BODY = 256 * 1024 * 1024;

const STATUS_TEXT = {
	'100': 'Continue', '200': 'OK', '204': 'No Content',
	'400': 'Bad Request', '404': 'Not Found', '405': 'Method Not Allowed',
	'408': 'Request Timeout', '413': 'Payload Too Large',
	'429': 'Too Many Requests', '500': 'Internal Server Error',
	'503': 'Service Unavailable'
};

/* Write all of data, looping over partial sends. Returns false when the
 * peer is gone. MSG_NOSIGNAL: a closed peer must not SIGPIPE the server. */
function send_all(conn, data) {
	let off = 0, n = length(data);

	while (off < n) {
		let chunk = (off == 0) ? data : substr(data, off);
		let sent = conn.send(chunk, socket.MSG_NOSIGNAL);

		if (sent == null || sent <= 0)
			return false;

		off += sent;
	}

	return true;
}

/* Parse the header block. Returns the request object (without body) or
 * null when malformed. */
export function parse_head(head) {
	let lines = split(head, '\r\n');

	if (!length(lines))
		return null;

	let rl = split(lines[0], ' ');

	if (length(rl) < 3)
		return null;

	let target = rl[1];
	let q = index(target, '?');
	let req = {
		method: uc(rl[0]),
		target: target,
		path: (q >= 0) ? substr(target, 0, q) : target,
		query: (q >= 0) ? substr(target, q + 1) : '',
		version: rl[2],
		headers: {}
	};

	for (let i = 1; i < length(lines); i++) {
		let c = index(lines[i], ':');

		if (c <= 0)
			continue;

		req.headers[lc(trim(substr(lines[i], 0, c)))] = trim(substr(lines[i], c + 1));
	}

	return req;
}

/* Request-body readers: accumulate the body incrementally so a large
 * body costs O(n) total, not O(n^2) from repeated string concatenation
 * (the old `buf += data` / `body += substr(…)` copies). feed() returns
 * the complete body string when the body is done, or null while more
 * data is needed; it raises on malformed input.
 *
 * Both are exported (rather than private) so tests/httpd.uc can drive
 * them directly with sliced input, no sockets.
 */

/* Content-Length body: buffer-backed, one O(cl) copy at the end. */
export function make_cl_reader(cl) {
	let body = struct.buffer();

	return {
		feed: function (data) {
			if (length(data))
				body.put('*', data);

			if (body.length() >= cl)
				return body.slice(0, cl);

			return null;
		}
	};
}

/* Chunked body: incremental, O(n) total. The offset-aware index() finds
 * chunk framing without copying the accumulated stream; each chunk's data
 * is copied into the buffer exactly once. */
export function make_chunked_reader() {
	let body = struct.buffer();
	let size = -1;      // current chunk size; -1 = expecting a size line
	let szline = '';    // partial size line (small, one recv at a time)

	return {
		feed: function (data) {
			let off = 0, n = length(data);

			while (true) {
				if (size < 0) {
					// collect the chunk-size line. It may be split across feeds
					// (even the terminating CRLF), so search the carried partial
					// plus this feed's remaining bytes.
					let sl = length(szline);
					let rest = substr(data, off);
					let search = (sl == 0) ? rest : (szline + rest);
					let e = index(search, '\r\n');

					if (e < 0) {
						szline = search;   // still no full CRLF; carry it all
						return null;
					}

					size = hex(trim(split(substr(search, 0, e), ';')[0]));
					szline = '';
					off += (e + 2) - sl;   // advance past the CRLF (data coords)

					if (size == null || size != size)
						die('malformed chunk size');

					if (size == 0)
						return body.slice();   // final chunk, body complete
				}

				// collect `size` bytes of chunk data
				let take = min(n - off, size);

				if (take > 0) {
					body.put('*', substr(data, off, take));
					off += take;
					size -= take;

					if (body.length() > MAX_BODY)
						die('body too large');
				}

				if (size == 0) {
					// skip the chunk's trailing CRLF, then expect a size line
					if (n - off < 2)
						return null;
					off += 2;
					size = -1;
				}
				else if (n == off)
					return null;   // need more data for this chunk
			}
		}
	};
}

function make_response(conn, cleanup) {
	let finished = false;
	let streaming = false;
	let dead = false;

	function head(status, headers, extra) {
		let s = `HTTP/1.1 ${status} ${STATUS_TEXT['' + status] || 'Status'}\r\n`;

		headers = headers || {};

		for (let k in headers)
			s += k + ': ' + headers[k] + '\r\n';

		s += 'Access-Control-Allow-Origin: *\r\n';
		s += 'Connection: close\r\n';

		if (extra)
			s += extra;

		return s + '\r\n';
	}

	function close() {
		if (finished)
			return;

		finished = true;
		cleanup();
	}

	let res = {
		send: function (status, headers, body) {
			if (finished)
				return;

			body = (body == null) ? '' : '' + body;

			if (!send_all(conn, head(status, headers,
					`Content-Length: ${length(body)}\r\n`) + body))
				dead = true;

			close();
		},

		json: function (status, obj, headers) {
			this.send(status, {'Content-Type': 'application/json', ...(headers || {})},
				sprintf('%J', obj));
		},

		/* Start a chunked streamed response (SSE by default). */
		stream: function (status, headers) {
			if (finished || streaming)
				return;

			streaming = true;
			headers = headers || {
				'Content-Type': 'text/event-stream',
				'Cache-Control': 'no-cache',
				'X-Accel-Buffering': 'no'
			};

			if (!send_all(conn, head(status, headers, 'Transfer-Encoding: chunked\r\n')))
				dead = true;
		},

		/* Write one chunk of a streamed response. Returns false when the
		 * peer is gone. */
		write: function (data) {
			if (finished || dead)
				return false;

			if (length(data) == 0)
				return true;

			if (!send_all(conn, sprintf('%x\r\n', length(data)) + data + '\r\n'))
				dead = true;

			return !dead;
		},

		/* One SSE event carrying a JSON payload (or a raw string). */
		sse: function (payload) {
			return this.write('data: ' +
				((type(payload) == 'string') ? payload : sprintf('%J', payload)) +
				'\n\n');
		},

		/* SSE comment line: keeps proxies/clients from timing out and
		 * probes the connection. */
		ping: function (text) {
			return this.write(': ' + (text || 'ping') + '\n\n');
		},

		end: function () {
			if (finished)
				return;

			if (streaming && !dead)
				send_all(conn, '0\r\n\r\n');

			close();
		},

		/* False once the client has disconnected. */
		alive: function () {
			if (finished || dead)
				return false;

			let ev = socket.poll(0, [conn, socket.POLLRDHUP | socket.POLLHUP | socket.POLLERR]);

			if (ev && length(ev) && (ev[0][1] & (socket.POLLRDHUP | socket.POLLHUP | socket.POLLERR)))
				dead = true;

			return !dead;
		},

		finished: function () {
			return finished;
		}
	};

	return res;
}

/* Start listening. handler(req, res) is called once per complete
 * request. Returns the listening socket. */
export function listen(host, port, handler) {
	let srv = socket.listen(host, port, {socktype: socket.SOCK_STREAM}, 128, true);

	if (!srv)
		die(`httpd: cannot listen on ${host}:${port}: ${socket.error()}`);

	uloop.handle(srv, function () {
		let conn = srv.accept();

		if (!conn)
			return;

		conn.setopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, true);

		let buf = '';
		let req = null;
		let reader = null;
		let hdl = null;
		let done = false;
		let sent_continue = false;

		function cleanup() {
			if (hdl) {
				hdl.delete();
				hdl = null;
			}

			conn.close();
		}

		function fail(status, msg) {
			done = true;
			make_response(conn, cleanup).json(status, {
				error: {message: msg, type: 'invalid_request_error', code: null}
			});
		}

		hdl = uloop.handle(conn, function () {
			if (done)
				return;

			let data = conn.recv(65536, socket.MSG_DONTWAIT);

			if (data == null)
				return;     // EAGAIN

			if (data == '') {
				done = true;
				cleanup();
				return;
			}

			let r;

			if (!req) {
				buf += data;

				// RFC 9112 2.2: ignore empty lines before the request line (clients
				// send a stray CRLF after a body on a kept-alive connection)
				while (index(buf, '\r\n') == 0)
					buf = substr(buf, 2);

				let he = index(buf, '\r\n\r\n');

				if (he < 0) {
					if (length(buf) > MAX_HEADER)
						fail(413, 'request header too large');

					return;
				}

				try {
					req = parse_head(substr(buf, 0, he));
				}
				catch (e) {
					req = null;
				}

				let rest = substr(buf, he + 4);
				buf = null;

				if (!req)
					return fail(400, 'malformed request');

				if (lc(req.headers['expect'] || '') == '100-continue' && !sent_continue) {
					sent_continue = true;
					send_all(conn, 'HTTP/1.1 100 Continue\r\n\r\n');
				}

				let te = lc(req.headers['transfer-encoding'] || '');
				let cl = int(req.headers['content-length'] || 0);

				if (index(te, 'chunked') >= 0)
					reader = make_chunked_reader();
				else {
					if (cl > MAX_BODY)
						return fail(413, 'request body too large');

					reader = make_cl_reader(cl);
				}

				// the recv that completed the header may already carry body bytes
				try {
					r = reader.feed(rest);
				}
				catch (e) {
					return fail(400, 'malformed body');
				}
			}
			else {
				try {
					r = reader.feed(data);
				}
				catch (e) {
					return fail(400, 'malformed body');
				}
			}

			if (r == null)                    // an empty body ('' for Content-Length 0 / no body) is complete
				return;

			req.body = r;

			// request complete: stop reading; the handler owns the
			// connection from here (disconnects are detected with
			// res.alive() / failed writes)
			done = true;

			if (hdl) {
				hdl.delete();
				hdl = null;
			}

			let res = make_response(conn, cleanup);

			try {
				handler(req, res);
			}
			catch (e) {
				warn(`httpd: handler error: ${e.message}\n${e.stacktrace ? e.stacktrace[0].context : ''}\n`);

				if (!res.finished())
					res.json(500, {error: {message: 'internal error: ' + e.message,
						type: 'server_error', code: null}});
			}
		}, uloop.ULOOP_READ);
	}, uloop.ULOOP_READ);

	return srv;
}
