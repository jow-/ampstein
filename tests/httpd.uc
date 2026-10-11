// tests/httpd.uc — ucode/httpd.uc
//
// Unit tests for the exported body readers and parse_head, driven directly
// with sliced input (no sockets). This is where the old O(n^2) string
// concatenation lived (`buf += data` / `body += substr(…)`), so the tests
// feed a large body in many small slices and check both correctness and
// that a big Content-Length body completes quickly (O(n), not O(n^2)).
//
// The chunked reader is exercised at awkward boundaries: chunk data split
// across feeds, the size line split across feeds, and the size line's own
// terminating CRLF split across feeds (a real recv-boundary case).
//
// The full listen loop (seeding the body bytes that arrive in the same recv
// as the header, 100-continue, completion detection) is exercised by the
// live server / a curl round-trip; an in-process socket client was tried
// but is held back because ucode's non-blocking client-socket uloop teardown
// is out of scope for this change (the module itself only accepts — see the
// clean server-only runs). The readers above are the only stateful logic the
// refactor touched.

import * as httpd from 'ampstein.httpd';
import * as math from 'math';

let fails = 0;

function check(name, ok, detail) {
	if (!ok)
		fails++;
	print((ok ? 'ok: ' : 'FAIL: ') + name + (detail ? ' — ' + detail : '') + '\n');
}

function now_ms() {
	let t = clock(true);

	return t[0] * 1e3 + t[1] / 1e6;
}

// ---------- parse_head ----------
let r = httpd.parse_head('POST /v1/chat/completions?stream=1 HTTP/1.1\r\n' +
	'Host: localhost:8000\r\nContent-Type: application/json\r\n' +
	'X-Custom:  abc \r\n\r\n');
check('parse_head empty head is malformed', httpd.parse_head('') == null);
check('parse_head bare request line is malformed', httpd.parse_head('GET /') == null);
check('parse_head method', r && r.method == 'POST', r && r.method);
check('parse_head path', r && r.path == '/v1/chat/completions', r && r.path);
check('parse_head query', r && r.query == 'stream=1', r && r.query);
check('parse_head version', r && r.version == 'HTTP/1.1', r && r.version);
check('parse_head host header', r && r.headers['host'] == 'localhost:8000', r && r.headers['host']);
check('parse_head header value trimmed', r && r.headers['x-custom'] == 'abc', r && r.headers['x-custom']);
check('parse_head malformed (no space) is null',
	httpd.parse_head('GETNOPE\r\n\r\n') == null, '');
check('parse_head malformed (bad line) skipped',
	httpd.parse_head('GET / HTTP/1.1\r\n:bad:line\r\nOk: 1\r\n\r\n').headers['ok'] == '1', '');

// ---------- make_cl_reader ----------
// fed in pieces, completes exactly at the boundary
let cl = httpd.make_cl_reader(10);
check('cl: incomplete after 5', cl.feed('01234') == null, '');
check('cl: complete at boundary', cl.feed('56789') == '0123456789', '');

// body that arrives with more bytes than cl (trailing ignored)
let cl2 = httpd.make_cl_reader(4);
check('cl: trailing bytes ignored', cl2.feed('WORLD') == substr('WORLD', 0, 4), '');

// cl = 0 -> empty body immediately
check('cl: zero length is empty', httpd.make_cl_reader(0).feed('') == '', '');

// a large Content-Length body fed in 1 KiB slices: O(n) by construction;
// the bound below is generous enough to be non-flaky yet would be exceeded
// by the old O(n^2) concatenation on this size.
let BIG = 512 * 1024;
let block = '';
for (let i = 0; i < 1024; i++)
	block += substr('abcdefghij', i % 10, 1);   // 1 KiB pattern
let big = block;
while (length(big) < BIG)
	big += big;                        // geometric doubling, O(n) total
let bigbody = substr(big, 0, BIG);
let breader = httpd.make_cl_reader(BIG);
let t0 = now_ms();
let got = null;
for (let o = 0; o < BIG; o += 1024) {
	got = breader.feed(substr(bigbody, o, 1024));
	if (got != null && length(got) < BIG)
		break;
}
let dt = now_ms() - t0;
check('cl: 512 KiB body reassembled exactly', got == bigbody, 'len ' + (got ? length(got) : 'null'));
check('cl: 512 KiB body is fast (O(n))', dt < 2000, '' + math.floor(dt) + ' ms');

// ---------- make_chunked_reader ----------
// two chunks; framing split across feeds at awkward boundaries. The full
// stream is 5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n (body "hello world").
let ch = httpd.make_chunked_reader();
check('chunk: size line + partial data', ch.feed('5\r\nhel') == null, '');
check('chunk: finish data + next size', ch.feed('lo\r\n6\r\n world\r\n') == null, '');
check('chunk: final chunk', ch.feed('0\r\n\r\n') == 'hello world', '');

// the size line "7\r\n" is split as "7\r" + "\n..." — the terminating CRLF
// itself spans two feeds, which the reader must reassemble. Full stream:
// 7\r\n1234567\r\n0\r\n\r\n (body "1234567").
let ch2 = httpd.make_chunked_reader();
check('chunk: size line split (CRLF across feeds)', ch2.feed('7\r') == null, '');
check('chunk: size line rest + partial data', ch2.feed('\n12345') == null, '');
check('chunk: rest of data + final', ch2.feed('67\r\n0\r\n\r\n') == '1234567', '');

// a single chunk whose data is split across three feeds. Full stream:
// a\r\n0123456789\r\n0\r\n\r\n (body "0123456789").
let ch3 = httpd.make_chunked_reader();
check('chunk3 a (size line)', ch3.feed('a\r\n') == null, '');
check('chunk3 b (partial data)', ch3.feed('01234') == null, '');
check('chunk3 c (more data)', ch3.feed('5678') == null, '');
check('chunk3 d (final byte + CRLF + final)', ch3.feed('9\r\n0\r\n\r\n') == '0123456789', '');

// chunked body with an extension on the size line. Full stream:
// 5;ext=1\r\nhello\r\n0\r\n\r\n (body "hello").
let ch4 = httpd.make_chunked_reader();
check('chunk4 a (size line with extension)', ch4.feed('5;ext=1\r\nhello') == null, '');
check('chunk4 b (chunk CRLF + final)', ch4.feed('\r\n0\r\n\r\n') == 'hello', '');

// malformed size line raises
let raised = false;
try {
	let ch5 = httpd.make_chunked_reader();
	ch5.feed('zz\r\n');
} catch (e) {
	raised = true;
}
check('chunk: malformed size raises', raised, '');

if (fails)
	die(fails + ' failures');
print('all green\n');
