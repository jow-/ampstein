// fd_write / fd_read / fd_send_all / fd_recv_all: raw tensor bytes over a
// socket (docs/RPC.md phase 1). Needs no GPU for the transfer itself; the
// pinned buffer and the device round trip run when CUDA is present.
//   ucode -L <build-dir>/lib -L src/ucode -ltorch=ampstein.torch tests/fd_tensor.uc

import * as socket from 'socket';

let fails = 0;

function check(what, ok, extra) {
	if (!ok) {
		warn(`FAIL: ${what}${extra ? ' (' + extra + ')' : ''}\n`);
		fails++;
	}
	else
		warn(`ok: ${what}\n`);
}

function raises(fn) {
	try {
		fn();
	}
	catch (e) {
		return true;
	}

	return false;
}

function fill(n, seed) {
	let v = [];

	for (let i = 0; i < n; i++)
		push(v, (i * 31 + seed) % 251);

	return torch.from_array(v, 'uint8');
}

function same(a, b) {
	return a.to('int32').sub(b.to('int32')).abs().max().item() == 0;
}

let pr = socket.pair(socket.AF_UNIX, socket.SOCK_STREAM), a = pr[0], b = pr[1];
let fa = a.fileno(), fb = b.fileno();

// small tensor, both calls blocking-style
let src = fill(4096, 7), dst = torch.zeros([4096], 'uint8');

torch.fd_send_all(fa, src);
torch.fd_recv_all(fb, dst);
check('4 KiB over a socketpair', same(src, dst));

// the other direction, float16 view and a pinned buffer
let h = torch.pinned_empty([1024], 'float16');
let hv = torch.arange(1024).to('float16');

h.copy_(hv);

let h2 = torch.pinned_empty([1024], 'float16');

torch.fd_send_all(fb, h);
torch.fd_recv_all(fa, h2);
check('pinned float16 tensor', same(h.to('float32'), h2.to('float32')) && h2.to('float32').sum().item() == hv.to('float32').sum().item());

// a slice of a bigger buffer
let big = fill(10000, 3), out = torch.zeros([10000], 'uint8');

torch.fd_send_all(fa, big.narrow(0, 1000, 2000));
torch.fd_recv_all(fb, out.narrow(0, 5000, 2000));
check('narrow() slice lands at its offset', same(big.narrow(0, 1000, 2000), out.narrow(0, 5000, 2000)) &&
	out.narrow(0, 0, 5000).sum().item() == 0);

// nothing to read on a non-blocking attempt
check('fd_read: EAGAIN is 0', torch.fd_read(fb, dst, 0, 100) == 0);

// partial progress with fd_write / fd_read: a tensor far bigger than the socket buffers,
// pumped from one thread by alternating both ends
let N = 8 * 1024 * 1024;
let tx = torch.arange(N).to('uint8'), rx = torch.zeros([N], 'uint8');
let sent = 0, got = 0, spins = 0, blocked = 0;

while (got < N && spins++ < 100000) {
	// several writes before each read, so that the socket buffer fills up
	for (let k = 0; k < 16 && sent < N; k++) {
		let w = torch.fd_write(fa, tx, sent, N - sent > 65536 ? 65536 : N - sent);

		if (w == 0) {
			blocked++;
			break;
		}

		sent += w;
	}

	let r = torch.fd_read(fb, rx, got, N - got);

	if (r < 0)
		die('fd_read: stream ended early\n');

	got += r;
}

check('8 MiB through fd_write / fd_read', got == N && same(tx, rx), `${got}/${N}`);
check('the socket buffer filled up at least once (EAGAIN path)', blocked > 0, `${blocked}`);

// fd_send_all on a full socket times out instead of hanging
let fill_t = torch.zeros([1 << 20], 'uint8'), limit = 0;

while (torch.fd_write(fa, fill_t, 0, 1 << 20) > 0 && limit++ < 1000);

check('fd_send_all: timeout on a full socket', raises(() => torch.fd_send_all(fa, fill_t, 50)));

// and so does fd_recv_all with nothing to read (drain first)
let drain = torch.zeros([1 << 20], 'uint8');

while (torch.fd_read(fb, drain, 0, 1 << 20) > 0);

check('fd_recv_all: timeout on an empty socket', raises(() => torch.fd_recv_all(fb, torch.zeros([16], 'uint8'), 50)));

// an orderly close: fd_read -1, fd_recv_all raises
a.close();
check('fd_read: orderly EOF is -1', torch.fd_read(fb, drain, 0, 16) == -1);
check('fd_recv_all: closed peer raises', raises(() => torch.fd_recv_all(fb, drain.narrow(0, 0, 16))));
b.close();

// GPU round trip: device -> pinned -> wire -> pinned -> device
if (torch.device_count() > 0) {
	let pr2 = socket.pair(socket.AF_UNIX, socket.SOCK_STREAM), c = pr2[0], d = pr2[1];
	let dev = fill(65536, 5).cuda(0), back = torch.zeros([65536], 'uint8', 'cuda:0');
	let pa = torch.pinned_empty([65536], 'uint8'), pb = torch.pinned_empty([65536], 'uint8');

	torch.copy_async(pa, dev);
	torch.synchronize();
	torch.fd_send_all(c.fileno(), pa);
	torch.fd_recv_all(d.fileno(), pb);
	torch.copy_h2d_async(back, pb);
	torch.synchronize();
	check('device -> pinned -> socket -> pinned -> device', same(dev.cpu(), back.cpu()));
	c.close();
	d.close();
}

if (fails) {
	warn(`${fails} failures\n`);
	exit(1);
}

warn('fd_tensor: ok\n');
