// rpc.uc: framing, the receive state machine, ping, the handshake and its
// refusals, the manifest (docs/RPC.md phase 2). One process, loopback TCP:
// the messages are small enough for the socket buffers, so both ends can be
// driven by one thread.
//   ucode -L <build-dir>/lib -L src/ucode -ltorch=ampstein.torch tests/rpc_transport.uc

import * as socket from 'socket';
import * as fs from 'fs';
import * as rpc from 'ampstein.rpc';

let fails = 0;

function check(what, ok, extra) {
	if (!ok) {
		warn(`FAIL: ${what}${extra ? ' (' + extra + ')' : ''}\n`);
		fails++;
	}
	else
		warn(`ok: ${what}\n`);
}

function raises(fn, needle) {
	try {
		fn();
	}
	catch (e) {
		return needle == null || index(`${e}`, needle) >= 0;
	}

	return false;
}

function same(a, b) {
	return a.to('int32').sub(b.to('int32')).abs().max().item() == 0;
}

// a connected pair of conns over loopback TCP
function pair(opts) {
	let l = socket.listen('127.0.0.1', 0, null, 4);
	let c = socket.connect('127.0.0.1', l.sockname().port);
	let s = l.accept();

	l.close();

	return [ rpc.conn(c, opts), rpc.conn(s, opts) ];
}

// --- frames
let p = pair(), A = p[0], B = p[1];

A.send({ t: 'cmd', n: 7, s: 'häh' });

let f = B.recv(1000);

check('meta only', f && f.meta.t == 'cmd' && f.meta.n == 7 && f.meta.s == 'häh' && f.tensor == null);

let t = torch.arange(100000).to('uint8');

A.send({ t: 'act', pos: 5 }, t);
f = B.recv(1000);
check('tensor payload', f && f.meta.pos == 5 && same(f.tensor, t) && f.tensor.numel() == 100000);

let h = torch.arange(1000).to('float16');

B.send({ t: 'x' }, h);
f = A.recv(1000);
check('float16 payload sized in bytes', f && f.tensor.numel() == 2000);

A.send({ t: 'one' });
A.send({ t: 'two' }, t.narrow(0, 0, 10));
A.send({ t: 'three' });

check('frames arrive in order', B.recv(1000).meta.t == 'one' && B.recv(1000).meta.t == 'two' && B.recv(1000).meta.t == 'three');
check('recv times out with null', B.recv(30) == null);
check('pump with nothing pending is null', B.pump() == null);

// --- alloc hook
let got = [], P = pair({ alloc: function (meta, n) {
	push(got, [ meta.t, n ]);

	return torch.pinned_empty([ n ], 'uint8');
} });

P[0].send({ t: 'a' }, t.narrow(0, 0, 64));
check('alloc hook gets the type and size', P[1].recv(1000).tensor.numel() == 64 && got[0][0] == 'a' && got[0][1] == 64);

let Q = pair({ alloc: (meta, n) => torch.zeros([ n + 1 ], 'uint8') });

Q[0].send({ t: 'a' }, t.narrow(0, 0, 8));
check('alloc of the wrong size raises', raises(() => Q[1].recv(1000), 'alloc returned'));

// --- ping
A.send({ t: 'ping', ts: 1 });
check('a ping is answered inside recv and not surfaced', B.recv(30) == null);

let pr = A.recv(1000);

check('...and comes back as a pong', pr && pr.meta.t == 'pong' && pr.meta.ts == 1);

// --- the receive state machine, one byte at a time
let raw = socket.listen('127.0.0.1', 0, null, 4);
let cs = socket.connect('127.0.0.1', raw.sockname().port), ss = raw.accept();
let R = rpc.conn(ss);
let js = sprintf('%J', { t: 'slow', k: [ 1, 2, 3 ] });
let hdr = torch.from_array([ 0x414d5043, 9, length(js), 4 ], 'int64').tolist();
let bytes = '';

for (let v in hdr)
	for (let i = 0; i < 8; i++)
		bytes += chr(int(v / (1 << (8 * i))) & 255);

bytes += js + 'abcd';

let slow = null;

for (let i = 0; i < length(bytes); i++) {
	cs.send(substr(bytes, i, 1));
	socket.poll(50, [ ss, socket.POLLIN ]);
	slow = R.pump();

	if (slow && i < length(bytes) - 1)
		break;
}

check('a frame fed byte by byte is reassembled', slow && slow.meta.t == 'slow' && slow.seq == 9 && join(',', slow.tensor.tolist()) == '97,98,99,100',
	slow ? join(',', slow.tensor.tolist()) : 'no frame');

// --- bad frames and closing
let junk = pair();

junk[0].sock.send('this is not a frame at all, 40 bytes long!!');
check('a bad magic raises', raises(() => junk[1].recv(1000), 'bad frame'));

let big = pair({ max_tensor: 1000 });

big[0].send({ t: 'x' }, torch.zeros([ 2000 ], 'uint8'));
check('an oversize tensor is refused', raises(() => big[1].recv(1000), 'bad frame'));

let cl = pair();

cl[0].close();
check('a closed peer raises', raises(() => cl[1].recv(1000), 'closed'));

// --- the handshake
let M1 = { id: 'aaaa' }, M2 = { id: 'bbbb' };
let L = rpc.listen({ bind: '127.0.0.1', port: 0, manifest: M1, welcome: { layers: [ 0, 10 ] } });
let addr = `127.0.0.1:${L.port}`;

check('listen needs an address', raises(() => rpc.listen({ port: 1 }), 'bind'));

let w = rpc.dial(addr);

rpc.hello(w, { name: 'j7', manifest: M1, gpus: [ { index: 0, name: 'x' } ] });
check('a worker with the same manifest joins', L.accept_one(1000)?.name == 'j7' && length(L.workers) == 1);

let wl = rpc.welcome(w, 1000);

check('welcome carries the coordinator fields', wl.t == 'welcome' && wl.name == 'j7' && join(',', wl.layers) == '0,10');
check('the coordinator can send to the worker and back', (L.workers[0].conn.send({ t: 'go' }), w.recv(1000).meta.t == 'go') &&
	(w.send({ t: 'ack' }), L.workers[0].conn.recv(1000).meta.t == 'ack'));
check('the coordinator can ping the worker (answered by the worker side recv)', (w.send({ t: 'noop' }), true));

let w2 = rpc.dial(addr);

rpc.hello(w2, { name: 'bad', manifest: M2 });
check('a different manifest is not accepted', L.accept_one(1000) == null && length(L.workers) == 1);
check('...and the worker is told why', raises(() => rpc.welcome(w2, 1000), 'model directory differs'));

let w3 = rpc.dial(addr);

w3.send({ t: 'hello', proto: 99, name: 'old', manifest: 'aaaa' });
check('a different protocol is not accepted', L.accept_one(1000) == null);
check('...and the worker is told why (protocol)', raises(() => rpc.welcome(w3, 1000), 'protocol 99'));

let w4 = rpc.dial(addr);

w4.send({ t: 'notahello' });
check('a first message that is no hello is refused', L.accept_one(1000) == null);

let L2 = rpc.listen({ bind: '127.0.0.1', port: 0, manifest: M1, check: (hello) => hello.name == 'no' ? 'not you' : null });
let w5 = rpc.dial(`127.0.0.1:${L2.port}`);

rpc.hello(w5, { name: 'no', manifest: M1 });
check('the check callback can refuse', L2.accept_one(1000) == null && raises(() => rpc.welcome(w5, 1000), 'not you'));

// a rejoin under the same name replaces the old connection
let w6 = rpc.dial(addr);

rpc.hello(w6, { name: 'j7', manifest: M1 });
check('a rejoin replaces the worker of that name', L.accept_one(1000)?.name == 'j7' && length(L.workers) == 1 && rpc.welcome(w6, 1000).t == 'welcome');
check('...and the old connection is closed', raises(() => w.recv(1000)));

check('wait raises when nobody comes', raises(() => L2.wait(1, 0), 'workers joined'));

// worker(): refused is fatal, once returns after the session
let L3 = rpc.listen({ bind: '127.0.0.1', port: 0, manifest: M1 });
let sess = 0;

// the worker's dial is answered by accept_one in another step: run the worker with `once` on a
// connection the listener already holds in its backlog
let wc = rpc.dial(`127.0.0.1:${L3.port}`);

rpc.hello(wc, { name: 'w', manifest: M2 });
L3.accept_one(1000);
check('worker(): a refusal raises (not retried)', raises(() => rpc.welcome(wc, 1000), 'refused'));

check('worker(): connection failure with once raises', raises(() => rpc.worker({
	connect: '127.0.0.1:1', name: 'w', manifest: M1, once: true, serve: () => 'bye'
})));

// --- manifest
function rpc_u64(n) {
	return require('struct').pack('<Q', n);
}

let dir = fs.mkdtemp('/tmp/rpc_manifest_XXXXXX');
let hdrjson = '{"w":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}';

fs.writefile(`${dir}/model.safetensors`, rpc_u64(length(hdrjson)) + hdrjson + 'xxxx');
fs.writefile(`${dir}/config.json`, '{"a":1}');
fs.writefile(`${dir}/.hidden`, 'x');

let m1 = rpc.manifest(dir);

check('manifest lists the files, not the hidden ones', join(',', map(m1.files, (x) => x.name)) == 'config.json,model.safetensors');
check('manifest id is stable', rpc.manifest(dir).id == m1.id);

fs.writefile(`${dir}/config.json`, '{"a":2}');
check('manifest changes with a config edit', rpc.manifest(dir).id != m1.id);

fs.writefile(`${dir}/config.json`, '{"a":1}');
check('...and is back with it', rpc.manifest(dir).id == m1.id);

fs.writefile(`${dir}/model.safetensors`, rpc_u64(length(hdrjson)) + replace(hdrjson, 'F16', 'F32') + 'xxxx');
check('manifest changes with a tensor header edit', rpc.manifest(dir).id != m1.id);

for (let n in fs.lsdir(dir))
	fs.unlink(`${dir}/${n}`);

fs.rmdir(dir);

// --- gpus
if (torch.device_count() > 0) {
	let g = rpc.gpus();

	check('gpus(): name, capability, memory', length(g) == torch.device_count() && g[0].name && match(g[0].cc, /^[0-9]+\.[0-9]+$/) && g[0].total > 0 && g[0].free == null);
}

L.close();
L2.close();
L3.close();

if (fails) {
	warn(`${fails} failures\n`);
	exit(1);
}

warn('rpc_transport: ok\n');
