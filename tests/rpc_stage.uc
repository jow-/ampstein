// rpc_stage.uc: the worker's command loop and the coordinator's client of the remote first
// stages (docs/RPC.md phase 4), against a mock model on the CPU: one process, loopback TCP,
// the worker driven step by step through K.pump. The GPU model is checked by glm5_split.uc and
// the two-host run.
//   ucode -L <build-dir>/lib -L src/ucode -ltorch=ampstein.torch tests/rpc_stage.uc

import * as socket from 'socket';
import * as fs from 'fs';
import * as rpc from 'ampstein.rpc';
import * as stage from 'ampstein.rpc_stage';

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

function pair() {
	let l = socket.listen('127.0.0.1', 0, null, 4);
	let c = socket.connect('127.0.0.1', l.sockname().port);
	let s = l.accept();

	l.close();

	return [ rpc.conn(c), rpc.conn(s) ];
}

const HC = 4, HID = 8;

// a stand-in for the partial model of the worker's stages: layers 0..2 of 5, stage 0 = layers 0-1, stage 1 = layer 2
let handoffs = 0;
let mock = {
	stage_of: [ 0, 0, 1, 1, 1 ],
	own_layers: [ 0, 3 ],
	new_cache: (max_seq) => ({ max_seq: max_seq, length: 0 }),
	embed: function (ids, mm, pos) {
		let n = length(ids);

		return [ torch.from_array(ids, 'float32').reshape([ n, 1, 1 ]).add(pos * 0.5).expand([ n, HC, HID ]).contiguous() ];
	},
	handoff: function (h, from, to) {
		handoffs++;

		return h;
	},
	layer: function (i, h, cache, pos) {
		if (i == 2 && cache.length != pos)
			die(`layer ${i}: cache at ${cache.length}, called at ${pos}`);

		return [ h[0].add(10.0 ** i) ];
	}
};

let resets = 0, captured = null, packed = {}, zeroed = [], specs = [], unpacked = [];

// entries (a conversation parked on the worker): two tensors per entry
mock.entry_pack = function (cache, n) {
	return { n: n, bytes: 40, tensors: { 'l0.ckv': torch.arange(n * 2).to('float32').reshape([ n, 2 ]), 'l1.r0.rec': torch.arange(3).to('float32') } };
};
mock.entry_unpack = function (cache, blob, start, live) {
	push(unpacked, [ blob.n, start, live, join(',', blob.tensors['l0.ckv'].tolist()[0]), join(',', blob.tensors['l1.r0.rec'].tolist()) ]);
};

mock.spec_alloc = (cache, W) => push(specs, `alloc ${W}`);
mock.spec_restore = (cache, a) => push(specs, `restore ${a}`);

// the states of layer 1 on the worker: a byte tensor that travels both ways
mock.state_pack = (cache, i) => torch.from_array([ i, 7, 8, 9, 200, 255 ], 'uint8');
mock.state_unpack = function (cache, i, bytes) {
	packed[i] = bytes.tolist();
};
mock.state_zero = (cache, i) => push(zeroed, i);

mock.reset_cache = function (cache) {
	resets++;
	cache.length = 0;
};
mock.graphs = function (cache, sizes) {
	captured = sizes;

	return {
		has: (n) => index(sizes, n) >= 0,
		run: function (ids, pos) {
			cache.length = pos + length(ids);

			// "replay": the same arithmetic as the eager layers 0..2 on the embedded ids
			let n = length(ids);
			let h = torch.from_array(ids, 'float32').reshape([ n, 1, 1 ]).add(pos * 0.5 + 111).expand([ n, HC, HID ]).contiguous();

			return { h: h };
		}
	};
};

let p = pair(), A = p[0], B = p[1];
let W = stage.worker(B, mock), K = stage.client(A);

K.pump = () => W.step(B.recv(1000));

let cid = K.new_cache(64);

check('new_cache answers an id', type(cid) == 'int' && cid >= 1);

let r = K.forward(cid, 0, [ 3, 5, 7 ]);

// the layers add 1 + 10 + 100 to the embedded id + pos * 0.5
check('forward returns n, pos and the streams', r.n == 3 && r.pos == 0 && join(',', r.h.shape()) == `3,${HC},${HID}`);
check('...computed by the worker\'s own layers', r.h.tolist()[1][2][3] == 5 + 111 && r.h.tolist()[2][0][0] == 7 + 111,
	sprintf('%J', r.h.tolist()[1][2][3]));
check('...with the hand-off between its two stages', handoffs == 1);

r = K.forward(cid, 3, [ 1 ]);
check('the next chunk at pos 3 sees the cache advanced', r.h.tolist()[0][0][0] == 1 + 1.5 + 111);

// a begin/end pair: two requests in flight
K.pump = null;
K.forward_begin(cid, 4, [ 9 ]);
K.forward_begin(cid, 5, [ 10 ]);
check('two requests in flight', K.inflight == 2);
W.step(B.recv(1000));
W.step(B.recv(1000));

let r1 = K.forward_end(), r2 = K.forward_end();

check('answered in order', r1.pos == 4 && r2.pos == 5 && K.inflight == 0);

K.pump = () => W.step(B.recv(1000));

// errors travel as error frames; the connection stays up
check('an unknown cache raises on the coordinator, naming it', raises(() => K.forward(99, 0, [ 1 ]), 'no cache 99'));
check('a position the cache is not at is the model\'s error, reported', raises(() => K.forward(cid, 100, [ 1 ]), 'cache at'));
check('...and the connection still works', K.stats().forwards == 4, sprintf('%J', K.stats()));

// graphs: captured on request, replayed with the graph flag, reset on demand
check('forward with graph before capture is an error', raises(() => K.forward(cid, 0, [ 1 ], true), 'no graph'));
K.graphs(cid, [ 1, 3 ]);
check('graphs captured the sizes asked for', join(',', captured) == '1,3');

K.reset(cid);
check('reset reaches the model', resets == 1);

// (the streams are a view of the connection's buffer: read each before the next call)
let g1 = K.forward(cid, 0, [ 4 ], true).h.tolist()[0][3][7];
let g3 = K.forward(cid, 1, [ 1, 2, 3 ], true).h.tolist()[2][0][0];

check('a graph replay answers the same streams as the eager layers', g1 == 4 + 111 && g3 == 3 + 0.5 + 111);
check('...and counts as a replay', K.stats().replays == 2);
check('a step size without a graph is an error', raises(() => K.forward(cid, 4, [ 1, 2 ], true), 'no graph for 2'));

// speculative windows: per-position states allocated once, restored after a rejection
K.spec_alloc(cid, 4);
K.spec_restore(cid, 2);
check('spec_alloc / spec_restore reach the model', join(',', specs) == 'alloc 4,restore 2');

// entries: parked on the worker, restored, dropped, saved to a file and loaded back
let ex = K.entry_export(cid, 5, '5:11:22');

check('entry_export answers an id and the bytes', type(ex.entry) == 'int' && ex.bytes == 40);
K.entry_import(cid, ex.entry, 3, true);
check('entry_import hands the blob, start and live flag to the model', join(';', map(unpacked[0], (x) => `${x}`)) == '5;3;true;0,1;0,1,2');
check('an unknown entry is an error', raises(() => K.entry_import(cid, 999, 1, false), 'no entry 999'));

let sv = K.entry_save(ex.entry, 'unit-test');

check('entry_save writes a file and reports the bytes', sv.bytes == 40 && fs.stat('/tmp/Ampstein-wcache-unit-test.safetensors') != null);
K.entry_drop(ex.entry);
check('...the dropped entry is gone', raises(() => K.entry_import(cid, ex.entry, 1, false), 'no entry'));

check('entry_load refuses another conversation (checksum) and consumes the file',
	raises(() => K.entry_load('unit-test', 5, '5:99:99'), 'another conversation') && fs.stat('/tmp/Ampstein-wcache-unit-test.safetensors') == null);

K.entry_save(K.entry_export(cid, 5, '5:11:22').entry, 'unit-test2');

let ld = K.entry_load('unit-test2', 5, '5:11:22');

check('entry_load restores the entry and consumes the file', ld.bytes == 40 && fs.stat('/tmp/Ampstein-wcache-unit-test2.safetensors') == null);
K.entry_import(cid, ld.entry, 5, true);
check('...with the saved tensors', join(';', map(unpacked[length(unpacked) - 1], (x) => `${x}`)) == '5;5;true;0,1;0,1,2');
check('a bad key is an error', raises(() => K.entry_load('../etc', 5, 'x'), 'bad key'));
check('a missing file is an error', raises(() => K.entry_load('nothing-here', 5, 'x'), 'no saved entry'));

// recurrent state of a layer travels as bytes
check('state_get returns the packed bytes of a layer', join(',', K.state_get(cid, 4).tolist()) == '4,7,8,9,200,255');
K.state_put(cid, 3, torch.from_array([ 1, 2, 3, 250 ], 'uint8'));
check('state_put hands the bytes to the model', join(',', packed[3]) == '1,2,3,250');
K.state_zero(cid, 5);
check('state_zero reaches the model', join(',', zeroed) == '5');
check('state_put without bytes is an error', raises(() => { A.send({ t: 'state_put', cache: cid, layer: 3 }); K.pump(); let f = A.recv(1000); if (f.meta.t == 'error') die(f.meta.message); }, 'no bytes'));

K.free(cid);
check('free drops the cache', raises(() => K.forward(cid, 0, [ 1 ]), 'no cache'));
check('stats count the caches', K.stats().caches == 0);

let before = W.forwards;

check('an unknown command is an error, not a crash', raises(() => {
	A.send({ t: 'nonsense' });
	K.pump();
	K.stats = K.stats;
	let f = A.recv(1000);

	if (f.meta.t == 'error')
		die(f.meta.message);
}, 'unknown command'));

// bye
A.send({ t: 'bye' });
check('bye ends the serve loop', W.step(B.recv(1000)) == 'bye');

// the serve loop itself, on a worker whose connection drops
let q = pair();
let W2 = stage.worker(q[1], mock);

q[0].close();
check('serve raises when the coordinator is gone', raises(() => W2.serve()));

if (fails) {
	warn(`${fails} failures\n`);
	exit(1);
}

warn('rpc_stage: ok\n');
