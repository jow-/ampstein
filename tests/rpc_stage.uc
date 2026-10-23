// rpc_stage.uc: the worker's command loop and the coordinator's client of the remote first
// stages (docs/RPC.md phase 4), against a mock model on the CPU: one process, loopback TCP,
// the worker driven step by step through K.pump. The GPU model is checked by glm5_split.uc and
// the two-host run.
//   ucode -L <build-dir>/lib -L src/ucode -ltorch=ampstein.torch tests/rpc_stage.uc

import * as socket from 'socket';
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
