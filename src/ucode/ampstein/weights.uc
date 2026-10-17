// ucode/weights.uc — model weight resolution over safetensors files.
//
// Wraps one or more stfile resources behind a uniform key-addressed
// interface and handles both on-disk layouts:
//
//   - single file:  <dir>/model.safetensors
//   - sharded HF:   <dir>/model.safetensors.index.json whose
//                   "weight_map" routes every tensor key to a shard
//                   file (model-00001-of-00002.safetensors, ...)
//
// Shard stfiles are opened lazily on first access to that shard's keys
// (mmap'ed, zero-copy tensor views — no data is read until a tensor is
// consumed).
//
// Usage:
//   import * as weights from 'ampstein.weights';
//   let ws = weights.open('/path/to/model-dir');
//   ws.has('model.embed_tokens.weight');   // -> true/false
//   ws.tensor('model.norm.weight');        // -> CPU tensor view
//   ws.fp16('model.norm.weight');          // -> CPU tensor, fp16 cast
//   ws.info('model.embed_tokens.weight');  // -> {shape, dtype}
//   ws.upload();                           // batched async upload of ALL
//                                          // tensors to the GPU
//   ws.gpu('model.norm.weight');           // -> CUDA tensor (cached)
//   ws.gpu16('model.norm.weight');         // -> CUDA tensor, fp16 cast
//
// GPU cache: gpu()/gpu16() return cached CUDA tensors, uploading on
// first touch (synchronously, one tensor at a time). upload() instead
// populates the cache for every tensor of the model in ONE batched
// pinned-staged async transfer (torch.h2d) — the load-time path; the
// per-tensor sync copies only run when upload() was not called.

import {readfile, stat} from 'fs';

const INDEX_NAME = 'model.safetensors.index.json';
const SINGLE_NAME = 'model.safetensors';

export function open(model_dir, only_file) {
	let shards = {};      // shard filename -> stfile resource (lazy)
	let map = null;       // sharded: key -> shard filename
	let single = null;    // single-file: stfile resource
	let keylist = null;      // cached key list (single-file)
	let gpu_raw = {};     // key -> CUDA tensor (uploaded dtype)
	let gpu_cast = {};    // key -> CUDA tensor cast to a named dtype

	if (only_file) {
		/* one safetensors file of the directory by itself (e.g. an
		 * mtp.safetensors, which is outside the index) */
		single = torch.st(model_dir + '/' + only_file);
		keylist = single.list();
	} else if (stat(model_dir + '/' + INDEX_NAME) != null) {
		let idx = json(readfile(model_dir + '/' + INDEX_NAME));

		map = idx.weight_map;

		if (!map)
			die(`weights: ${INDEX_NAME} has no weight_map`);
	} else {
		single = torch.st(model_dir + '/' + SINGLE_NAME);
		keylist = single.list();
	}

	function st_for(shard) {
		if (!shards[shard])
			shards[shard] = torch.st(model_dir + '/' + shard);

		return shards[shard];
	}

	/* Normalize a device argument: null/undefined -> 0; a number is a
	 * device index. (Strings "cuda:N" also work — the module parses
	 * them — but numbers keep the ucode side simple.) */
	function dev_of(d) {
		return (d == null) ? 0 : d;
	}

	function tensor_for(key) {
		if (map) {
			let shard = map[key];

			if (!shard)
				die(`weights: unknown key '${key}'`);

			return st_for(shard).tensor(key);
		}

		return single.tensor(key);
	}

	function info_for(key) {
		if (map) {
			let shard = map[key];

			if (!shard)
				die(`weights: unknown key '${key}'`);

			return st_for(shard).info(key);
		}

		return single.info(key);
	}

	function all_keys() {
		if (map)
			return keys(map);

		return keylist;
	}

	/* CUDA tensor for `key` (raw dtype), cached. dtype == null uploads
	 * as-is; a named dtype uploads raw (if not cached yet) and casts on
	 * the GPU — cheaper than a CPU-side cast and keeps the copy
	 * page-aligned. */
	function gpu_for(key, dtype, dev) {
		dev = dev_of(dev);

		if (!dtype) {
			let ck = key + '\x02' + dev;

			if (!gpu_raw[ck])
				gpu_raw[ck] = tensor_for(key).cuda(dev);

			return gpu_raw[ck];
		}

		let ck = key + '\x01' + dtype + '\x02' + dev;

		if (!gpu_cast[ck]) {
			let raw = gpu_for(key, null, dev);

			gpu_cast[ck] = (raw.dtype() == dtype) ? raw : raw.to(dtype);
		}

		return gpu_cast[ck];
	}

	/* Batched load-time upload: sequential file reads into a pinned
	 * staging ring with overlapping async device copies (torch.h2d_file)
	 * for every tensor of the model, populating the raw GPU cache.
	 * Avoids both the mmap page-fault path and per-tensor sync copies.
	 *
	 * devfn (optional): function(key) -> device index, for multi-GPU
	 * layer splits; tensors are uploaded per device in one batched
	 * transfer each. Default: everything to device 0. */
	function upload_all(devfn) {
		let ks = all_keys();
		let by_dev = {};
		let devlist = [];

		for (let i = 0; i < length(ks); i++) {
			let key = ks[i];
			let dev = devfn ? devfn(key) : 0;

			/* null/undefined: skip the tensor entirely (it stays on
			 * the host; a later gpu(key, dev) uploads it lazily) */
			if (dev == null)
				continue;

			dev = dev_of(dev);
			let st = map ? st_for(map[key]) : single;

			if (!by_dev[dev]) {
				by_dev[dev] = [];
				push(devlist, dev);
			}

			push(by_dev[dev], [st, key]);
		}

		for (let d = 0; d < length(devlist); d++) {
			let dev = devlist[d];
			let items = by_dev[dev];
			let ups = torch.h2d_file(items, dev);

			for (let i = 0; i < length(items); i++)
				gpu_raw[items[i][1] + '\x02' + dev] = ups[i];
		}
	}

	return {
		/* true when `key` resolves to a tensor in this model. */
		has: function (key) {
			if (map)
				return map[key] != null;

			return index(keylist, key) >= 0;
		},

		/* the stfile resource holding `key` (row gathers straight from the file) */
		file: function (key) {
			if (map) {
				let shard = map[key];

				if (!shard)
					die(`weights: unknown key '${key}'`);

				return st_for(shard);
			}

			return single;
		},

		/* CPU tensor view for `key` (zero-copy mmap view). */
		tensor: function (key) {
			return tensor_for(key);
		},

		/* tensor(key) cast to float16. Non-quantized weights in exl3
		 * repos may be bf16; casting at load keeps every activation
		 * fp16 end-to-end (mixed bf16 x fp16 ops would silently
		 * promote to fp32 and break the fp16 kernels). */
		fp16: function (key) {
			return this.tensor(key).to('float16');
		},

		/* {shape, dtype} for `key` without materializing a tensor. */
		info: function (key) {
			return info_for(key);
		},

		/* All tensor keys of the model (unordered). */
		list: function () {
			return all_keys();
		},

		/* Cached CUDA tensor for `key` (raw dtype), uploading on first
		 * touch when upload() has not (yet) covered it. dev: device
		 * index (default 0). */
		gpu: function (key, dev) {
			return gpu_for(key, null, dev);
		},

		/* Cached CUDA tensor for `key`, cast to fp16 on the GPU. */
		gpu16: function (key, dev) {
			return gpu_for(key, 'float16', dev);
		},

		/* Drop the cached raw-dtype GPU copy of `key` (after a cast copy
		 * was taken): frees the duplicate for big tables such as the
		 * embedding. The cast cache entry stays valid. */
		release: function (key, dev) {
			delete gpu_raw[key + '\x02' + dev_of(dev)];
		},

		/* Forget every cached GPU copy of `key` on device `dev` (raw and cast
		 * ones): frees staging copies a sharding loader has sliced up. */
		drop: function (key, dev) {
			dev = dev_of(dev);

			delete gpu_raw[key + '\x02' + dev];

			for (let dt in [ 'float16', 'float32', 'bfloat16' ])
				delete gpu_cast[key + '\x01' + dt + '\x02' + dev];
		},

		/* Load-time fast path: batched pinned-async upload of every
		 * tensor (see upload_all above). devfn: optional function(key)
		 * -> device index for multi-GPU splits. */
		upload: function (devfn) {
			upload_all(devfn);
		}
	};
}