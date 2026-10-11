// ucode/kvcache.uc — contiguous fp16 KV cache for the llama family.
//
// One k/v tensor pair per layer, shape (batch, kv_heads, max_seq,
// head_dim), fp16 on GPU. Writes go through narrow() + copy_() (the
// in-place copy primitive); reads return narrowed views — no copies on
// the attention read path beyond what SDPA itself needs.
//
// Usage:
//   import * as kvcache from 'ampstein.kvcache';
//   let cache = kvcache.create(cfg, max_seq, batch);
//   cache.write(i, k, v, pos);      // append S slots at pos
//   let K = cache.keys(i, len);     // (batch, kvh, len, hd) view

/* cfg: parsed llama config (num_key_value_heads, head_dim,
 * num_hidden_layers); max_seq: capacity in positions; batch: batch size
 * (default 1). Returns the cache object; slots [0, length) are valid. */
export function create(cfg, max_seq, batch) {
	batch = batch || 1;

	let kvh = cfg.num_key_value_heads;
	let hd = cfg.head_dim;
	let nl = cfg.num_hidden_layers;

	let k = [], v = [];

	for (let i = 0; i < nl; i++) {
		push(k, torch.zeros([batch, kvh, max_seq, hd], 'float16').cuda());
		push(v, torch.zeros([batch, kvh, max_seq, hd], 'float16').cuda());
	}

	return {
		cfg: cfg,
		max_seq: max_seq,
		batch: batch,
		length: 0,
		k: k,
		v: v,

		reset: function () {
			this.length = 0;
		},

		/* Append k/v for S new positions starting at slot `pos`.
		 * k, v: (batch, kv_heads, S, head_dim) fp16 CUDA tensors.
		 * Advances length to pos + S (the cache is append-only). */
		write: function (i, k, v, pos) {
			let S = k.shape()[2];

			this.k[i].narrow(2, pos, S).copy_(k);
			this.v[i].narrow(2, pos, S).copy_(v);

			if (i == this.cfg.num_hidden_layers - 1)
				this.length = pos + S;
		},

		/* Views (batch, kv_heads, len, head_dim) of slots [0, len). */
		keys: function (i, len) {
			return this.k[i].narrow(2, 0, len);
		},

		values: function (i, len) {
			return this.v[i].narrow(2, 0, len);
		}
	};
}