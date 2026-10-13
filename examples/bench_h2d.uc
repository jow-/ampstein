// examples/bench_h2d.uc — pinned host -> device copy bandwidth (the expert-streaming budget,
// docs/EXPERT_STREAMING.md). Run one instance per GPU with CUDA_VISIBLE_DEVICES to test links
// concurrently.
let dev = int(ARGV[0] ?? 0);
function bw(label, MB, reps) {
	let h = torch.pinned_empty([MB * 1048576], 'uint8');
	let d = torch.zeros([MB * 1048576], 'uint8', 'cuda:' + dev);
	for (let i = 0; i < 3; i++) d.copy_(h);
	torch.synchronize();
	let c0 = clock(true);
	for (let i = 0; i < reps; i++) d.copy_(h);
	torch.synchronize();
	let c1 = clock(true);
	let s = (c1[0] - c0[0]) + (c1[1] - c0[1]) / 1e9;
	printf('%-12s %4d MB x%d: %.2f GB/s\n', label, MB, reps, MB * reps / 1024.0 / s);
}
bw('H2D 9 MB', 9, 50);
bw('H2D 64 MB', 64, 20);
bw('H2D 512 MB', 512, 5);
