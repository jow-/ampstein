// Phase 0.5 test: one EXL3 layer from a REAL model dir vs the exllamav3
// oracle. Fixture: tests/fixtures/exl3_real_layer.safetensors — the q_proj
// of layer 0 from turboderp/Llama-3.2-1B-Instruct-exl3 (4.0bpw branch),
// with reference outputs from exllamav3 itself. Regenerate with
// `.venv/bin/python tests/oracle/make_exl3_real_fixture.py` (needs the
// model dir; see the script header).
//
// Checks:
//   - exl3_reconstruct: bit-exact against the oracle's plain reconstruct()
//     output (W_dequant, Hadamard basis without scales) for a real
//     2048x2048 K=4 layer — real calibrated scales, not synthetic ones.
//   - exl3_gemm: matches the oracle reference Y = X @ W_orig within
//     fp16 accumulation tolerance (rel-rms < 0.02, upstream's bound);
//     W_orig is reconstruct_had_slice() output, the basis the gemm
//     consumes with suh/svh (no transpose — L15).
//
// Run via CTest or manually:
//   ucode -L <build-dir> tests/exl3_real.uc

const FIXTURE = `${sourcepath(0, true)}/fixtures/exl3_real_layer.safetensors`;
const K = 4, K_BITS = 2048, K_OUT = 2048, K_M = 8;

// squared relative-RMS error bound, identical to exllamav3's own gemm tests
const REL_RMS2_LIMIT = 0.02 * 0.02;

function rms2_err(c, y) {
    // mean((c - y)^2) / mean(y^2)  ==  rel-rms error, squared
    let se = 0.0, sy = 0.0, n = 0;

    for (let i = 0; i < length(c); i++)
        for (let j = 0; j < length(c[i]); j++) {
            let d = c[i][j] - y[i][j];
            se += d * d;
            sy += y[i][j] * y[i][j];
            n++;
        }

    return (se / n) / (sy / n);
}

let f = torch.st(FIXTURE);

let trellis = f.tensor('trellis').cuda();
let suh = f.tensor('suh').cuda();
let svh = f.tensor('svh').cuda();
let X = f.tensor('X').cuda();
let W_dq = f.tensor('W_dequant').cuda();

// sanity: the fixture must carry the shapes the test assumes
assert(length(trellis.shape()) == 3, 'trellis is 3-D');
assert(trellis.shape()[0] == K_BITS / 16, 'trellis k/16');
assert(trellis.shape()[1] == K_OUT / 16, 'trellis n/16');
assert(trellis.shape()[2] == 16 * K, 'trellis 16*K');

// --- reconstruct: bit-exact dequantization of a real layer ---------------
let unpacked = torch.zeros([K_BITS, K_OUT], 'float16').cuda();
torch.exl3_reconstruct(unpacked, trellis, K);

let got = unpacked.tolist();
let want = W_dq.tolist();
let bitexact = true;

for (let i = 0; i < K_BITS && bitexact; i++)
    for (let j = 0; j < K_OUT && bitexact; j++)
        if (got[i][j] !== want[i][j])
            bitexact = false;

assert(bitexact, 'reconstruct not bit-exact vs oracle W_dequant');

// --- gemm: quantized matmul vs oracle reference ---------------------------
let C = torch.zeros([K_M, K_OUT], 'float16').cuda();
torch.exl3_gemm(X, trellis, C, suh, svh);

let rel2 = rms2_err(C.tolist(), f.tensor('Y').tolist());

assert(rel2 < REL_RMS2_LIMIT,
    `gemm rel-rms^2 error ${rel2} exceeds ${REL_RMS2_LIMIT}`);

printf('exl3 real layer (2048x2048 K=4): reconstruct bit-exact, gemm rel-rms^2 %.8f\n', rel2);

f.free();

printf('exl3_real: all assertions passed\n');