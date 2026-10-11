// Phase 0.4 test: vendored EXL3 kernels vs the exllamav3 oracle.
// Fixtures: tests/fixtures/exl3_K{2,4,8}.safetensors (deterministic;
// regenerate with `.venv/bin/python tests/oracle/make_exl3_fixture.py`).
//
// Checks, per bitrate K:
//   - exl3_reconstruct: bit-exact against the oracle's reconstruct() output
//     (W_dequant). Dequantized tile math must match the reference
//     implementation exactly, not approximately.
//   - exl3_gemm: matches the oracle reference Y_orig = X @ W_orig within
//     fp16 accumulation tolerance (rel-rms < 0.02, the same bound upstream's
//     tests/test_sm75_gemm.py uses). Orientation: C = A @ dequant(B) with
//     the weight as (k, n) — no transpose.
//
// Run via CTest or manually:
//   ucode -L <build-dir> tests/exl3.uc

const FIXTURES = `${sourcepath(0, true)}/fixtures`;
const KS = [2, 4, 8];
const K_BITS = 256, K_OUT = 256, K_M = 8;

// squared relative-RMS error bound, identical to exllamav3's own gemm
// tests (rel-rms < 0.02); squared to avoid needing the math module
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

for (let ki = 0; ki < length(KS); ki++) {
    let K = KS[ki];
    let f = torch.st(`${FIXTURES}/exl3_K${K}.safetensors`);

    let trellis = f.tensor('trellis').cuda();
    let suh = f.tensor('suh').cuda();
    let svh = f.tensor('svh').cuda();
    let X = f.tensor('X').cuda();
    let W_dq = f.tensor('W_dequant').cuda();

    // --- reconstruct: bit-exact dequantization ---------------------------
    let unpacked = torch.zeros([K_BITS, K_OUT], 'float16').cuda();
    torch.exl3_reconstruct(unpacked, trellis, K);

    let got = unpacked.tolist();
    let want = W_dq.tolist();
    let bitexact = true;

    for (let i = 0; i < K_BITS && bitexact; i++)
        for (let j = 0; j < K_OUT && bitexact; j++)
            if (got[i][j] !== want[i][j])
                bitexact = false;

    assert(bitexact, `K=${K}: reconstruct not bit-exact vs oracle W_dequant`);

    // --- gemm: quantized matmul vs oracle reference ----------------------
    let C = torch.zeros([K_M, K_OUT], 'float16').cuda();
    torch.exl3_gemm(X, trellis, C, suh, svh);

    let c = C.tolist();
    let y = f.tensor('Y_orig').tolist();
    let rel2 = rms2_err(c, y);

    assert(rel2 < REL_RMS2_LIMIT,
        `K=${K}: gemm rel-rms^2 error ${rel2} exceeds ${REL_RMS2_LIMIT}`);

    printf('exl3 K=%d: reconstruct bit-exact, gemm rel-rms^2 %.8f\n', K, rel2);

    f.free();
}

printf('exl3: all assertions passed\n');