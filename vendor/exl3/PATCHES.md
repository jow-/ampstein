# Vendored-code patches

Upstream: https://github.com/turboderp-org/exllamav3 @ d3739fd (v1.5.3)

Patches are build adaptations only — kernel math is untouched (AGENTS.md
invariant 3). Re-apply after any vendor refresh.

1. `exllamav3_ext/graph.cuh` — removed unused `#include <pybind11/pybind11.h>`
   and `namespace py = pybind11;`. The include is vestigial upstream (no
   py:: uses in the file) and drags Python.h into every CUDA TU, which the
   ampstein build does not have.

2. `exllamav3_ext/ngram.cu`, `ple.cu` — removed the pybind11 include, the
   `namespace py = pybind11;` alias and the `py::gil_scoped_release`
   statements (no Python in the host process; math untouched).

3. `exllamav3_ext/quant/exl3_moe_coop*.cu*`, `comp_units/exl3_moe_coop_inst_dyn.cu` — mixed bit widths per
   expert for the fused decode MoE (the DeepSeek-V4.1 SAGE packs mix K = 1..6 inside one layer).
   `MoeCoopParams` gets `g_bits / u_bits / d_bits` (uint8, one entry per local expert); when all three are
   set the launcher picks the `bits == 0` kernel instances, whose blocks switch on the expert's width and
   run the unchanged static `gemv_tile<K, mul1, WIDE>` body of that width. The static-K kernels and
   their numerics are untouched; `exl3_moe_coop` takes the three tables as optional trailing arguments.

4. `exllamav3_ext/quant/exl3_moe*.cu*` — the same for the token-sorted prefill MoE (`exl3_moe`): optional
   `bits_g / bits_u / bits_d` tables (uint8, per expert); the K = 0 (run-time switch) instances read
   the expert's widths from them (`k_*_e` kernel arguments). Without tables nothing changes.
