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
