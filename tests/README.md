# Tests

Everything runs through CTest:

```sh
ctest --test-dir build --output-on-failure
```

- `tests/*.uc` — ucode tests run with the freshly built `torch.so`
  (`-L build/lib -L src/ucode -ltorch=ampstein.torch`). The oracle-backed ones
  compare ids, logits and layer outputs with fixtures generated from exllamav3
  or the Hugging Face tokenizers.
- `tests/oracle/*.py` — fixture generators (Python, test-time only, never
  shipped). They run exllamav3 / torch / tokenizers as the reference and write
  `tests/fixtures/` (gitignored).

A test that needs a model or a fixture that is not there exits with code 125,
which CTest reports as *skipped*. GPU tests are `RUN_SERIAL` (one model in
VRAM at a time); run them on an idle GPU (`docs/LESSONS.md` L31, L77).

## Models

Models are found through `AMP_MODEL_*` (the default is a directory under
`/data/models`). Further per-test knobs are listed in `docs/ENVIRONMENT.md`.

| Variable | Used by |
|---|---|
| `AMP_MODEL_1B` | llama tests (Llama-3.2-1B-Instruct EXL3) |

## Fixtures

Run the generators from the repository root with a Python environment that has
torch and exllamav3 installed:

```sh
python3 tests/oracle/make_fixture.py              # st_reader
```

Each script documents its arguments (model directory, output) in its docstring.

## Conventions

- Name test files distinctly from the modules they import: a test called
  `weights.uc` would import itself (`docs/LESSONS.md` L36).
- Pass `-L` directories as absolute paths (CTest does).
- Every binding raises on wrong arguments (`bindings_errors.uc`).
