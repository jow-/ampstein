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
| `AMP_MODEL_QWEN35` | Qwen3.5 / 3.8 tests |
| `AMP_MODEL_GLM5` | GLM-5.3-Flash tests |
| `AMP_MODEL_QWEN4` | Qwen3.8-Flash-Next tests |

## Fixtures

Run the generators from the repository root with a Python environment that has
torch and exllamav3 installed:

```sh
python3 tests/oracle/make_fixture.py              # st_reader
python3 tests/oracle/make_exl3_fixture.py         # exl3
python3 tests/oracle/make_exl3_real_fixture.py    # exl3_real, exl3_recon
python3 tests/oracle/make_exl3_stack_oracle.py    # exl3_stack
python3 tests/oracle/make_shard_fixture.py        # weightmap
python3 tests/oracle/make_llama_logits_oracle.py  # llama_*, graphdec*
python3 tests/oracle/make_kld_oracle.py           # kld
python3 tests/oracle/make_chat_prompt_oracle.py   # chat_prompt
python3 tests/oracle/make_tokenizer_oracle.py     # tokenizer
python3 tests/oracle/make_chat_qwen35_oracle.py   # utpl_template, chat_jinja, chat_qwen35
python3 tests/oracle/make_qwen35_layer_oracle.py  # qwen35_layers
python3 tests/oracle/make_qwen35_full_oracle.py   # qwen35_full, qwen35_mtp
python3 tests/oracle/make_vision_oracle.py        # vision_tower, qwen35_vision
python3 tests/oracle/make_mrope_oracle.py         # mrope
python3 tests/oracle/make_glm5_oracle.py          # glm5_blocks .. glm5_mtp
python3 tests/oracle/make_glm5_long_oracle.py     # glm5_long
python3 tests/oracle/make_glm5_ppl_oracle.py      # glm5_ppl
python3 tests/oracle/make_glm5_vision_oracle.py   # glm5_vision
python3 tests/oracle/make_chat_glm5_oracle.py     # glm5_chat, chat_jinja, utpl_template
python3 tests/oracle/make_qwen4_oracle.py         # qwen4_full, qwen4_mtp, qwen4_tp
python3 tests/oracle/make_qwen4_depth_oracle.py   # qwen4_depth*
python3 tests/oracle/make_qwen4_vision_oracle.py  # qwen4_vision
```

Each script documents its arguments (model directory, output) in its docstring.

## Conventions

- Name test files distinctly from the modules they import: a test called
  `weights.uc` would import itself (`docs/LESSONS.md` L36).
- Pass `-L` directories as absolute paths (CTest does).
- Every binding raises on wrong arguments (`bindings_errors.uc`).
