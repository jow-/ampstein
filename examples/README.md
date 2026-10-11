# Examples

Small runnable ucode scripts. Run them with the module on the search path, e.g.

```sh
ucode -L build/lib -L src/ucode -ltorch=ampstein.torch examples/<script>.uc [-- args]
```

| Script | What |
|---|---|
| `version.uc` | the module loads and reports its libtorch version |
