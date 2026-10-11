# Oracle: load the Qwen3.8-27B exl3 model with exllamav3 and dump reference
# logits / GDN intermediates for the Ampstein port. Test-time only.
import os, sys, json
os.environ.setdefault("EXLLAMAV3_TUNE_CACHE", "/tmp/exl3_tune_qwen35")
import torch
from exllamav3 import Config, Model, Tokenizer

model_dir = "/data/models/Qwen3.8-27B-exl3-5.00bpw"
outdir = "tests/fixtures"
os.makedirs(outdir, exist_ok=True)

config = Config.from_directory(model_dir)
model = Model.from_config(config)
tokenizer = Tokenizer(config)

print("loading model (autosplit over available GPUs)...", flush=True)
model.load()
print("loaded.", flush=True)

prompt = "The capital of France is"
ids = tokenizer.encode(prompt, add_bos=True)
print("ids:", ids.tolist() if hasattr(ids, 'tolist') else ids, flush=True)

with torch.inference_mode():
    logits = model.forward(ids)
print("logits", logits.shape, logits.dtype, flush=True)
lp = logits[0, -1, :].float().cpu()
top = torch.topk(lp, 10)
print("top10:", [(int(i), float(v)) for v, i in zip(top.values, top.indices)], flush=True)

torch.save({
    "ids": ids.cpu(),
    "logits": logits.float().cpu(),
}, f"{outdir}/qwen35_oracle_logits.pt")
print("saved fixture.", flush=True)

model.unload()
print("done.", flush=True)