#!/usr/bin/env python3
"""Oracle (test-time only): GLM-5.3 chat prompts rendered by the checkpoint's
own jinja template, for byte-exact comparison with ucode/chat_glm5.uc.

    python3 tests/oracle/make_chat_glm5_oracle.py [MODEL_DIR]

Writes tests/fixtures/chat_glm5.json: [{messages, tools, kwargs, prompt}, ...]
(rendering mirrors transformers: ImmutableSandboxedEnvironment, trim_blocks,
lstrip_blocks, loopcontrols, tojson without ASCII escaping).
"""
import json, os, sys
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/GLM-5.3-Flash-exl3-3.05bpw"
FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fixtures")
OUT = os.path.join(FIX_DIR, "chat_glm5.json")
TPL_OUT = os.path.join(FIX_DIR, "chat_glm5_template.jinja")


def tojson(x, ensure_ascii = False, indent = None, separators = None, sort_keys = False):
    return json.dumps(x, ensure_ascii = ensure_ascii, indent = indent, separators = separators,
                      sort_keys = sort_keys)


def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)


env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True,
                                    extensions = [jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(os.path.join(MODEL_DIR, "chat_template.jinja")).read())
# the transpiler test (tests/utpl_template.uc) renders this exact source,
# so the test stays self-contained without the model directory
open(TPL_OUT, "w").write(open(os.path.join(MODEL_DIR, "chat_template.jinja")).read())

weather = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                   "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
calc = {"type": "function", "function": {
    "name": "calc", "description": "Evaluate an expression",
    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}}}

U = lambda t: {"role": "user", "content": t}
A = lambda t, **k: {"role": "assistant", "content": t, **k}
S = lambda t: {"role": "system", "content": t}
tc = lambda i, n, a: {"id": i, "type": "function", "function": {"name": n, "arguments": a}}

cases = [
    dict(messages = [U("Hello")]),
    dict(messages = [U("Hello")], kwargs = {"reasoning_effort": "low"}),
    dict(messages = [U("Hello")], kwargs = {"reasoning_effort": "high"}),
    dict(messages = [S("You are terse."), U("Hi"), A("Hello!"), U("Capital of France?")]),
    dict(messages = [U("Q1"), A("A1", reasoning_content = "thinking one"), U("Q2")]),
    dict(messages = [U("Q1"), A("A1", reasoning_content = "thinking one"), U("Q2"),
                     A("A2", reasoning_content = "thinking two")]),
    dict(messages = [U("Q1"), A("A1", reasoning_content = "thinking one"), U("Q2")],
         kwargs = {"clear_thinking": False}),
    dict(messages = [A("<think>inline</think>body"), U("next")]),
    dict(messages = [U("Weather in Paris?")], tools = [weather]),
    dict(messages = [U("Weather in Paris and 2+2?")], tools = [weather, calc]),
    dict(messages = [U("Weather in Paris?"),
                     A("", tool_calls = [tc("call_1", "get_weather", {"city": "Paris", "unit": "c"})]),
                     {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"}],
         tools = [weather]),
    dict(messages = [U("Do both"),
                     A("Let me check.", tool_calls = [tc("a", "get_weather", {"city": "Paris"}),
                                                       tc("b", "calc", {"expr": "2+2"})]),
                     {"role": "tool", "tool_call_id": "b", "content": "4"},
                     {"role": "tool", "tool_call_id": "a", "content": "rain"}],
         tools = [weather, calc]),
    dict(messages = [U("Do both"),
                     A("", tool_calls = [tc(None, "get_weather", {"city": "Paris", "n": 3, "ok": True})]),
                     {"role": "tool", "content": "x"}],
         tools = [weather]),
    dict(messages = [{"role": "user", "content": [{"type": "text", "text": "part one "},
                                                  {"type": "text", "text": "part two"}]}]),
]

out = []
for c in cases:
    kw = c.get("kwargs", {})
    prompt = tpl.render(messages = c["messages"], tools = c.get("tools"), add_generation_prompt = True, **kw)
    out.append({"messages": c["messages"], "tools": c.get("tools"), "kwargs": kw, "prompt": prompt})

os.makedirs(os.path.dirname(OUT), exist_ok = True)
json.dump(out, open(OUT, "w"), ensure_ascii = False, indent = 1)
print("wrote", OUT, len(out), "cases")
print("wrote", TPL_OUT)
for o in out[:3]:
    print(repr(o["prompt"]))
