#!/usr/bin/env python3
# Oracle for tests/chat_qwen35.uc: renders the checkpoint's own
# chat_template.jinja with jinja2 (configured like HF transformers:
# trim_blocks, lstrip_blocks, tojson = json.dumps(ensure_ascii=False))
# for a battery of conversations, and stores cases + expected strings.
#
# Test-time only (invariant 1). Usage:
#   .venv/bin/python tests/oracle/make_chat_qwen35_oracle.py [model-dir]

import copy, json, os, sys
import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

model = sys.argv[1] if len(sys.argv) > 1 else '/data/models/Qwen3.8-27B-exl3-5.00bpw'
fdir = os.path.join(os.path.dirname(__file__), '..', 'fixtures')
out = os.path.join(fdir, 'chat_qwen35_cases.json')
tpl_path = os.path.join(fdir, 'chat_qwen35_template.jinja')

def raise_exception(msg):
    raise jinja2.exceptions.TemplateError(msg)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                    extensions=['jinja2.ext.loopcontrols'])
env.filters['tojson'] = lambda x, indent=None, separators=None, sort_keys=False: \
    json.dumps(x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
env.globals['raise_exception'] = raise_exception
tpl = env.from_string(open(os.path.join(model, 'chat_template.jinja')).read())
# the transpiler test (tests/utpl_template.uc) renders this exact source,
# so the test stays self-contained without the model directory
open(tpl_path, 'w').write(open(os.path.join(model, 'chat_template.jinja')).read())

tools = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path to the file"},
            "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command — returns stdout.",
        "parameters": {"type": "object", "properties": {
            "cmd": {"type": "string"}, "env": {"type": "object"}}}}},
]

def call(name, args):
    return {"id": "call_x", "type": "function",
            "function": {"name": name, "arguments": args}}

cases = [
    {"name": "plain", "messages": [{"role": "user", "content": "Hello!"}]},
    {"name": "system", "messages": [
        {"role": "system", "content": "  You are terse.  "},
        {"role": "user", "content": "Hi"}]},
    {"name": "no_think", "kwargs": {"enable_thinking": False},
     "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]},
    {"name": "effort_low", "kwargs": {"reasoning_effort": "low"},
     "messages": [{"role": "user", "content": "Q"}]},
    {"name": "effort_medium", "kwargs": {"reasoning_effort": "medium"},
     "messages": [{"role": "user", "content": "Q"}]},
    {"name": "multi_turn", "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "simple\n"},
        {"role": "user", "content": [{"type": "text", "text": "And 3+3?"}]}]},
    {"name": "tools", "tools": tools, "messages": [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "Show me README.md"},
        {"role": "assistant", "content": "Let me look.",
         "tool_calls": [call("read_file", {"path": "README.md", "limit": 20})]},
        {"role": "tool", "tool_call_id": "call_x", "content": "# Title\nbody"},
        {"role": "tool", "tool_call_id": "call_y", "content": "second"},
        {"role": "assistant", "content": "",
         "tool_calls": [call("run", {"cmd": "ls -la", "env": {"A": "1"}}),
                        call("read_file", {"path": "ä.txt"})]},
        {"role": "tool", "tool_call_id": "call_z", "content": "ok"},
        {"role": "user", "content": "thanks"}]},
    {"name": "tools_no_system", "tools": tools, "kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": "x"}]},
    {"name": "preserve_off", "kwargs": {"preserve_thinking": False}, "messages": [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b", "reasoning_content": "r1"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d", "reasoning_content": "r2"},
        {"role": "tool", "content": "t"}]},
]

for c in cases:
    kw = dict(c.get("kwargs", {}))
    # HF passes tool-call arguments as mappings; the ucode side gets the
    # OpenAI wire format (a JSON string) and must convert
    r = tpl.render(messages=c["messages"], tools=c.get("tools"),
                   add_generation_prompt=True, **kw)
    c["expected"] = r
    wire = copy.deepcopy(c["messages"])
    for m in wire:
        for tc in m.get("tool_calls", []) or []:
            tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
    c["messages"] = wire

json.dump(cases, open(out, 'w'), ensure_ascii=False, indent=1)
print(f"wrote {len(cases)} cases to {out}")
print(f"wrote template to {tpl_path}")
