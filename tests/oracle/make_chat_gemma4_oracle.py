#!/usr/bin/env python3
"""Chat-format oracle for Gemma 4 (tests/chat_gemma4.uc): conversations rendered by the checkpoint's own
chat_template.jinja with jinja2, in the OpenAI wire format ampstein receives them.

  python3 tests/oracle/make_chat_gemma4_oracle.py [model_dir]  ->  tests/fixtures/chat_gemma4_oracle.json
"""
import copy
import json
import os
import sys

import jinja2

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/models/gemma-4-26B-A4B-it-exl3-6.10bpw"
OUT = os.path.join(os.path.dirname(__file__), "..", "fixtures", "chat_gemma4_oracle.json")

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string", "description": "The city"},
         "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
         "days": {"type": "integer"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search",
     "parameters": {"type": "object", "properties": {
         "q": {"type": "string"},
         "opts": {"type": "object", "properties": {"limit": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}}}},
      "required": ["q"]}}},
]


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CASES = [
    dict(name="plain", messages=[{"role": "user", "content": "Hi"}]),
    dict(name="system", messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]),
    dict(name="thinking", think=True, messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "2+2?"}]),
    dict(name="history", messages=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"},
                                    {"role": "user", "content": "Capital of France?"}]),
    dict(name="tools", tools=TOOLS, messages=[{"role": "user", "content": "Weather in Paris?"}]),
    dict(name="tools and system", tools=TOOLS, think=True,
         messages=[{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"}]),
    dict(name="tool call and result", tools=TOOLS,
         messages=[{"role": "user", "content": "Weather in Paris?"},
                   {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Paris", "unit": "celsius"}, "c1")]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"temp\": 18, \"sky\": \"cloudy\"}"}]),
    dict(name="tool result answer and new turn", tools=TOOLS,
         messages=[{"role": "user", "content": "Weather in Paris?"},
                   {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Paris"}, "c1")]},
                   {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                   {"role": "assistant", "content": "It is sunny."},
                   {"role": "user", "content": "And Rome?"}]),
    dict(name="parallel calls", tools=TOOLS,
         messages=[{"role": "user", "content": "Both please"},
                   {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Rome"}, "a"),
                                                                       call("search", {"q": "rome food", "opts": {"limit": 3, "tags": ["a", "b"]}}, "b")]},
                   {"role": "tool", "tool_call_id": "a", "content": "sunny"},
                   {"role": "tool", "tool_call_id": "b", "content": "pizza"}]),
    dict(name="content parts", messages=[{"role": "user", "content": [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]}]),
]


def raise_exception(msg):
    raise Exception(msg)


env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
env.globals["raise_exception"] = raise_exception
tpl = env.from_string(open(os.path.join(MODEL_DIR, "chat_template.jinja")).read())

def to_hf(messages):
    """OpenAI wire format -> what the template reads: tool-call arguments as mappings, the tool messages after an
    assistant message with calls become its `tool_responses`, and the answer that follows continues the same turn."""
    out = []
    i = 0
    msgs = copy.deepcopy(messages)
    while i < len(msgs):
        m = msgs[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            names = {tc["id"]: tc["function"]["name"] for tc in m["tool_calls"]}
            for tc in m["tool_calls"]:
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
            res = []
            while i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool":
                t = msgs[i + 1]
                try:
                    v = json.loads(t["content"])
                    v = v if isinstance(v, dict) else t["content"]
                except Exception:
                    v = t["content"]
                res.append({"name": names[t["tool_call_id"]], "response": v})
                i += 1
            m["tool_responses"] = res
            if res and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant" and not msgs[i + 1].get("tool_calls"):
                m["content"] = msgs[i + 1]["content"]
                i += 1
        out.append(m)
        i += 1
    return out


out = []
for c in CASES:
    ctx = dict(messages=to_hf(c["messages"]), tools=c.get("tools"), add_generation_prompt=True,
               bos_token="<bos>", enable_thinking=bool(c.get("think")))
    if not c.get("tools"):
        ctx.pop("tools")
    out.append(dict(name=c["name"], messages=c["messages"], tools=c.get("tools"), think=bool(c.get("think")),
                    expect=tpl.render(**ctx)))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print(f"wrote {len(out)} cases to {OUT}")
