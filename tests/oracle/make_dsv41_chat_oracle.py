#!/usr/bin/env python3
"""Chat-format oracle for DeepSeek-V4.1-Flash (tests/chat_dsv41.uc): conversations rendered by DeepSeek's
own encoding/encoding.py, in the OpenAI wire format ampstein receives them.

  python3 make_dsv41_chat_oracle.py <encoding_dir> <out.json>

<encoding_dir>: DeepSeek-V4.1-Flash/encoding (encoding.py). Top-level tools are attached to the leading system
message (a new empty one when there is none), as the Hugging Face chat template does.
"""
import copy
import json
import sys

sys.path.insert(0, sys.argv[1])
import encoding  # noqa: E402

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                    "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "description": "Web search",
     "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
]


def call(name, args, cid):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CASES = [
    dict(name="plain chat", messages=[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]),
    dict(name="no system, thinking", mode="thinking",
         messages=[{"role": "user", "content": "What is 2+2?"}]),
    dict(name="system, thinking, effort max", mode="thinking", effort="max",
         messages=[{"role": "system", "content": "You are a calculator."}, {"role": "user", "content": "2+2"}]),
    dict(name="effort 33", mode="thinking", effort=33,
         messages=[{"role": "user", "content": "x"}]),
    dict(name="history with reasoning, thinking", mode="thinking",
         messages=[{"role": "user", "content": "Hello"},
                   {"role": "assistant", "reasoning_content": "greet back", "content": "Hi!"},
                   {"role": "user", "content": "Capital of France?"},
                   {"role": "assistant", "reasoning_content": "Paris", "content": "Paris."}]),
    dict(name="history with reasoning, chat",
         messages=[{"role": "user", "content": "Hello"},
                   {"role": "assistant", "reasoning_content": "greet back", "content": "Hi!"},
                   {"role": "user", "content": "Capital of France?"}]),
    dict(name="tools with system", mode="thinking", tools=TOOLS,
         messages=[{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"}]),
    dict(name="tools without system", mode="thinking", tools=TOOLS,
         messages=[{"role": "user", "content": "Weather in Paris?"}]),
    dict(name="tool call and result", mode="thinking", tools=TOOLS,
         messages=[{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Weather in Paris?"},
                   {"role": "assistant", "reasoning_content": "call it", "content": "",
                    "tool_calls": [call("get_weather", {"city": "Paris", "days": 2}, "c1")]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"t\": 12}"}]),
    dict(name="parallel calls, results out of order, new user turn", mode="thinking", tools=TOOLS,
         messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Both please"},
                   {"role": "assistant", "reasoning_content": "two", "content": "ok",
                    "tool_calls": [call("get_weather", {"city": "Rome"}, "a"), call("search", {"q": "rome food"}, "b")]},
                   {"role": "tool", "tool_call_id": "b", "content": "pizza"},
                   {"role": "tool", "tool_call_id": "a", "content": "sunny"},
                   {"role": "assistant", "reasoning_content": "done", "content": "Sunny, pizza."},
                   {"role": "user", "content": "thanks"}]),
    dict(name="developer message and string args", mode="chat",
         messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "q"},
                   {"role": "assistant", "content": "", "tool_calls": [call("search", {"q": "x", "n": [1, 2], "flag": True}, "z")]},
                   {"role": "tool", "tool_call_id": "z", "content": "r"}]),
    dict(name="content parts", mode="chat",
         messages=[{"role": "user", "content": [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]}]),
    dict(name="latest reminder", mode="chat",
         messages=[{"role": "system", "content": "S"}, {"role": "latest_reminder", "content": "2026-02-21"},
                   {"role": "user", "content": "q"}]),
]

out = []
for c in CASES:
    msgs = copy.deepcopy(c["messages"])
    if c.get("tools"):
        if msgs[0]["role"] == "system" and not msgs[0].get("tools"):
            msgs[0]["tools"] = c["tools"]
        else:
            msgs = [{"role": "system", "content": "", "tools": c["tools"]}] + msgs
    # encoding.py takes tool-call arguments as JSON strings, as received on the wire
    prompt = encoding.encode_messages(msgs, thinking_mode=c.get("mode", "chat"), reasoning_effort=c.get("effort"))
    out.append({"name": c["name"], "messages": c["messages"], "tools": c.get("tools"),
                "mode": c.get("mode", "chat"), "effort": c.get("effort"), "expect": prompt})
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[2], len(out))
