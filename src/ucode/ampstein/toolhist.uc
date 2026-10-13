// ucode/toolhist.uc — the tool-call history of a chat request, for the
// --call-hook policy (see core.uc, examples/call_policy_loop.uc).

/* The tool calls of the request's conversation so far, oldest first, with
 * the result each got: [{name, arguments (canonical JSON text), result}].
 * result is null while no tool message answers the call. */
export function tool_history(messages) {
	let out = [], by_id = {};

	for (let m in (messages || [])) {
		if (m.role == 'assistant' && type(m.tool_calls) == 'array') {
			for (let c in m.tool_calls) {
				let f = c.function || {};
				let a = f.arguments;

				if (type(a) == 'string') {
					let o = json(a);

					a = (o != null) ? sprintf('%J', o) : a;
				}
				else {
					a = sprintf('%J', a ?? {});
				}

				let e = {name: f.name, arguments: a, result: null};

				push(out, e);

				if (c.id != null)
					by_id[c.id] = e;
			}
		}
		else if (m.role == 'tool') {
			let e = by_id[m.tool_call_id];
			let t = m.content;

			if (type(t) == 'array')
				t = join('', map(t, (x) => (type(x) == 'object' && x.text != null) ? x.text : ''));

			if (e)
				e.result = (type(t) == 'string') ? t : '';
		}
	}

	return out;
}
