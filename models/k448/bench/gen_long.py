"""Long-prompt test: python gen_long.py PORT FILE [chars] [max_tokens]. Sends the first N characters of FILE plus a question, twice
(the second time the server's prompt cache should make the prompt almost free) and prints the server's own timings."""
import json, sys, urllib.request

port, path = sys.argv[1], sys.argv[2]
chars = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
n = int(sys.argv[4]) if len(sys.argv) > 4 else 200
text = open(path, encoding="utf-8", errors="replace").read()[:chars]
msgs = [{"role": "user", "content": "Here is a source file:\n\n" + text + "\n\nExplain in 5 bullet points what this file does and name its main functions."}]
for i in range(2):
    body = {"messages": msgs, "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = json.load(urllib.request.urlopen(req, timeout=1800))["timings"]
    print(f"pass {i + 1}: prompt {t['prompt_n']} tokens at {t['prompt_per_second']:.1f} tok/s ({t['prompt_ms'] / 1000:.1f} s), "
          f"decode {t['predicted_n']} tokens at {t['predicted_per_second']:.1f} tok/s, draft {t.get('draft_n_accepted')}/{t.get('draft_n')}", flush=True)
