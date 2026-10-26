"""Decode speed against a running llama-server: python gen.py PORT [runs] [max_tokens]. Prints tokens/s from the server's own timings."""
import json, sys, urllib.request

port = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
prompts = ["Write a Python function that merges overlapping intervals, with a docstring and three test cases.",
           "Implement an LRU cache class in Python with get and put in O(1), and explain the design briefly.",
           "Write a Python script that parses a CSV file and prints the average of each numeric column."]
for i in range(runs):
    body = {"messages": [{"role": "user", "content": prompts[i % len(prompts)]}], "max_tokens": n, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    t = r["timings"]
    print(f"run {i + 1}: {t['predicted_n']} tokens, decode {t['predicted_per_second']:.1f} tok/s, prompt {t['prompt_n']} tokens at {t['prompt_per_second']:.1f} tok/s"
          + (f", draft accepted {t['draft_n_accepted']}/{t['draft_n']}" if t.get("draft_n") else ""), flush=True)
