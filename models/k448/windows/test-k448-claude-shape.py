"""Sends the request shape Claude Code uses (a system message AFTER the user turn) to the k448 server's Anthropic endpoint and prints the answer.
It returned HTTP 500 with the model's stock chat template; with k448-chat-template.jinja it must answer."""
import json
import urllib.request

body = {"model": "k448", "max_tokens": 40, "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Reply with the single word PONG."}]},
    {"role": "system", "content": "Reminder: answer as briefly as possible."}]}
req = urllib.request.Request("http://127.0.0.1:8090/v1/messages", json.dumps(body).encode(), {"Content-Type": "application/json", "x-api-key": "local", "anthropic-version": "2023-06-01"})
r = json.load(urllib.request.urlopen(req, timeout=300))
print("status ok:", r["content"][0]["text"].strip()[:60], "| stop:", r.get("stop_reason"))
