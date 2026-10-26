"""Builds k448-chat-template.jinja from the model's own template (k448-chat-template.orig.jinja): a system message that is not the first message
(Claude Code sends one after the user turn) is rendered in place as a user turn instead of raising 'System message must be at the beginning',
so no rewriting proxy is needed and the text stays at the same position from request to request (the prompt cache keeps matching)."""
from pathlib import Path

here = Path(__file__).resolve().parent
t = (here / "k448-chat-template.orig.jinja").read_text(encoding="utf-8")
old = "        {%- if not loop.first %}\n            {{- raise_exception('System message must be at the beginning.') }}\n        {%- endif %}"
new = "        {%- if not loop.first %}\n            {{- '<|im_start|>user\\n' + content + '<|im_end|>' + '\\n' }}\n        {%- endif %}"
assert old in t
(here / "k448-chat-template.jinja").write_text(t.replace(old, new, 1), encoding="utf-8", newline="\n")
print("k448-chat-template.jinja written")
