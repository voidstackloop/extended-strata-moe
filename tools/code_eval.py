"""tools/code_eval.py - a small coding check for a model served through tools/serve.py: six Python tasks, each answer run against asserts.

    python tools/code_eval.py MODEL.gguf [--port 8099] [--max-tokens 700] [--suite hard]

Starts serve.py (the adaptive cache), asks each task at temperature 0 through /v1/chat/completions, takes the first
```python block, runs it with the task's asserts (10 s limit) and prints pass/fail, the speed and the total.  Six tasks are a smoke
test of whether a quantization still codes, not a benchmark.
"""
import argparse
import json
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

TASKS = {
    "merge_intervals": ("Write a Python function merge_intervals(intervals) that takes a list of [start, end] lists and returns the merged "
                        "intervals sorted by start (touching intervals such as [1,2],[2,3] merge).",
                        "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
                        "assert merge_intervals([[1,2],[2,3]]) == [[1,3]]\nassert merge_intervals([]) == []\n"
                        "assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]"),
    "lru_cache": ("Write a Python class LRUCache(capacity) with get(key) (returns -1 if missing) and put(key, value), evicting the least "
                  "recently used key when over capacity. Both operations must be O(1).",
                  "c = LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1) == 1; c.put(3,3); assert c.get(2) == -1\n"
                  "c.put(4,4); assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4"),
    "valid_brackets": ("Write a Python function valid_brackets(s) returning True if the brackets ()[]{} in s are balanced and correctly "
                       "nested, ignoring all other characters.",
                       "assert valid_brackets('()[]{}') and valid_brackets('{[()]}') and valid_brackets('a(b)c')\n"
                       "assert not valid_brackets('(]') and not valid_brackets('([)]') and not valid_brackets('((')"),
    "top_k_words": ("Write a Python function top_k_words(words, k) returning the k most frequent words as a list, most frequent first, "
                    "ties broken alphabetically.",
                    "assert top_k_words(['b','a','b','c','a','b'], 2) == ['b','a']\n"
                    "assert top_k_words(['x','y','z'], 2) == ['x','y']\nassert top_k_words([], 3) == []"),
    "roman_to_int": ("Write a Python function roman_to_int(s) converting a Roman numeral (1 to 3999) to an integer.",
                     "assert roman_to_int('III') == 3 and roman_to_int('LVIII') == 58 and roman_to_int('MCMXCIV') == 1994\n"
                     "assert roman_to_int('IV') == 4 and roman_to_int('MMMCMXCIX') == 3999"),
    "parse_kv": ("Write a Python function parse_kv(s) that parses 'a=1;b=two;c=3.5' into a dict, converting values that look like an int "
                 "to int, a float to float, and leaving the rest as str. Empty input gives {}. Ignore empty segments.",
                 "assert parse_kv('a=1;b=two;c=3.5') == {'a': 1, 'b': 'two', 'c': 3.5}\nassert parse_kv('') == {}\n"
                 "assert parse_kv('x=-4;;y=') == {'x': -4, 'y': ''}"),
}


RAISES = "def raises(exc, f, *a):\n    try:\n        f(*a)\n    except exc:\n        return True\n    return False\n"

# --suite hard: ten harder tasks (a parser, a bug fix, edge cases and traps such as exponential matching, recursion depth, boundary times)
HARD = {
    "dijkstra": ("Write a Python function dijkstra(graph, src) where graph maps a node to a list of (neighbor, weight) tuples with "
                 "non-negative weights. Return a dict with the shortest distance from src to every node reachable from it (src itself is 0). "
                 "Nodes that only appear as neighbors have no entry in graph.",
                 "g = {'a': [('b', 1), ('c', 4)], 'b': [('c', 2), ('d', 5)], 'c': [('d', 1)]}\n"
                 "assert dijkstra(g, 'a') == {'a': 0, 'b': 1, 'c': 3, 'd': 4}\nassert dijkstra(g, 'd') == {'d': 0}\n"
                 "assert dijkstra({}, 'x') == {'x': 0}"),
    "calculator": ("Write a Python function evaluate(expr) that evaluates an arithmetic string with + - * / and parentheses, unary minus, "
                   "decimal numbers and spaces, using normal precedence and left associativity, and returns the result. Do not use eval, exec "
                   "or ast. Raise ZeroDivisionError on division by zero and ValueError on malformed input such as '2+', '(1' or ''.",
                   RAISES + "assert evaluate('1+2*3') == 7 and evaluate('(1+2)*3') == 9 and evaluate('-2*-3') == 6\n"
                   "assert evaluate(' 10 / 4 ') == 2.5 and evaluate('2*(3+4)-5/(1+1)') == 11.5 and evaluate('-(2+3)') == -5\n"
                   "assert evaluate('8-3-2') == 3 and evaluate('8/4/2') == 1 and evaluate('1.5+0.5') == 2\n"
                   "assert evaluate('-2+3') == 1 and evaluate('-2-3') == -5 and evaluate('-2*3+4') == -2 and evaluate('2*-3+1') == -5\n"
                   "assert raises(ZeroDivisionError, evaluate, '1/0') and raises(ZeroDivisionError, evaluate, '1/(2-2)')\n"
                   "assert all(raises(ValueError, evaluate, s) for s in ['2+', '(1', '', '1 2', '2 $ 3', '()', '*3'])"),
    "topo_sort": ("Write a Python function topo_sort(deps) where deps maps a task to the list of tasks it depends on (dependencies must come "
                  "first). Return a list of all tasks, including ones that only appear as dependencies, in a valid order; among the tasks that "
                  "are ready at any moment always take the alphabetically smallest. Raise ValueError if there is a cycle.",
                  RAISES + "assert topo_sort({'a': ['b', 'c'], 'b': ['c'], 'c': []}) == ['c', 'b', 'a']\n"
                  "assert topo_sort({'x': [], 'a': [], 'm': ['x']}) == ['a', 'x', 'm']\nassert topo_sort({'a': ['z']}) == ['z', 'a']\n"
                  "assert topo_sort({}) == []\nassert raises(ValueError, topo_sort, {'a': ['b'], 'b': ['a']})\n"
                  "assert raises(ValueError, topo_sort, {'a': ['a']})"),
    "glob_match": ("Write a Python function glob_match(pattern, text) that returns True if the whole text matches the pattern, where '?' "
                   "matches exactly one character and '*' matches any sequence of characters, including an empty one. Do not use re or "
                   "fnmatch. It must stay fast on inputs like pattern 'a*a*a*a*a*a*a*a*b' with text 'a' * 40.",
                   "assert glob_match('a*b', 'aXXb') and not glob_match('a*b', 'aXXc') and not glob_match('?', '')\n"
                   "assert glob_match('*', '') and glob_match('', '') and not glob_match('', 'a') and glob_match('**a', 'ba')\n"
                   "assert glob_match('a?c', 'abc') and not glob_match('a?c', 'ac') and glob_match('*a*', 'bab')\n"
                   "assert not glob_match('a*a*a*a*a*a*a*a*b', 'a' * 40)"),
    "fix_bsearch": ("This function should return the index of the leftmost occurrence of t in the sorted list a, or -1 if t is absent. "
                    "It has bugs. Return the corrected function only.\n```python\ndef bsearch(a, t):\n    lo, hi = 0, len(a)\n"
                    "    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < t:\n            lo = mid\n        else:\n"
                    "            hi = mid - 1\n    return lo if a and a[lo] == t else -1\n```",
                    "assert bsearch([1, 2, 2, 2, 3], 2) == 1 and bsearch([], 1) == -1 and bsearch([1], 1) == 0\n"
                    "assert bsearch([1, 3, 5], 4) == -1 and bsearch([1, 3, 5], 6) == -1 and bsearch([1, 3, 5], 0) == -1\n"
                    "assert bsearch(list(range(1000)), 999) == 999 and bsearch([2, 2, 2, 2], 2) == 0"),
    "text_justify": ("Write a Python function justify(words, width) that packs words greedily into lines of exactly width characters and "
                     "fully justifies them: the spaces between words are spread as evenly as possible, extra spaces going to the leftmost "
                     "gaps. The last line and any line with a single word are left-justified and padded with spaces to width. Return the "
                     "list of lines.",
                     "assert justify(['This', 'is', 'an', 'example', 'of', 'text', 'justification.'], 16) == "
                     "['This    is    an', 'example  of text', 'justification.  ']\n"
                     "assert justify(['What', 'must', 'be', 'acknowledgment', 'shall', 'be'], 16) == "
                     "['What   must   be', 'acknowledgment  ', 'shall be        ']\nassert justify(['a'], 3) == ['a  ']"),
    "rate_limiter": ("Write a Python class RateLimiter(max_calls, period, clock) where clock() returns the current time in seconds. "
                     "allow() returns True and records the call if fewer than max_calls allowed calls happened in the last period seconds, "
                     "otherwise it returns False and records nothing. A call allowed at time s counts until time s + period (exclusive): at "
                     "now == s + period it no longer counts.",
                     "t = [0]; r = RateLimiter(2, 10, lambda: t[0])\nassert r.allow()\nt[0] = 1; assert r.allow()\n"
                     "t[0] = 2; assert not r.allow()\nt[0] = 9.99; assert not r.allow()\nt[0] = 10; assert r.allow()\n"
                     "t[0] = 10.5; assert not r.allow()\nt[0] = 11; assert r.allow()\nt[0] = 11.5; assert not r.allow()"),
    "csv_line": ("Write a Python function parse_csv_line(line) that splits one CSV line into a list of strings: fields are separated by "
                 "commas, a field may be wrapped in double quotes (then it may contain commas), and two double quotes inside a quoted "
                 "field mean one literal quote. An empty line gives [''], and a trailing comma gives a final empty field. Do not use the "
                 "csv module.",
                 "assert parse_csv_line('a,b,c') == ['a', 'b', 'c'] and parse_csv_line('\"a,b\",c') == ['a,b', 'c']\n"
                 "assert parse_csv_line('a,\"he said \"\"hi\"\"\",c') == ['a', 'he said \"hi\"', 'c']\n"
                 "assert parse_csv_line('') == [''] and parse_csv_line('a,,b') == ['a', '', 'b'] and parse_csv_line('a,') == ['a', '']\n"
                 "assert parse_csv_line('\"\",x') == ['', 'x'] and parse_csv_line('\"\"\"\"') == ['\"']"),
    "num_islands": ("Write a Python function num_islands(grid) that counts the islands in a grid given as a list of strings of '1' (land) "
                    "and '0' (water); land cells join islands only horizontally and vertically. Grids can be large (up to 500 x 500).",
                    "assert num_islands(['11000', '11000', '00100', '00011']) == 3 and num_islands([]) == 0\n"
                    "assert num_islands(['00000']) == 0 and num_islands(['1' * 300] * 300) == 1\n"
                    "assert num_islands(['101', '010', '101']) == 5"),
    "retry": ("Write a Python decorator factory retry(times, exceptions) where times is the total number of attempts and exceptions is a "
              "tuple of exception types to retry on. If every attempt fails, re-raise the last exception; any other exception propagates "
              "at once. The decorated function must keep its name and docstring and pass through positional and keyword arguments.",
              RAISES + "calls = []\n@retry(3, (ValueError,))\ndef f():\n    \"\"\"doc\"\"\"\n    calls.append(1)\n"
              "    if len(calls) < 3:\n        raise ValueError('x')\n    return 'ok'\n"
              "assert f() == 'ok' and len(calls) == 3 and f.__name__ == 'f' and f.__doc__ == 'doc'\n"
              "c2 = []\n@retry(2, (KeyError,))\ndef g():\n    c2.append(1)\n    raise KeyError('k')\n"
              "assert raises(KeyError, g) and len(c2) == 2\n"
              "c3 = []\n@retry(5, (KeyError,))\ndef h():\n    c3.append(1)\n    raise TypeError('t')\n"
              "assert raises(TypeError, h) and len(c3) == 1\n"
              "@retry(2, (ValueError,))\ndef add(a, b=0):\n    return a + b\nassert add(1, b=2) == 3"),
}


def ask(base, prompt, max_tokens):
    body = {"model": "m", "messages": [{"role": "system", "content": "You are a careful programmer."},
                                       {"role": "user", "content": prompt + " Reply with one ```python code block and nothing else."}],
            "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"], time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--suite", choices=["smoke", "hard"], default="smoke", help="smoke: six easy tasks; hard: ten harder ones")
    ap.add_argument("--out", default="build/code_eval.json")
    ap.add_argument("--no-server", action="store_true", help="a server is already running on --port: use it, do not start one")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    srv = None if a.no_server else subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).parent / "serve.py"), a.model, "--port", str(a.port),
         "--ctx", "8192", "--alias", "m"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    results = {}
    try:
        for _ in range(1800):
            try:
                urllib.request.urlopen(base + "/health", timeout=1)
                break
            except OSError:
                time.sleep(1)
        else:
            raise SystemExit("the server did not come up")
        for name, (prompt, asserts) in (HARD if a.suite == "hard" else TASKS).items():
            text, n, secs = ask(base, prompt, a.max_tokens)
            m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
            code = m.group(1) if m else text
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code + "\n\n" + asserts + "\n")
            try:
                r = subprocess.run([sys.executable, "-I", f.name], capture_output=True, text=True, timeout=10)
                ok, err = r.returncode == 0, r.stderr.strip().splitlines()[-1:]
            except subprocess.TimeoutExpired:
                ok, err = False, ["timeout"]
            results[name] = {"pass": ok, "tokens": n, "tok_per_s": round(n / secs, 2), "error": err, "answer": text}
            print(f"{name:16s} {'PASS' if ok else 'FAIL'}  {n} tokens, {n / secs:.1f} tokens/s  {err if not ok else ''}", flush=True)
    finally:
        if srv is not None:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"{sum(r['pass'] for r in results.values())} of {len(results)} tasks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
