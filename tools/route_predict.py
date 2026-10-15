"""tools/route_predict.py - can a layer's experts be guessed before its attention runs?

Reads route_dump's file (the residual stream after every layer, and the experts each layer's router picked) and the model's
router and post-attention norm weights, and scores two guesses for layer L of token t against the experts really picked:
  router-early: layer L's router applied to layer L's INPUT (the residual before attention), normed with L's post-attention norm
  previous:     the experts layer L picked for token t-1
Recall = the share of the really picked experts that the guess contains.  A guess of g > k experts (prefetch more) is also scored.

    python tools/route_predict.py MODEL.gguf DUMP.bin [--eps 1e-6]   (MODEL: the first file of a split model)
"""
import argparse
import glob
import re
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def read_dump(path):
    raw = Path(path).read_bytes()
    hid, ids = {}, {}
    off, start = 0, 0
    while off < len(raw):
        kind, layer, n_tok, width = np.frombuffer(raw, np.int32, 4, off)
        off += 16
        if kind == 2:
            start = layer
            continue
        dt = np.float32 if kind == 0 else np.int32
        a = np.frombuffer(raw, dt, n_tok * width, off).reshape(n_tok, width)
        off += a.nbytes
        (hid if kind == 0 else ids).setdefault(layer, {})[start] = a
    join = lambda d: {l: np.concatenate([p[s] for s in sorted(p)]) for l, p in d.items()}
    return join(hid), join(ids)


def weights(model):
    files = sorted(glob.glob(re.sub(r"-00001-of-(\d+)\.gguf$", r"-*-of-\1.gguf", model))) or [model]
    w = {}
    for f in files:
        for t in GGUFReader(f).tensors:
            if t.name.endswith(("ffn_gate_inp.weight", "post_attention_norm.weight")):
                w[t.name] = dequantize(t.data, t.tensor_type).reshape([int(x) for x in reversed(t.shape)]).astype(np.float32)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump")
    ap.add_argument("--eps", type=float, default=1e-6)
    a = ap.parse_args()
    hid, ids = read_dump(a.dump)
    w = weights(a.model)
    layers = sorted(l for l in ids if l - 1 in hid and f"blk.{l}.ffn_gate_inp.weight" in w)
    k = ids[layers[0]].shape[1]
    n_tok = ids[layers[0]].shape[0]
    print(f"{n_tok} tokens, {len(layers)} layers scored (1..{layers[-1]}), top-{k}, {w[f'blk.{layers[0]}.ffn_gate_inp.weight'].shape[0]} experts")
    guesses = sorted({k, k + 6, 2 * k})
    rec = {g: [] for g in guesses}
    prev = []
    per_layer = []
    for l in layers:
        x = hid[l - 1]                                       # [t, n_embd], the input of layer l
        xn = x / np.sqrt((x * x).mean(1, keepdims=True) + a.eps) * w[f"blk.{l}.post_attention_norm.weight"]
        logits = xn @ w[f"blk.{l}.ffn_gate_inp.weight"].T    # [t, n_expert]
        order = np.argsort(-logits, 1)
        real = ids[l]
        hit = lambda guess: np.mean([len(set(real[t]) & set(guess[t])) / k for t in range(n_tok)])
        for g in guesses:
            rec[g].append(hit(order[:, :g]))
        prev.append(np.mean([len(set(real[t]) & set(real[t - 1])) / k for t in range(1, n_tok)]))
        per_layer.append((l, rec[k][-1], prev[-1]))
    for g in guesses:
        print(f"router-early, guess {g:2d} experts: {100 * np.mean(rec[g]):5.1f}% of the picked experts")
    print(f"previous token's experts:     {100 * np.mean(prev):5.1f}%")
    worst = sorted(per_layer, key=lambda r: r[1])[:3]
    print("weakest layers (router-early, top-k):", ", ".join(f"{l}: {100 * r:.0f}%" for l, r, _ in worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
