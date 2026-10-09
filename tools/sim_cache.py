"""tools/sim_cache.py - expert-cache policies replayed on a real routing trace, at the VRAM llama.cpp would give them.

The groups and their slots come from tools/moe_plan.py (llama.cpp's own arithmetic) for --cache-mib.  The profile is
ranked on the TRAIN trace and every policy is scored on the TEST trace (held out), token by token in order, as decoding
would see them (one token, its k experts per layer):

  layer split   --n-cpu-moe: the experts of the last layers that fit are on the GPU, the rest computed on the CPU
  lru cold      llama.cpp's --moe-cache-mib: empty at start; every miss uploaded, the least recently used evicted
  lru warm      the same, filled from the profile at load (--moe-cache-profile)
  static        Strata's tier: the profile's hottest stay, nothing is uploaded, misses computed on the CPU

Hit rate counts distinct experts per (token, layer), as llama.cpp's own counter does.  Bytes are per token: PCIe
uploads for the LRU policies, CPU expert reads for the others.

    python tools/sim_cache.py MODEL.gguf --cache-mib N --train a.bin [...] --test b.bin [...]
"""
import argparse
import struct
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import moe_plan as mp


def read_tokens(paths):
    """Token-major routing: [{layer: ids}] in sequence order.  The tracer writes each ubatch layer by layer, all of the
    ubatch's tokens per layer, so a run of records of one layer is one (ubatch, layer) block."""
    tokens = []
    for p in paths:
        blob = Path(p).read_bytes()
        off, blocks = 0, []                              # [(layer, [ids per token])]
        while off + 8 <= len(blob):
            layer, k = struct.unpack_from("<ii", blob, off)
            ids = struct.unpack_from(f"<{k}i", blob, off + 8)
            off += 8 + 8 * k
            if blocks and blocks[-1][0] == layer:
                blocks[-1][1].append(ids)
            else:
                blocks.append((layer, [ids]))
        base = len(tokens)
        seen_first = None
        for layer, rows in blocks:
            if seen_first is None or layer == seen_first:       # a new ubatch starts at the first layer again
                if seen_first is not None:
                    base = len(tokens)
                seen_first = layer
                tokens.extend({} for _ in rows)
            for j, ids in enumerate(rows):
                tokens[base + j][layer] = ids
    return tokens


def frequencies(tokens):
    f = defaultdict(int)
    for t in tokens:
        for l, ids in t.items():
            for e in ids:
                f[(l, e)] += 1
    return f


class Lru:
    """llama.cpp's moe_cache_lru for one group: hits are touched first, then each miss takes the least recently used slot."""

    def __init__(self, n_slots, warm=()):
        self.n, self.od = n_slots, OrderedDict()
        for p in reversed(warm[:n_slots]):               # the hottest end most recent
            self.od[p] = True

    def access(self, layer, ids):
        uniq = sorted(set(ids))
        hits = [e for e in uniq if (layer, e) in self.od]
        for e in hits:
            self.od.move_to_end((layer, e))
        for e in uniq:
            if (layer, e) in self.od:
                continue
            if len(self.od) >= self.n:
                self.od.popitem(last=False)
            self.od[(layer, e)] = True
        return len(hits), len(uniq) - len(hits)


def simulate(info, train, test, pin_reserve=0.0):
    group_of = info.group_of()
    ranked = mp.rank_pairs(info, frequencies(train))
    hot = set(mp.hot_set(info, ranked))
    per_group_hot = defaultdict(list)
    for p in ranked:
        g = group_of[p[0]]
        if len(per_group_hot[id(g)]) < g.n_slots:
            per_group_hot[id(g)].append(p)
    expert_layers = sorted(group_of)
    n_slots = sum(g.n_slots for g in info.groups)
    gpu_layers = set(expert_layers[len(expert_layers) - n_slots // info.n_expert:]) if n_slots >= info.n_expert else set()

    # pinned: the profile's hottest stay for good, the last `reserve` slots of each group are an LRU for the misses
    reserve = {id(g): min(g.n_slots, max(info.n_used, int(g.n_slots * pin_reserve))) for g in info.groups}
    pinned = {p for g in info.groups for p in per_group_hot[id(g)][:g.n_slots - reserve[id(g)]]}

    res = {}
    for name in ("layer split", "lru cold", "lru warm", "pinned", "static"):
        size = lambda g: reserve[id(g)] if name == "pinned" else g.n_slots
        lrus = {id(g): Lru(size(g), per_group_hot[id(g)] if name == "lru warm" else ()) for g in info.groups if g.n_slots}
        hits = misses = 0
        miss_bytes = 0
        for t in test:
            for l, ids in sorted(t.items()):
                g = group_of.get(l)
                if g is None:
                    continue
                uniq = set(ids)
                if name == "layer split":
                    h = len(uniq) if l in gpu_layers else 0
                elif name == "static":
                    h = sum((l, e) in hot for e in uniq)
                elif g.n_slots == 0:
                    h = 0
                elif name == "pinned":
                    rest = [e for e in uniq if (l, e) not in pinned]
                    h = len(uniq) - len(rest) + (lrus[id(g)].access(l, rest)[0] if rest else 0)
                else:
                    h, _ = lrus[id(g)].access(l, ids)
                hits += h
                misses += len(uniq) - h
                miss_bytes += (len(uniq) - h) * g.expert_bytes
        n = hits + misses
        res[name] = (hits / n if n else 0.0, miss_bytes / max(len(test), 1))
    return res, n_slots, len(gpu_layers)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", nargs="+")
    ap.add_argument("--cache-mib", type=int, nargs="+", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--pin-reserve", type=float, default=0.0,
                    help="pinned policy: share of each group's slots left to the LRU (at least top-k; default 0)")
    a = ap.parse_args(argv)
    info = mp.inspect(a.gguf)
    train, test = read_tokens(a.train), read_tokens(a.test)
    print(f"{info.arch}: {info.n_expert} experts top-{info.n_used}; train {len(train)} tokens, test {len(test)} tokens (held out)")
    print("| cache MiB | slots | layer split (--n-cpu-moe) | lru cold (upstream) | lru warm (--moe-cache-profile) "
          "| pinned (--moe-cache-pin) | static (Strata) |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for mib in a.cache_mib:
        if mp.size_cache(info, mib * mp.MIB) == 0:
            print(f"| {mib} | 0 | too small for llama.cpp | | | | |")
            continue
        res, n_slots, n_layers = simulate(info, train, test, a.pin_reserve)
        cell = lambda k, what: f"{100 * res[k][0]:.1f}% hit, {res[k][1] / mp.MIB:.0f} MiB/tok {what}"
        print(f"| {mib} | {n_slots} | {cell('layer split', 'CPU')} ({n_layers} layers) | {cell('lru cold', 'PCIe')} | "
              f"{cell('lru warm', 'PCIe')} | {cell('pinned', 'PCIe')} | {cell('static', 'CPU')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
