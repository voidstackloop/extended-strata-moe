"""tools/moe_plan.py - which experts of ANY MoE GGUF the llama.cpp MoE cache holds, and the flags that set it up.

Strata's expert tier (include/strata/core/expert_cache.hpp in Niko1221/Strata) on llama.cpp's MoE cache
(src/llama-moe-cache.cpp): the experts stay in RAM (--cpu-moe), the cache holds the hottest in VRAM, filled at load
from a profile (--moe-cache-profile), and with --moe-cache-hybrid the CPU computes the others.  This tool mirrors the
cache's own arithmetic - which tensors it holds, how it groups layers and splits the budget, how many slots a group
gets - so the slots and the hit rate it reports are the ones llama.cpp will have.  docs/ADR-001, docs/ADR-002.

    python tools/moe_plan.py MODEL.gguf [MORE-SHARDS.gguf ...] (--vram-gib 12 [--reserve-gib 2] | --cache-mib N)
                             [--trace routing.bin ...] [--out profile.strp]

The other shards of a split GGUF (`*-00001-of-00003.gguf`) are found from the first.  A trace is Strata's
`--dump-routing` format (tools/routing_trace writes it): records of int32 layer, int32 k, k int32 ids, k float32
weights.  Without a trace the layers share the cache evenly (round-robin), as Strata's fallback does.
"""
import argparse
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from gguf import GGUFReader

# common/common.h LLM_FFN_EXPS_REGEX: what --cpu-moe / --n-cpu-moe keep in RAM (weights, biases, scales)
HOST_RE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")
# llama_moe_cache_layer_experts: the tensors the cache holds, in its order
BANK_KINDS = ("gate_up", "gate", "up", "down")
BANK_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")
ALIGN = 128            # ggml CUDA buffer alignment; the cache pads each bank to it
MAGIC, VERSION = b"STRP", 1
MIB, GIB = 1 << 20, 1 << 30


@dataclass
class Group:
    """Layers whose bank tensors have the same types and shapes: they share one budget and one LRU in llama.cpp."""
    key: tuple
    expert_sizes: list                               # nb[2] of each bank tensor: bytes of one expert
    n_expert: int
    layers: list = field(default_factory=list)
    n_slots: int = 0

    @property
    def expert_bytes(self):
        return sum(self.expert_sizes)

    @property
    def host_bytes(self):
        return self.expert_bytes * self.n_expert * len(self.layers)

    def alloc(self, n_slots):                        # one extra slot per bank: CUDA MMQ can read past the last expert
        return sum(-(-s * (n_slots + 1) // ALIGN) * ALIGN for s in self.expert_sizes)


@dataclass
class MoeInfo:
    arch: str
    n_layers: int
    n_expert: int
    n_used: int
    dense_bytes: int = 0                             # stays in VRAM with --cpu-moe
    host_bytes: int = 0                              # everything --cpu-moe keeps in RAM
    groups: list = field(default_factory=list)

    def group_of(self):
        return {l: g for g in self.groups for l in g.layers}


def _field(r, key, default=None):
    f = r.get_field(key)
    if f is None:
        return default
    v = f.contents()
    return max(v) if isinstance(v, list) else v      # per-layer arrays: the largest


def shards(paths):
    """All files of a split GGUF from its first shard; other lists are returned as given."""
    if len(paths) != 1:
        return paths
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", str(paths[0]))
    if not m or int(m.group(3)) == 1:
        return paths
    found = [Path(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf") for i in range(1, int(m.group(3)) + 1)]
    missing = [str(p) for p in found if not p.exists()]
    if missing:
        raise SystemExit(f"split GGUF: missing {', '.join(missing)}")
    return found


def inspect(paths):
    readers = [GGUFReader(p) for p in shards(paths)]
    r0 = readers[0]
    arch = _field(r0, "general.architecture")
    n_expert = _field(r0, f"{arch}.expert_count", 0)
    if not n_expert:
        raise SystemExit(f"{paths[0]}: {arch} has no {arch}.expert_count - not an MoE model")
    info = MoeInfo(arch, _field(r0, f"{arch}.block_count"), n_expert, _field(r0, f"{arch}.expert_used_count", 0))
    banks = defaultdict(dict)                        # layer -> kind -> (type, shape, expert bytes)
    names = {t.name for r in readers for t in r.tensors}
    for r in readers:
        for t in r.tensors:
            if t.name == "token_embd.weight" and "output.weight" in names:
                continue                             # the input layer is always on the CPU (tied output: it's on the GPU too)
            if not HOST_RE.search(t.name):
                info.dense_bytes += int(t.n_bytes)
                continue
            info.host_bytes += int(t.n_bytes)
            m = BANK_RE.match(t.name)
            if not m:
                continue                             # biases, scales, chexps: in RAM, not in the cache
            ne = int(t.shape[-1])                    # ne order: the expert axis is the last one
            if int(t.n_bytes) % ne:
                raise SystemExit(f"{t.name}: {int(t.n_bytes)} bytes do not split into {ne} experts")
            banks[int(m.group(1))][m.group(2)] = (t.tensor_type.name, tuple(int(x) for x in t.shape), int(t.n_bytes) // ne)
    by_key = {}
    for layer in sorted(banks):
        kinds = [k for k in BANK_KINDS if k in banks[layer]]
        key = tuple((k,) + banks[layer][k][:2] for k in kinds)
        g = by_key.get(key)
        if g is None:
            g = by_key[key] = Group(key, [banks[layer][k][2] for k in kinds], banks[layer][kinds[0]][1][-1])
        g.layers.append(layer)
    info.groups = list(by_key.values())
    return info


def size_cache(info, cache_bytes):
    """llama_moe_cache's split: each group gets the budget in proportion to its host experts, then as many slots as
    fit; a group with fewer slots than experts per token is not cached."""
    total = sum(g.host_bytes for g in info.groups)
    used = 0
    for g in info.groups:
        budget = int(float(cache_bytes) * float(g.host_bytes) / float(total))   # as llama.cpp: in double, truncated
        cap = len(g.layers) * g.n_expert
        lo, hi = 0, cap                              # largest n with alloc(n) <= budget (alloc is monotonic)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if g.alloc(mid) <= budget:
                lo = mid
            else:
                hi = mid - 1
        g.n_slots = lo if lo >= info.n_used else 0
        used += g.alloc(g.n_slots) if g.n_slots else 0
    return used


def min_cache_mib(info):
    """The smallest --moe-cache-mib at which some group holds a token's experts: below it llama.cpp refuses the cache
    ("too small to hold the experts of one token") and the context is not created."""
    lo, hi = 1, 1
    while size_cache(info, hi * MIB) == 0:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if size_cache(info, mid * MIB):
            hi = mid
        else:
            lo = mid + 1
    return lo


def thrashes(info):
    """The cached groups whose slots are fewer than the experts one token selects across the group's layers: the LRU
    then evicts every expert before its layer comes round again (Granite, 64-128 MiB: 0% hits, measured)."""
    return [g for g in info.groups if g.n_slots and g.n_slots < len(g.layers) * info.n_used]


def thrash_free_mib(info):
    """The smallest cache in which no group thrashes."""
    lo, hi = 1, 1
    while not (size_cache(info, hi * MIB) and not thrashes(info)):
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if size_cache(info, mid * MIB) and not thrashes(info):
            hi = mid
        else:
            lo = mid + 1
    return lo


def layer_split(info, budget_bytes):
    """--n-cpu-moe N for the same VRAM: llama.cpp keeps the experts of the first N layers in RAM, so the GPU gets the
    last layers whose experts fit.  (N, layers on the GPU)."""
    host = {l: g.expert_bytes * g.n_expert for g in info.groups for l in g.layers}
    used, n_gpu, first_gpu = 0, 0, info.n_layers
    for l in sorted(host, reverse=True):
        if used + host[l] > budget_bytes:
            break
        used += host[l]
        n_gpu += 1
        first_gpu = l
    return first_gpu, n_gpu


def read_traces(paths, n_layers, n_expert):
    freq = defaultdict(int)
    for p in paths:
        blob = Path(p).read_bytes()
        off = 0
        while off + 8 <= len(blob):
            layer, k = struct.unpack_from("<ii", blob, off)
            off += 8
            for e in struct.unpack_from(f"<{k}i", blob, off):
                if 0 <= layer < n_layers and 0 <= e < n_expert:
                    freq[(layer, e)] += 1
            off += 8 * k
    return freq


def rank_pairs(info, freq, whole_layers=0):
    """Every (layer, expert) pair of the cached layers: by frequency, then round-robin across the layers.
    whole_layers: all experts of the last N layers first, so that the hybrid holds them whole and they need no CPU
    (each layer that does costs a GPU-CPU round trip, the same as a --n-cpu-moe layer)."""
    layers = sorted(l for g in info.groups for l in g.layers)
    rr = [(l, e) for e in range(info.n_expert) for l in layers]
    order = {p: i for i, p in enumerate(rr)}
    # one layer at a time, the last first (the layers --n-cpu-moe puts on the GPU): ranked together by frequency, a group
    # without room for all of them filled every one partly and held few whole (13 asked, 6 held on Granite at 384 MiB)
    whole = layers[len(layers) - whole_layers:][::-1] if whole_layers > 0 else []
    rank_of = {l: i for i, l in enumerate(whole)}
    return sorted(rr, key=lambda p: (rank_of.get(p[0], len(whole)), -freq.get(p, 0), order[p]))


def hot_set(info, ranked):
    """What --moe-cache-profile puts in the cache: each group takes its own layers' pairs in rank order."""
    group_of, taken, hot = info.group_of(), defaultdict(int), []
    for p in ranked:
        g = group_of.get(p[0])
        if g is not None and taken[id(g)] < g.n_slots:
            taken[id(g)] += 1
            hot.append(p)
    return hot


def write_profile(path, n_layers, n_expert, ranked, slots):
    """Strata's write_expert_profile layout: STRP, (1, n_layers, n_expert, slots, n_ranked) u32, the (u16, u16)
    pairs, then an n_layers x n_expert int32 table of each pair's rank (-1 unranked).  Atomic via .tmp."""
    rank = [-1] * (n_layers * n_expert)
    for i, (l, e) in enumerate(ranked):
        rank[l * n_expert + e] = i
    blob = MAGIC + struct.pack("<5I", VERSION, n_layers, n_expert, slots, len(ranked))
    blob += b"".join(struct.pack("<HH", l, e) for l, e in ranked)
    blob += struct.pack(f"<{len(rank)}i", *rank)
    tmp = Path(str(path) + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)


def read_profile(path):
    blob = Path(path).read_bytes()
    if blob[:4] != MAGIC:
        raise ValueError(f"{path}: not a Strata profile")
    ver, nl, ne, slots, n = struct.unpack_from("<5I", blob, 4)
    if ver != VERSION:
        raise ValueError(f"{path}: profile version {ver}, expected {VERSION}")
    return nl, ne, slots, [struct.unpack_from("<HH", blob, 24 + 4 * i) for i in range(n)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf", nargs="+")
    size = ap.add_mutually_exclusive_group(required=True)
    size.add_argument("--vram-gib", type=float, help="the GPU's memory; the cache gets what the dense weights and --reserve-gib leave")
    size.add_argument("--cache-mib", type=int, help="size the cache exactly (as --moe-cache-mib)")
    ap.add_argument("--reserve-gib", type=float, default=2.0, help="KV cache, compute buffers, driver (default 2)")
    ap.add_argument("--trace", nargs="*", default=[])
    ap.add_argument("--whole-layers", type=int, default=0,
                    help="hybrid: rank every expert of the last N layers first, so they run on the GPU with no CPU round trip")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    info = inspect(a.gguf)
    if not info.groups:
        raise SystemExit(f"{info.arch}: no ffn_*_exps.weight tensors - nothing for the MoE cache to hold")
    n_cached_layers = sum(len(g.layers) for g in info.groups)
    print(f"{info.arch}: {info.n_layers} layers, {n_cached_layers} with experts, {info.n_expert} experts, top-{info.n_used}")
    print(f"VRAM (dense, with --cpu-moe): {info.dense_bytes / GIB:.2f} GiB   RAM (experts): {info.host_bytes / GIB:.2f} GiB")
    if a.cache_mib is not None:
        cache_mib = a.cache_mib
    else:
        cache_mib = int(((a.vram_gib - a.reserve_gib) * GIB - info.dense_bytes) // MIB)
        if cache_mib <= 0:
            print(f"the dense weights alone ({info.dense_bytes / GIB:.2f} GiB) leave no room in "
                  f"{a.vram_gib - a.reserve_gib:.2f} GiB for an expert cache")
            return 1
    used = size_cache(info, cache_mib * MIB)
    if used == 0:
        need = min_cache_mib(info)
        print(f"a {cache_mib} MiB cache holds no layer's {info.n_used} experts per token: llama.cpp refuses it. "
              f"The smallest that works is {need} MiB")
        return 1
    freq = read_traces(a.trace, info.n_layers, info.n_expert) if a.trace else {}
    ranked = rank_pairs(info, freq, a.whole_layers)
    hot = hot_set(info, ranked)
    for g in info.groups:
        types = "/".join(f"{k[0]}:{k[1]}" for k in g.key)
        n_pairs = len(g.layers) * g.n_expert
        state = f"{g.n_slots:6d} slots ({100 * g.n_slots / n_pairs:5.1f}%)" if g.n_slots else "  not cached (fewer slots than experts per token)"
        print(f"  {len(g.layers):3d} layers {types}: {g.expert_bytes / MIB:.2f} MiB/expert, {state}")
    n_pairs = n_cached_layers * info.n_expert
    hot_set_ = set(hot)
    whole = sum(all((l, e) in hot_set_ for e in range(info.n_expert)) for l in info.group_of())
    if whole:   # llama.cpp logs the same count at load ("N layers held whole")
        print(f"  {whole} layers held whole: with --moe-cache-hybrid they run on the GPU with no CPU round trip")
    total = sum(freq.values())
    hit = f", predicted hit rate {100 * sum(freq.get(p, 0) for p in hot) / total:.1f}% (on the trace)" if total else ", no trace: no hit rate"
    print(f"cache: {len(hot)} of {n_pairs} experts ({100 * len(hot) / n_pairs:.1f}%), {used / MIB:.0f} of {cache_mib} MiB{hit}")
    if a.out:
        write_profile(a.out, info.n_layers, info.n_expert, ranked, len(hot))
        print(f"wrote {a.out}")
    # -fit off: llama.cpp otherwise re-fits unset parameters to free memory, around the cache it does not know about
    flags = f"-ngl 999 -fit off --cpu-moe --moe-cache-mib {cache_mib}"
    if a.out:
        print(f"llama.cpp, hybrid (the CPU computes the experts the cache does not hold):\n"
              f"  {flags} --moe-cache-profile {a.out} --moe-cache-hybrid")
    else:
        print(f"llama.cpp: {flags}   (the hybrid also needs a profile: give --out)")
    n_cpu, n_gpu = layer_split(info, cache_mib * MIB)
    print(f"layer split in the same VRAM (the experts of the last {n_gpu} layers):\n  -ngl 999 -fit off --n-cpu-moe {n_cpu}")
    bad = thrashes(info)
    if bad:   # the hybrid's set is static and never thrashes; the LRU of upload mode does below this size
        g = bad[0]
        need = thrash_free_mib(info)
        size_cache(info, cache_mib * MIB)
        print(f"note: without --moe-cache-hybrid every miss is uploaded, and {g.n_slots} slots for {len(g.layers)} layers "
              f"x top-{info.n_used} = {len(g.layers) * info.n_used} experts per token make its LRU evict each expert "
              f"before its layer comes round again: upload mode needs at least {need} MiB")
    print("Measure them on your machine, at a few thread counts: tools/bench_cache.py --hybrid -t 4 6 8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
