"""tools/test_moe_plan.py - the planner on synthetic MoE GGUFs.  No GPU, no download: python tools/test_moe_plan.py"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

sys.path.insert(0, str(Path(__file__).parent))
import moe_plan as mp

N_LAYERS, N_EXPERT, EMBD, FF = 3, 8, 16, 4


def add_kv(w):
    w.add_block_count(N_LAYERS)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(2)


def add_layer(w, l):
    dt = np.float32 if l == 1 else np.float16                                   # mixed types: two cache groups
    w.add_tensor(f"blk.{l}.ffn_gate_inp.weight", np.zeros((N_EXPERT, EMBD), np.float32))
    for n in ("gate", "up"):
        w.add_tensor(f"blk.{l}.ffn_{n}_exps.weight", np.zeros((N_EXPERT, FF, EMBD), dt))
    w.add_tensor(f"blk.{l}.ffn_down_exps.weight", np.zeros((N_EXPERT, EMBD, FF), dt))
    w.add_tensor(f"blk.{l}.ffn_up_exps.bias", np.zeros((N_EXPERT, FF), np.float32))   # RAM, not the cache


def write(w):
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def make_gguf(path):
    w = GGUFWriter(str(path), "qwen3moe")
    add_kv(w)
    w.add_tensor("token_embd.weight", np.zeros((32, EMBD), np.float32))         # RAM: llama.cpp's input layer
    w.add_tensor("output.weight", np.zeros((32, EMBD), np.float32))
    w.add_tensor("blk.0.ffn_up.weight", np.zeros((FF, EMBD), np.float32))      # layer 0: dense FFN, no experts
    add_layer(w, 1)
    add_layer(w, 2)
    write(w)


def make_split(d):
    """The same model as two shards, named as llama.cpp's gguf-split names them."""
    w = GGUFWriter(str(d / "s-00001-of-00002.gguf"), "qwen3moe")
    add_kv(w)
    w.add_tensor("token_embd.weight", np.zeros((32, EMBD), np.float32))
    w.add_tensor("output.weight", np.zeros((32, EMBD), np.float32))
    w.add_tensor("blk.0.ffn_up.weight", np.zeros((FF, EMBD), np.float32))
    add_layer(w, 1)
    write(w)
    w = GGUFWriter(str(d / "s-00002-of-00002.gguf"), "qwen3moe")
    add_layer(w, 2)
    write(w)


def make_trace(path, records):
    blob = b""
    for layer, ids in records:
        blob += struct.pack("<ii", layer, len(ids)) + struct.pack(f"<{len(ids)}i", *ids)
        blob += struct.pack(f"<{len(ids)}f", *([0.5] * len(ids)))
    path.write_bytes(blob)


def check_info(info):
    assert (info.arch, info.n_layers, info.n_expert, info.n_used) == ("qwen3moe", 3, 8, 2), info
    gate_inp = N_EXPERT * EMBD * 4
    assert info.dense_bytes == 32 * EMBD * 4 + FF * EMBD * 4 + 2 * gate_inp, info.dense_bytes   # output, ffn_up, routers
    bias = N_EXPERT * FF * 4
    assert info.host_bytes == N_EXPERT * 3 * FF * EMBD * (4 + 2) + 2 * bias, info.host_bytes
    assert [g.layers for g in info.groups] == [[1], [2]], [g.layers for g in info.groups]
    assert [g.expert_sizes for g in info.groups] == [[FF * EMBD * 4] * 3, [FF * EMBD * 2] * 3]


def main():
    d = Path(tempfile.mkdtemp())
    make_gguf(d / "m.gguf")
    info = mp.inspect([d / "m.gguf"])
    check_info(info)
    g1, g2 = info.groups

    # llama_moe_cache's arithmetic: banks padded to 128 B with one spare slot; budget split by host bytes
    assert g1.alloc(3) == 3 * 256 * 4 and g2.alloc(3) == 3 * 128 * 4
    assert mp.size_cache(info, 4608) == 4608 and (g1.n_slots, g2.n_slots) == (3, 3)
    mp.size_cache(info, 4607)                       # one byte short: budgets 3071 / 1535 B hold 2 slots each
    assert (g1.n_slots, g2.n_slots) == (2, 2), (g1.n_slots, g2.n_slots)
    assert mp.size_cache(info, 2304) == 0 and (g1.n_slots, g2.n_slots) == (0, 0)   # 1 slot each < top-2: uncached
    mp.size_cache(info, 1 << 30)
    assert (g1.n_slots, g2.n_slots) == (8, 8)       # capped at every expert
    assert mp.min_cache_mib(info) == 1              # these experts are bytes: 1 MiB holds them all

    # --n-cpu-moe for the same VRAM: whole layers from the last one down (layer 2: 3072 B of experts, layer 1: 6144 B)
    assert mp.layer_split(info, 3072) == (2, 1) and mp.layer_split(info, 3071) == (N_LAYERS, 0)
    assert mp.layer_split(info, 9216) == (1, 2)
    mp.size_cache(info, 4608)
    assert mp.thrashes(info) == []                  # 3 slots >= 1 layer x top-2 in each group

    # no trace: round-robin; each group takes its own layers' pairs in rank order
    mp.size_cache(info, 4608)
    ranked = mp.rank_pairs(info, {})
    assert ranked[:4] == [(1, 0), (2, 0), (1, 1), (2, 1)], ranked[:4]
    assert mp.hot_set(info, ranked) == [(1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (2, 2)]

    # a trace decides the order within each group
    make_trace(d / "t.bin", [(1, [5, 2])] * 6 + [(2, [3, 1])] * 3 + [(2, [3, 0]), (1, [5, 7])])
    freq = mp.read_traces([d / "t.bin"], N_LAYERS, N_EXPERT)
    ranked = mp.rank_pairs(info, freq)
    hot = mp.hot_set(info, ranked)
    assert hot == [(1, 5), (1, 2), (2, 3), (2, 1), (2, 0), (1, 7)], hot   # ties (1 each): round-robin order
    assert len(ranked) == 2 * N_EXPERT and len(set(ranked)) == len(ranked)

    # whole layers first: every expert of the last layer, hottest first, then the rest by frequency
    whole = mp.rank_pairs(info, freq, whole_layers=1)
    assert whole[:N_EXPERT] == [(2, 3), (2, 1), (2, 0)] + [(2, e) for e in (2, 4, 5, 6, 7)], whole[:N_EXPERT]
    assert whole[N_EXPERT:N_EXPERT + 2] == [(1, 5), (1, 2)] and sorted(whole) == sorted(ranked)
    # two whole layers: all of the last one, then all of the one before, never interleaved
    two = mp.rank_pairs(info, freq, whole_layers=2)
    assert [l for l, _ in two] == [2] * N_EXPERT + [1] * N_EXPERT, two
    assert two[N_EXPERT:N_EXPERT + 3] == [(1, 5), (1, 2), (1, 7)], two[N_EXPERT:]

    # profile round trip, Strata's layout, and its version check
    mp.write_profile(d / "p.strp", N_LAYERS, N_EXPERT, ranked, len(hot))
    assert mp.read_profile(d / "p.strp") == (N_LAYERS, N_EXPERT, 6, ranked)
    bad = bytearray((d / "p.strp").read_bytes())
    bad[4] = 2
    (d / "v2.strp").write_bytes(bad)
    try:
        mp.read_profile(d / "v2.strp")
        raise AssertionError("version 2 accepted")
    except ValueError as e:
        assert "version 2" in str(e)

    # a split GGUF: found from its first shard, same answer as the single file
    make_split(d)
    assert [p.name for p in mp.shards([d / "s-00001-of-00002.gguf"])] == ["s-00001-of-00002.gguf", "s-00002-of-00002.gguf"]
    check_info(mp.inspect([d / "s-00001-of-00002.gguf"]))
    (d / "s-00002-of-00002.gguf").rename(d / "gone.gguf")
    try:
        mp.shards([d / "s-00001-of-00002.gguf"])
        raise AssertionError("missing shard accepted")
    except SystemExit as e:
        assert "missing" in str(e)

    assert mp.main([str(d / "m.gguf"), "--cache-mib", "1", "--trace", str(d / "t.bin"), "--out", str(d / "cli.strp")]) == 0
    assert mp.main([str(d / "m.gguf"), "--cache-mib", "1", "--whole-layers", "1"]) == 0
    assert mp.main([str(d / "m.gguf"), "--vram-gib", "1", "--reserve-gib", "1"]) == 1   # no room: says so, exit 1

    # llama.cpp's reader (strata-moe branch) reads what this wrote; built by cmake into build/
    reader = Path(__file__).resolve().parent.parent / "build" / "read_profile"
    if reader.exists():
        out = subprocess.run([reader, d / "p.strp", str(N_LAYERS), str(N_EXPERT)], capture_output=True, text=True, check=True)
        assert [tuple(map(int, ln.split())) for ln in out.stdout.splitlines()] == ranked
        huge = bytearray((d / "p.strp").read_bytes())
        struct.pack_into("<I", huge, 20, 0xFFFFFFF0)          # n_ranked: more pairs than the model has
        (d / "huge.strp").write_bytes(huge)
        for p, msg in (("p.strp", "made for 3 layers x 8 experts"), ("v2.strp", "version"), ("huge.strp", "ranks 4294967280 pairs")):
            dims = (N_LAYERS + 1, N_EXPERT) if p == "p.strp" else (N_LAYERS, N_EXPERT)
            r = subprocess.run([reader, d / p, *map(str, dims)], capture_output=True, text=True)
            assert r.returncode == 1 and msg in r.stderr, (msg, r.stderr)
        print("test_moe_plan: C++ reader agrees")
    else:
        print("test_moe_plan: build/read_profile not built, C++ reader not checked")
    print("test_moe_plan: ok")


if __name__ == "__main__":
    main()
