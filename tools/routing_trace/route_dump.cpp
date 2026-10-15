// tools/routing_trace/route_dump.cpp - per token and layer, the layer's input (the residual stream, `l_out` of the layer before)
// and the experts its router picked (`ffn_moe_topk`), to measure how well a layer's experts can be guessed before its
// attention runs (tools/route_predict.py).  CPU-only is fine.
//
//   route_dump MODEL.gguf TEXT.txt OUT.bin [n_tokens=1024] [n_threads=8]
//
// OUT.bin: records of int32 {kind, layer, n_tok, width} then n_tok*width values: kind 0 = the residual stream after the layer
// (f32, width n_embd), kind 1 = the router's top-k (i32, width k), kind 2 = a new decode starting at token `layer` (no data).
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Dump {
    FILE * out = nullptr;
    std::set<std::string> done;   // tensors already written for this decode
    std::vector<char> buf;
};

bool on_tensor(ggml_tensor * t, bool ask, void * user) {
    int layer = -1;
    int kind = -1;
    if (std::sscanf(t->name, "l_out-%d", &layer) == 1 && t->type == GGML_TYPE_F32) {
        kind = 0;
    } else if (std::sscanf(t->name, "ffn_moe_topk-%d", &layer) == 1 && t->type == GGML_TYPE_I32) {
        kind = 1;
    }
    if (kind < 0) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    auto * d = static_cast<Dump *>(user);
    if (!d->done.insert(t->name).second) {
        return true;   // a second router under the same name (GroveMoE): the first is the experts proper
    }
    const int32_t width = (int32_t) t->ne[0];
    const int32_t n_tok = (int32_t) (ggml_nelements(t) / width);
    const size_t row = (size_t) width * ggml_type_size(t->type);
    d->buf.resize(row * n_tok);
    for (int32_t j = 0; j < n_tok; ++j) {   // row by row: top-k is a view of an argsort
        ggml_backend_tensor_get(t, d->buf.data() + j * row, j * t->nb[1], row);
    }
    const int32_t hdr[4] = { kind, layer, n_tok, width };
    std::fwrite(hdr, sizeof hdr, 1, d->out);
    std::fwrite(d->buf.data(), 1, d->buf.size(), d->out);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf TEXT.txt OUT.bin [n_tokens=1024] [n_threads=8]\n", argv[0]);
        return 2;
    }
    const int max_tok = argc > 4 ? std::atoi(argv[4]) : 1024;
    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    Dump d;
    d.out = std::fopen(argv[3], "wb");
    if (!d.out) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), true, false);
    if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    n = std::min(n, max_tok);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::max(n, 256);
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = argc > 5 ? std::atoi(argv[5]) : 8;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &d;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create context\n"); return 1; }

    // one ubatch per decode, every token an output (the last layer is otherwise computed for the output rows only)
    const int piece = (int) llama_n_ubatch(ctx);
    llama_batch batch = llama_batch_init(piece, 0, 1);
    for (int i = 0; i < n; i += piece) {
        const int m = std::min(piece, n - i);
        batch.n_tokens = m;
        for (int j = 0; j < m; ++j) {
            batch.token[j] = toks[i + j];
            batch.pos[j] = i + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = 1;
        }
        const int32_t mark[4] = { 2, i, 0, 0 };
        std::fwrite(mark, sizeof mark, 1, d.out);
        d.done.clear();
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed at %d\n", i); return 1; }
        std::fprintf(stderr, "\r%d / %d tokens", i + m, n);
    }
    std::fprintf(stderr, "\n");
    std::fclose(d.out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
