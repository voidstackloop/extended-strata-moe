// tools/routing_trace/routing_trace.cpp - which experts any llama.cpp MoE model routes, per layer and token.
//
// Runs a text file through the model and records every `ffn_moe_topk-<layer>` tensor (the router's top-k ids,
// llama.cpp's build_moe_ffn names them so) in Strata's `--dump-routing` layout: per token and layer,
// int32 layer, int32 k, k int32 ids, k float32 weights.  The weights are written as 0: nothing downstream
// (tools/moe_plan.py, Strata's make_profile.py) reads them.  CPU-only is fine; it needs no CUDA toolkit.
//
//   routing_trace MODEL.gguf TEXT.txt OUT.bin [n_ctx=4096] [n_threads=4]
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Trace {
    FILE* out = nullptr;
    long records = 0;
    std::vector<int32_t> buf;
    std::vector<bool> layer_done;   // this graph already recorded the layer: reset before every llama_decode
};

bool on_tensor(ggml_tensor* t, bool ask, void* user) {
    int layer = -1;
    if (std::sscanf(t->name, "ffn_moe_topk-%d", &layer) != 1 || t->type != GGML_TYPE_I32) return ask ? false : true;
    if (ask) return true;
    auto* tr = static_cast<Trace*>(user);
    // a layer with a second router (GroveMoE's chunk experts) names it the same; the first is the experts proper,
    // the ones the MoE cache holds
    if (layer >= (int)tr->layer_done.size()) tr->layer_done.resize(layer + 1, false);
    if (tr->layer_done[layer]) return true;
    tr->layer_done[layer] = true;
    const int32_t k = (int32_t)t->ne[0];
    const int64_t n_tok = ggml_nelements(t) / k;
    tr->buf.resize(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, tr->buf.data(), 0, ggml_nbytes(t));
    } else {  // a view of argsort: copy row by row
        for (int64_t j = 0; j < n_tok; ++j) ggml_backend_tensor_get(t, tr->buf.data() + j * k, j * t->nb[1], k * sizeof(int32_t));
    }
    const std::vector<float> zeros(k, 0.0f);
    for (int64_t j = 0; j < n_tok; ++j) {
        const int32_t hdr[2] = {layer, k};
        std::fwrite(hdr, sizeof hdr, 1, tr->out);
        std::fwrite(tr->buf.data() + j * k, sizeof(int32_t), k, tr->out);
        std::fwrite(zeros.data(), sizeof(float), k, tr->out);
        ++tr->records;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf TEXT.txt OUT.bin [n_ctx=4096] [n_threads=4]\n", argv[0]);
        return 2;
    }
    const int n_ctx = argc > 4 ? std::atoi(argv[4]) : 4096;
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();

    Trace tr;
    tr.out = std::fopen(argv[3], "wb");
    if (!tr.out) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), true, false);
    if (n < 0) {   // more tokens than bytes: -n is the count needed
        toks.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks.data(), (int32_t)toks.size(), true, false);
    }
    if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = cp.n_ubatch = 512;
    if (argc > 5) {   // the default is 4; tracing is a long prompt, so it scales with cores
        cp.n_threads = cp.n_threads_batch = std::atoi(argv[5]);
    }
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &tr;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "cannot create context\n"); return 1; }

    // independent windows of n_ctx tokens; each window decoded in pieces of one ubatch.  Every token is an output:
    // llama.cpp computes the LAST layer only for output rows, so llama_batch_get_one (one output per batch) would trace
    // that layer on 1 token in n_batch.  Costs a piece x n_vocab logits buffer.
    // One ubatch per llama_decode, as the context really has it (it may lower n_ubatch): layer_done is reset per decode,
    // so a decode split into several ubatches would record the first and drop the rest
    const int piece = (int)std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
    llama_batch batch = llama_batch_init(piece, 0, 1);
    for (int w = 0; w < n; w += n_ctx) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int end = std::min(n, w + n_ctx);
        for (int i = w; i < end; i += piece) {
            const int m = std::min(piece, end - i);
            batch.n_tokens = m;
            for (int j = 0; j < m; ++j) {
                batch.token[j] = toks[i + j];
                batch.pos[j] = i - w + j;
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j] = 1;
            }
            std::fill(tr.layer_done.begin(), tr.layer_done.end(), false);
            if (llama_decode(ctx, batch) != 0) {
                std::fprintf(stderr, "decode failed at token %d\n", i);
                return 1;
            }
        }
        std::fprintf(stderr, "\r%d / %d tokens", end, n);
    }
    std::fprintf(stderr, "\n%d tokens, %ld routing records -> %s\n", n, tr.records, argv[3]);
    std::fclose(tr.out);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return tr.records > 0 ? 0 : 1;
}
