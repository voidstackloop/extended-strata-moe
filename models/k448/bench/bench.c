// One-token routed-expert matvec speed per quant type on the CPU backend of the native build (C:\Projects\llama.cpp-strata).
// Shapes follow k448's experts: gate/up (k=2048, m=1024) and down (k=1024, m=2048), 10 of 128 experts per op, ids rotated so the
// weights stream from RAM, not from the L3 cache. Result: us per op, Gweights/s, GB/s.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

typedef void (*set_threads_t)(ggml_backend_t, int);

static double now_s(void) {
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double) c.QuadPart / (double) f.QuadPart;
}

static unsigned long long rng = 88172645463325252ULL;
static unsigned int rnd(void) {
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (unsigned int) (rng >> 11);
}
static float frand(void) { return (float) (rnd() & 0xFFFFFF) / 16777216.0f - 0.5f; }

static void bench(ggml_backend_t be, set_threads_t st, enum ggml_type type, const char * shape, int k, int m, int n_expert, int n_used, int threads, int iters) {
    ggml_quantize_init(type);
    struct ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
    struct ggml_context * ctx = ggml_init(ip);
    struct ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, k, m, n_expert);
    struct ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, 1);
    struct ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    struct ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { printf("%-8s %-7s alloc failed\n", ggml_type_name(type), shape); ggml_free(ctx); return; }
    st(be, threads);

    size_t eb = ggml_row_size(type, k) * (size_t) m;
    float * src = (float *) malloc(sizeof(float) * (size_t) k * m);
    char * blk[2];
    for (int v = 0; v < 2; v++) {
        for (size_t i = 0; i < (size_t) k * m; i++) src[i] = frand();
        blk[v] = (char *) malloc(eb);
        ggml_quantize_chunk(type, src, blk[v], 0, m, k, NULL);
    }
    for (int e = 0; e < n_expert; e++) ggml_backend_tensor_set(as, blk[e & 1], (size_t) e * eb, eb);
    float * bv = (float *) malloc(sizeof(float) * k);
    for (int i = 0; i < k; i++) bv[i] = frand();
    ggml_backend_tensor_set(b, bv, 0, sizeof(float) * k);

    static int sets[32][16];
    for (int s = 0; s < 32; s++) {
        for (int j = 0; j < n_used; j++) {
            int e, dup;
            do {
                e = (int) (rnd() % (unsigned) n_expert);
                dup = 0;
                for (int q = 0; q < j; q++) if (sets[s][q] == e) dup = 1;
            } while (dup);
            sets[s][j] = e;
        }
    }
    for (int w = 0; w < 8; w++) {
        ggml_backend_tensor_set(ids, sets[w], 0, sizeof(int) * n_used);
        ggml_backend_graph_compute(be, gf);
    }
    double t0 = now_s();
    for (int it = 0; it < iters; it++) {
        ggml_backend_tensor_set(ids, sets[it & 31], 0, sizeof(int) * n_used);
        ggml_backend_graph_compute(be, gf);
    }
    double dt = (now_s() - t0) / iters;
    double w = (double) n_used * m * k;
    printf("%-8s %-7s thr=%2d  %7.1f us/op  %6.1f Gweights/s  %6.1f GB/s  (%.2f bits/weight)\n", ggml_type_name(type), shape, threads, dt * 1e6, w / dt / 1e9, (double) n_used * eb / dt / 1e9, 8.0 * eb / ((double) m * k));
    fflush(stdout);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    free(src); free(blk[0]); free(blk[1]); free(bv);
}

int main(int argc, char ** argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 10;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
    if (!be) { printf("no CPU backend\n"); return 1; }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
    set_threads_t st = (set_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (!st) { printf("no set_n_threads\n"); return 1; }
    enum ggml_type types[] = { GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q6_K };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        bench(be, st, types[i], "gate/up", 2048, 1024, 128, 10, threads, 400);
        bench(be, st, types[i], "down", 1024, 2048, 128, 10, threads, 400);
    }
    return 0;
}
