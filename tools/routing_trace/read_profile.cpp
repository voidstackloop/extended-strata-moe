// tools/routing_trace/read_profile.cpp - prints what llama.cpp's MoE cache reads from a STRP profile, one "layer expert"
// per line, so tools/test_moe_plan.py can check the Python writer against the C++ reader.
//   read_profile PROFILE N_LAYER N_EXPERT
#include "llama-moe-cache.h"

#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s PROFILE N_LAYER N_EXPERT\n", argv[0]);
        return 2;
    }
    std::vector<std::pair<int32_t, int32_t>> ranked;
    std::string err;
    if (!llama_moe_cache_read_profile(argv[1], std::atoi(argv[2]), std::atoi(argv[3]), ranked, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    for (const auto & [l, e] : ranked) {
        std::printf("%d %d\n", l, e);
    }
    return 0;
}
