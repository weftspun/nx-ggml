// Phase 1 throwaway harness: proves ggml's CPU backend compiles and computes
// correctly via the ggml-backend API (ggml_backend_cpu_init +
// ggml_backend_graph_compute), the same path the NIF will use from Phase 3
// onward for both CPU and (later) Vulkan. Not part of the eventual NIF build.

#include <cstdio>
#include <cstdlib>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

int main() {
    // 1. Build the graph description (no tensor storage allocated yet).
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        return 1;
    }

    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    struct ggml_tensor * sum = ggml_add(ctx, a, b);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, sum);

    // 2. Materialize storage for every tensor in ctx on the CPU backend.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "ggml_backend_cpu_init failed\n");
        return 1;
    }

    struct ggml_backend_buffer * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "ggml_backend_alloc_ctx_tensors failed\n");
        return 1;
    }

    // 3. Populate inputs and compute.
    float a_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_data[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    ggml_backend_tensor_set(a, a_data, 0, sizeof(a_data));
    ggml_backend_tensor_set(b, b_data, 0, sizeof(b_data));

    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute failed: %d\n", status);
        return 1;
    }

    // 4. Read back and verify.
    float result[4];
    ggml_backend_tensor_get(sum, result, 0, sizeof(result));

    float expected[4] = {11.0f, 22.0f, 33.0f, 44.0f};
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        printf("result[%d] = %f (expected %f)\n", i, result[i], expected[i]);
        if (result[i] != expected[i]) {
            ok = 0;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);

    if (!ok) {
        fprintf(stderr, "MISMATCH\n");
        return 1;
    }

    printf("OK\n");
    return 0;
}
