#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>
#include <nccl.h>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" {
#include "ds4_gpu.h"
}

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u,
    /* SSD expert-streaming (Part 2 port): resident GPU LRU-cache budgets. */
    DS4_CUDA_STREAM_EXPERT_DEFAULT = 8u * 64u,
    DS4_CUDA_STREAM_EXPERT_MAX = 61u * 384u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
static int g_quality_mode;

#define DS4_CUDA_MAX_DEVICES 16

/* SSD expert-streaming global ON/OFF (Part 2). Declared early so the CUDA-graph
 * capture gates (ds4_gpu_pp_chunk_graph_capture_begin, decode_graph_can_capture)
 * can refuse capture in streaming mode — the streaming gather does host-side
 * LRU work + cudaMemcpy that is illegal during graph capture (SEAM 2). Policy
 * flag, identical on every GPU, so it stays scalar (not per-device). */
static int g_ssd_streaming_mode;

/* CUDA Graph decode infrastructure (Phase 0-1) */
static cudaStream_t g_cuda_decode_stream;
static int g_cuda_decode_stream_created;
static void *g_cublas_workspace;
static uint64_t g_cublas_workspace_bytes;

/* Pipeline Parallelism (PP) state — forward declarations for ds4_decode_stream */
static int g_pp_decode_active;
static cudaStream_t g_pp_stream[DS4_CUDA_MAX_DEVICES];
static int g_pp_work_streams_enabled;

static cudaStream_t ds4_decode_stream(void) {
    if (getenv("DS4_CUDA_PP_DEFAULT_STREAM") != NULL) return 0;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        if (g_pp_decode_active && g_pp_work_streams_enabled &&
            dev >= 0 && dev < DS4_CUDA_MAX_DEVICES && g_pp_stream[dev]) {
            return g_pp_stream[dev];
        }
        if (!g_cuda_decode_stream_created) return 0;
        /* In legacy PP mode, only GPU 0's decode stream is valid.
         * Other GPUs use their default stream unless PP work streams are enabled. */
        if (dev != 0) return 0;
    }
    if (!g_cuda_decode_stream_created) return 0;
    return g_cuda_decode_stream;
}

/* Phase 3: captured decode graph state */
static cudaGraph_t g_cuda_decode_graph;
static cudaGraphExec_t g_cuda_decode_graph_exec;
static int g_cuda_decode_graph_captured;
static cudaGraphExec_t g_sub_graph_exec[2][DS4_GPU_MAX_LAYER];
static int g_sub_graph_exec_ready[2][DS4_GPU_MAX_LAYER];
static cudaGraph_t g_pp_chunk_graph[DS4_CUDA_MAX_DEVICES];
static cudaGraphExec_t g_pp_chunk_graph_exec[DS4_CUDA_MAX_DEVICES];
static int g_pp_chunk_graph_ready[DS4_CUDA_MAX_DEVICES];
/* Phase E: kernel patch records */
typedef struct {
    cudaGraphNode_t node;
    cudaKernelNodeParams params;
} local_kernel_patch;
static local_kernel_patch g_local_patches[2][DS4_GPU_MAX_LAYER][16];
static int g_graph_patch_count[2][DS4_GPU_MAX_LAYER];
static int g_decode_graph_capture_disabled_notice_printed;

static int cuda_model_direct_host_access_allowed(void);
static int cuda_moe_temp_weights_enabled(void);

static int ds4_gpu_tp_selftest(int ngpu);

/* Phase 2: device-resident decode params for CUDA Graph replay.
 * When ds4_cuda_params_active != 0, kernels read token/pos/etc. from
 * ds4_cuda_dev_params instead of their immediate kernel arguments.
 * This allows captured CUDA Graphs to use updated values on each replay. */
__device__ ds4_cuda_decode_params ds4_cuda_dev_params;
__device__ int ds4_cuda_params_active;

/* Copy host params to the device global symbol. Returns 1 on success. */
extern "C" int ds4_gpu_decode_params_push(const ds4_cuda_decode_params *host) {
    cudaStream_t s = ds4_decode_stream();
    cudaError_t ce = cudaMemcpyToSymbolAsync(
            ds4_cuda_dev_params, host, sizeof(*host), 0, cudaMemcpyHostToDevice, s);
    if (ce != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    const int active = 1;
    ce = cudaMemcpyToSymbolAsync(
            ds4_cuda_params_active, &active, sizeof(active), 0, cudaMemcpyHostToDevice, s);
    if (ce != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    return 1;
}

/* Deactivate device params (restore immediate-argument behavior). */
extern "C" int ds4_gpu_decode_params_deactivate(void) {
    const int inactive = 0;
    cudaStream_t s = ds4_decode_stream();
    cudaError_t ce = cudaMemcpyToSymbolAsync(
            ds4_cuda_params_active, &inactive, sizeof(inactive), 0, cudaMemcpyHostToDevice, s);
    if (ce != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    return 1;
}

extern "C" void ds4_gpu_pp_deactivate_decode_params_all(int ngpu) {
    int saved = 0;
    (void)cudaGetDevice(&saved);
    for (int g = 0; g < ngpu; ++g) {
        (void)cudaSetDevice(g);
        (void)ds4_gpu_decode_params_deactivate();
        (void)cudaDeviceSynchronize();
        (void)cudaGetLastError();
    }
    (void)cudaSetDevice(saved);
}

extern "C" void ds4_debug_decode_symbol_token(const char *where, uint32_t host_token, uint32_t n_vocab) {
    if (getenv("DS4_CUDA_PP_DEBUG") == NULL) return;
    int dev = -1;
    (void)cudaGetDevice(&dev);
    int active = -1;
    ds4_cuda_decode_params p;
    memset(&p, 0, sizeof(p));
    cudaMemcpyFromSymbol(&active, ds4_cuda_params_active, sizeof(active), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&p, ds4_cuda_dev_params, sizeof(p), 0, cudaMemcpyDeviceToHost);
    fprintf(stderr, "ds4: %s dev=%d active=%d sym_token=%d host_token=%u n_vocab=%u\n",
            where, dev, active, p.token, host_token, n_vocab);
}

/* Phase 3: Capture the work submitted to the decode stream since the last
 * begin_commands as a CUDA Graph. Must be called after the encode is complete
 * but before end_commands (which would force a synchronize). */
extern "C" int ds4_gpu_decode_graph_capture(void) {
    if (!g_cuda_decode_stream_created) return 0;
    /* Free any previous graph */
    if (g_cuda_decode_graph_exec) {
        (void)cudaGraphExecDestroy(g_cuda_decode_graph_exec);
        g_cuda_decode_graph_exec = NULL;
    }
    if (g_cuda_decode_graph) {
        (void)cudaGraphDestroy(g_cuda_decode_graph);
        g_cuda_decode_graph = NULL;
    }
    g_cuda_decode_graph_captured = 0;
    cudaError_t ce = cudaStreamBeginCapture(g_cuda_decode_stream,
                                             cudaStreamCaptureModeGlobal);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA graph capture begin failed: %s\n", cudaGetErrorString(ce));
        (void)cudaGetLastError();
        return 0;
    }
    /* Re-run the encode — this time captured into the graph */
    /* (The caller is responsible for calling encode between begin/end capture) */
    return 1;
}

/* Finalize capture and instantiate the graph. */
extern "C" int ds4_gpu_decode_graph_capture_end(void) {
    if (!g_cuda_decode_stream_created) return 0;
    cudaError_t ce = cudaStreamEndCapture(g_cuda_decode_stream, &g_cuda_decode_graph);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA graph capture end failed: %s\n", cudaGetErrorString(ce));
        (void)cudaGetLastError();
        g_cuda_decode_graph = NULL;
        return 0;
    }
    if (!g_cuda_decode_graph) return 0;
    ce = cudaGraphInstantiate(&g_cuda_decode_graph_exec, g_cuda_decode_graph,
                               NULL, NULL, 0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA graph instantiate failed: %s\n", cudaGetErrorString(ce));
        (void)cudaGetLastError();
        (void)cudaGraphDestroy(g_cuda_decode_graph);
        g_cuda_decode_graph = NULL;
        return 0;
    }
    g_cuda_decode_graph_captured = 1;
    fprintf(stderr, "ds4: CUDA decode graph captured and instantiated\n");
    return 1;
}
extern "C" int ds4_gpu_decode_graph_capture_end_store(int part, int layer) {
    if (layer < 0) {
        /* Abort capture on failure: end capture and discard the graph */
        cudaGraph_t graph = NULL;
        cudaError_t ce = cudaStreamEndCapture(g_cuda_decode_stream, &graph);
        if (graph) (void)cudaGraphDestroy(graph);
        if (ce != cudaSuccess) (void)cudaGetLastError();
        return 0;
    }
    if (layer >= DS4_GPU_MAX_LAYER || !g_cuda_decode_stream_created) return 0;
    cudaGraph_t graph = NULL;
    cudaError_t ce = cudaStreamEndCapture(g_cuda_decode_stream, &graph);
    if (ce != cudaSuccess || !graph) {
        if (ce != cudaSuccess) (void)cudaGetLastError();
        return 0;
    }
    ce = cudaGraphInstantiate(&g_sub_graph_exec[part][layer], graph, NULL, NULL, 0);

    /* Phase E: collect kernel node patches */
    {
        size_t n = 0;
        if (cudaGraphGetNodes(graph, NULL, &n) == cudaSuccess && n > 0) {
            std::vector<cudaGraphNode_t> nodes(n);
            (void)cudaGraphGetNodes(graph, nodes.data(), &n);
            int pc = 0;
            for (size_t i = 0; i < n && pc < 16; i++) {
                cudaGraphNodeType t;
                if (cudaGraphNodeGetType(nodes[i], &t) != cudaSuccess) continue;
                if (t != cudaGraphNodeTypeKernel) continue;
                cudaKernelNodeParams kp = {};
                if (cudaGraphKernelNodeGetParams(nodes[i], &kp) != cudaSuccess) continue;
                local_kernel_patch lp = { nodes[i], kp };
                g_local_patches[part][layer][pc++] = lp;
            }
            g_graph_patch_count[part][layer] = pc;
        } else {
            (void)cudaGetLastError();
        }
    }
    (void)cudaGraphDestroy(graph);
    if (ce != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    g_sub_graph_exec_ready[part][layer] = 1;
    return 1;
}

extern "C" int ds4_gpu_decode_subgraph_launch(int part, int layer) {
    if (layer < 0 || layer >= DS4_GPU_MAX_LAYER || !g_sub_graph_exec[part][layer]) return 0;
    return cudaGraphLaunch(g_sub_graph_exec[part][layer], ds4_decode_stream()) == cudaSuccess ? 1 : 0;
}

extern "C" int ds4_gpu_pp_chunk_graph_capture_begin(int gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_stream[gpu]) return 0;
    /* SEAM 2: refuse PP chunk-graph capture when SSD streaming is active — the
     * per-layer expert gather runs host-side work illegal under capture. */
    if (g_ssd_streaming_mode) return 0;
    cudaSetDevice(gpu);
    cudaStream_t stream = ds4_decode_stream();
    if (!stream) {
        fprintf(stderr, "ds4: PP chunk graph capture begin failed gpu%d: no PP work stream\n", gpu);
        return 0;
    }
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaError_t se = cudaStreamIsCapturing(stream, &st);
    if (se == cudaSuccess && st != cudaStreamCaptureStatusNone) {
        fprintf(stderr, "ds4: PP chunk graph capture skipped gpu%d: stream already capturing status=%d\n",
                gpu, (int)st);
        return 0;
    }
    (void)cudaGetLastError();
    if (g_pp_chunk_graph_exec[gpu]) {
        (void)cudaGraphExecDestroy(g_pp_chunk_graph_exec[gpu]);
        g_pp_chunk_graph_exec[gpu] = NULL;
    }
    if (g_pp_chunk_graph[gpu]) {
        (void)cudaGraphDestroy(g_pp_chunk_graph[gpu]);
        g_pp_chunk_graph[gpu] = NULL;
    }
    g_pp_chunk_graph_ready[gpu] = 0;
    cudaError_t ce = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP chunk graph capture begin failed gpu%d: %s\n",
                gpu, cudaGetErrorString(ce));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_pp_chunk_graph_capture_end(int gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_stream[gpu]) return 0;
    cudaSetDevice(gpu);
    cudaStream_t stream = ds4_decode_stream();
    if (!stream) return 0;
    cudaGraph_t graph = NULL;
    cudaError_t ce = cudaStreamEndCapture(stream, &graph);
    if (ce != cudaSuccess || !graph) {
        fprintf(stderr, "ds4: PP chunk graph capture end failed gpu%d: %s\n",
                gpu, cudaGetErrorString(ce));
        if (graph) (void)cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        return 0;
    }
    ce = cudaGraphInstantiate(&g_pp_chunk_graph_exec[gpu], graph, NULL, NULL, 0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP chunk graph instantiate failed gpu%d: %s\n",
                gpu, cudaGetErrorString(ce));
        (void)cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        return 0;
    }
    g_pp_chunk_graph[gpu] = graph;
    g_pp_chunk_graph_ready[gpu] = 1;
    fprintf(stderr, "ds4: PP chunk graph captured gpu%d\n", gpu);
    return 1;
}

extern "C" int ds4_gpu_pp_chunk_graph_capture_abort(int gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_stream[gpu]) return 0;
    cudaSetDevice(gpu);
    cudaStream_t stream = ds4_decode_stream();
    if (!stream) return 0;
    cudaGraph_t graph = NULL;
    cudaError_t ce = cudaStreamEndCapture(stream, &graph);
    if (graph) (void)cudaGraphDestroy(graph);
    if (ce != cudaSuccess) (void)cudaGetLastError();
    return 1;
}

extern "C" int ds4_gpu_pp_chunk_graph_launch(int gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_chunk_graph_ready[gpu] ||
        !g_pp_chunk_graph_exec[gpu]) return 0;
    cudaSetDevice(gpu);
    cudaError_t ce = cudaGraphLaunch(g_pp_chunk_graph_exec[gpu], ds4_decode_stream());
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP chunk graph launch failed gpu%d: %s\n",
                gpu, cudaGetErrorString(ce));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_pp_chunk_graph_ready(int gpu) {
    return gpu >= 0 && gpu < DS4_CUDA_MAX_DEVICES &&
           g_pp_chunk_graph_ready[gpu] && g_pp_chunk_graph_exec[gpu] != NULL;
}

extern "C" int ds4_gpu_decode_graph_patch_pre(int layer, uint32_t pos, uint32_t raw_row, uint32_t n_raw) {
    int pc = g_graph_patch_count[0][layer];
    for (int i = 0; i < pc; i++) {
        local_kernel_patch *p = &g_local_patches[0][layer][i];
        (void)cudaGraphKernelNodeSetParams(p->node, &p->params);
    }
    (void)pos; (void)raw_row; (void)n_raw;
    return 1;
}

extern "C" int ds4_gpu_decode_graph_patch_post(int layer, uint32_t pos) {
    int pc = g_graph_patch_count[1][layer];
    for (int i = 0; i < pc; i++) {
        local_kernel_patch *p = &g_local_patches[1][layer][i];
        (void)cudaGraphKernelNodeSetParams(p->node, &p->params);
    }
    (void)pos;
    return 1;
}

extern "C" int ds4_gpu_decode_subgraphs_ready(void) {
    for (int p = 0; p < 2; p++) for (int i = 0; i < DS4_GPU_MAX_LAYER; i++) if (!g_sub_graph_exec_ready[p][i]) return 0;
    return 1;
}


/* Launch the captured decode graph. */
extern "C" int ds4_gpu_decode_graph_launch(void) {
    if (!g_cuda_decode_graph_captured || !g_cuda_decode_graph_exec) return 0;
    /* The graph is launched on the decode stream. It replays all captured kernel
     * launches, reading updated values from ds4_cuda_dev_params. */
    cudaError_t ce = cudaGraphLaunch(g_cuda_decode_graph_exec, g_cuda_decode_stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA graph launch failed: %s\n", cudaGetErrorString(ce));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

/* Returns 1 if the decode graph has been captured and is ready for replay. */
extern "C" int ds4_gpu_decode_graph_captured(void) {
    return g_cuda_decode_graph_captured && g_cuda_decode_graph_exec != NULL;
}

extern "C" int ds4_gpu_decode_graph_can_capture(void) {
    /* SEAM 2: in SSD-streaming mode the per-step expert gather does host-side
     * LRU bookkeeping + cudaMemcpy that cannot run inside a captured graph and
     * would not refill the streamed buffer on replay (stale-expert garbage).
     * Disable decode-graph capture; the streaming copy latency dwarfs the
     * launch overhead the graph would have hidden. */
    if (g_ssd_streaming_mode) {
        if (!g_decode_graph_capture_disabled_notice_printed) {
            fprintf(stderr,
                    "ds4: CUDA graph capture disabled: SSD expert streaming is active "
                    "(experts are gathered on the host each step)\n");
            g_decode_graph_capture_disabled_notice_printed = 1;
        }
        return 0;
    }
    const int temp_moe_weights =
        cuda_moe_temp_weights_enabled() &&
        !g_model_device_owned &&
        !g_model_registered &&
        !cuda_model_direct_host_access_allowed();
    /* Compact selected-expert path is capture-safe; scratch gets allocated
     * on first use during the warmup token. */
    if (!temp_moe_weights || cuda_moe_temp_weights_enabled()) return 1;
    if (!g_decode_graph_capture_disabled_notice_printed) {
        fprintf(stderr,
                "ds4: CUDA graph capture disabled: MoE weights are streamed "
                "through transient/scratch buffers and are not replay-safe yet\n");
        g_decode_graph_capture_disabled_notice_printed = 1;
    }
    return 0;
}

/* Phase 2: Decode params — pinned host + device copies for dynamic update */
extern "C" int ds4_gpu_decode_params_alloc(ds4_cuda_decode_params **host,
                                            void                    **device,
                                            uint64_t                 *bytes) {
    *bytes = sizeof(ds4_cuda_decode_params);
    cudaError_t ce = cudaMallocHost((void **)host, (size_t)*bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: decode params host alloc failed: %s\n",
                cudaGetErrorString(ce));
        (void)cudaGetLastError();
        *host = NULL; *device = NULL; *bytes = 0;
        return 0;
    }
    memset(*host, 0, (size_t)*bytes);
    ce = cudaMalloc(device, (size_t)*bytes);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: decode params device alloc failed: %s\n",
                cudaGetErrorString(ce));
        (void)cudaGetLastError();
        cudaFreeHost(*host);
        *host = NULL; *device = NULL; *bytes = 0;
        return 0;
    }
    return 1;
}

extern "C" void ds4_gpu_decode_params_free(ds4_cuda_decode_params *host,
                                            void                    *device,
                                            uint64_t                 bytes) {
    if (device) (void)cudaFree(device);
    if (host)   (void)cudaFreeHost(host);
    (void)bytes;
}

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
    int device;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_temp_model_range {
    char *device_ptr;
    uint64_t bytes;
    const char *what;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
};

/* Tensor-Parallel weight shards: resident per-device slices the TP-aware decode
 * path uses. Kept in a registry SEPARATE from g_model_ranges so the existing
 * offset-keyed full-tensor lookups (prefill, non-TP decode) are never perturbed.
 * Each (stage,rank) device holds exactly its own shard of a sharded weight; the
 * TP encode resolves it by (parent offset, current device) via cuda_tp_shard_ptr. */
enum cuda_tp_shard_kind {
    DS4_TP_SHARD_COL = 0,    /* contiguous output-row slice (column-parallel)  */
    DS4_TP_SHARD_ROW = 1,    /* packed input-block slice    (row-parallel)     */
};
struct cuda_tp_shard {
    const void *host_base;  /* model_map of the parent tensor                  */
    uint64_t offset;        /* abs_offset of the parent (FULL) tensor          */
    int rank;               /* TP rank within its group                        */
    int k;                  /* TP degree                                       */
    int kind;               /* cuda_tp_shard_kind                              */
    int device;             /* CUDA device this shard lives on                 */
    char *device_ptr;       /* the shard's bytes on `device`                   */
    uint64_t bytes;         /* shard size                                      */
    uint64_t out_dim;       /* COL: rows on this rank; ROW: full out_dim       */
    uint64_t in_dim;        /* COL: full in_dim; ROW: this rank's in (bpr*32)  */
    uint64_t blocks;        /* COL: full blocks; ROW: this rank's blocks (bpr) */
};

/* =========================================================================
 * SSD expert streaming (Part 2 port from upstream antirez/ds4).
 * The resident GPU expert cache (cuda_stream_expert_cache) is an LRU of
 * recently-routed experts, backed by the OS page cache (1024GB host RAM) +
 * Optane SSD; the selected cache (cuda_stream_selected_cache) holds the
 * compacted gather of the current layer's routed experts. All three are
 * per-CUDA-device (see the [DS4_CUDA_MAX_DEVICES] globals below) because PP
 * runs each layer on a different GPU. ========================================*/
struct cuda_stream_selected_cache {
    int valid;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t slot_count;
    uint32_t compact_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    int32_t *slot_selected_ptr;
    uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
};

struct cuda_stream_expert_cache_slot {
    int valid;
    const void *model_map;
    uint64_t model_size;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t expert;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    uint64_t age;
};

struct cuda_stream_expert_cache {
    int valid;
    uint32_t capacity;
    uint32_t count;
    uint64_t tick;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char *gate_ptr;
    char *up_ptr;
    char *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    std::vector<cuda_stream_expert_cache_slot> slots;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static std::vector<cuda_tp_shard> g_tp_shards;
static uint64_t g_tp_shard_bytes;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp_dev[DS4_CUDA_MAX_DEVICES];
static uint64_t g_cuda_tmp_bytes_dev[DS4_CUDA_MAX_DEVICES];
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static int g_moe_temp_weights_notice_printed;
static int g_moe_compact_notice_printed;

/* SSD expert-streaming state (Part 2 port). PP runs each layer on a different
 * GPU, so every device keeps its own resident/selected cache + H2D staging
 * pool — all [DS4_CUDA_MAX_DEVICES], indexed by the current cudaGetDevice().
 * Exceptions: g_ssd_streaming_mode is a global ON/OFF flag, and
 * g_stream_expert_budget_override is a one-time user setting (CLI) shared by
 * all GPUs — both stay scalar. The four runtime sizing/notice fields are
 * per-device OOM state (each 16GB GPU may shrink its cap independently).
 * g_ssd_streaming_mode is declared near the top (before the graph-capture gates). */
static cuda_stream_selected_cache g_stream_selected_cache[DS4_CUDA_MAX_DEVICES];
static cuda_stream_expert_cache   g_stream_expert_cache[DS4_CUDA_MAX_DEVICES];
static uint32_t g_stream_expert_budget_override;
static uint32_t g_stream_expert_runtime_cap[DS4_CUDA_MAX_DEVICES];
static uint32_t g_stream_expert_memory_cap_notice[DS4_CUDA_MAX_DEVICES];
static uint64_t g_stream_expert_runtime_gate_bytes[DS4_CUDA_MAX_DEVICES];
static uint64_t g_stream_expert_runtime_down_bytes[DS4_CUDA_MAX_DEVICES];
static void *g_stream_selected_stage_raw[DS4_CUDA_MAX_DEVICES][4];
static void *g_stream_selected_stage[DS4_CUDA_MAX_DEVICES][4];
static cudaEvent_t g_stream_selected_stage_event[DS4_CUDA_MAX_DEVICES][4];
static uint64_t g_stream_selected_stage_bytes[DS4_CUDA_MAX_DEVICES];
static cudaStream_t g_stream_selected_upload_stream[DS4_CUDA_MAX_DEVICES];

/* ===== GPU-side gather (Part 2 perf, behind DS4_STREAM_GPU_GATHER) =====
 * Eliminate the per-layer host round-trip (cudaStreamSynchronize + D2H of the
 * router-selected ids + host LRU scans + D2D compact copies) that caps streaming
 * decode at ~8.7 t/s. A device directory mirrors the host LRU so the MoE kernel
 * can read the resident expert buffers in place via a GPU-resolved slot index.
 * d_expert_slot_dir[dev] = int32[MAX_LAYERS * n_total_expert], entry = LRU slot
 * holding (layer_id, expert) or -1. layer_id is a small dense id keyed by
 * gate_offset (the fork passes layer=0 from one_tensor, but gate_offset is
 * unique per layer). Entirely ds4_cuda.cu-local; no ds4.c/Metal ABI change. */
#define DS4_STREAM_GATHER_MAX_LAYERS 64u
static int g_stream_gpu_gather = -1; /* -1=uninit; 0/1 resolved from env */
static std::unordered_map<uint64_t, uint32_t> g_stream_gate_off_layerid;
static int32_t *g_stream_expert_slot_dir[DS4_CUDA_MAX_DEVICES];
static uint32_t g_stream_dir_experts[DS4_CUDA_MAX_DEVICES];

/* Pipeline Parallelism (PP) state */
static int g_pp_ngpu;
static int g_pp_requested;
static int g_pp_topology_ready;
static int g_pp_resident_ready;
/* Tensor-Parallel (TP) over PP stages: g_tp_degree logical devices cooperate on
 * each layer. Logical device g -> stage g/g_tp_degree, rank g%g_tp_degree. When
 * g_tp_degree<=1, TP is OFF and the PP path is byte-for-byte unchanged. */
static int g_tp_degree;
static int g_pp_layer_start[DS4_CUDA_MAX_DEVICES];
static int g_pp_layer_end[DS4_CUDA_MAX_DEVICES];
static float *g_pp_active[DS4_CUDA_MAX_DEVICES];
static ncclComm_t g_pp_nccl_comms[DS4_CUDA_MAX_DEVICES];
/* Model shape passed from ds4.c (the .cu cannot see g_ds4_shape macros). */
static int g_model_n_layer = 43;
static int g_model_n_embd = 4096;
static int g_pp_nccl_ready;
static cudaEvent_t g_pp_event[DS4_CUDA_MAX_DEVICES];
static cudaEvent_t g_pp_copy_event[DS4_CUDA_MAX_DEVICES];
/* Tensor-Parallel collective scratch (GPU-count-agnostic; indexed by device id) */
static cudaStream_t g_tp_stream[DS4_CUDA_MAX_DEVICES];
static cudaEvent_t  g_tp_event[DS4_CUDA_MAX_DEVICES];
static float       *g_tp_recv[DS4_CUDA_MAX_DEVICES];
static uint32_t     g_tp_recv_floats[DS4_CUDA_MAX_DEVICES];
static int g_moe_scratch_inited;

/* Persistent MoE scratch: pre-allocated buffer for one layer's full expert
 * weights (gate+up+down, all 256 experts). Allocated once instead of
 * cudaMalloc/free per layer, making it compatible with CUDA Graph capture. */
static char *g_moe_scratch_gate;
static char *g_moe_scratch_up;
static char *g_moe_scratch_down;
static uint64_t g_moe_scratch_gate_bytes;
static uint64_t g_moe_scratch_up_bytes;
static uint64_t g_moe_scratch_down_bytes;

/* Selected-expert compact scratch: 6 experts' weights (gate+up+down).
 * Allocated once, reused per layer. Replaces the full 256-expert transfer
 * with a 6-expert compact copy, reducing H2D volume from 1.7GB to ~42MB. */
/* Compact MoE scratch is per-device: PP decode runs each layer on a different
 * GPU, so a single global buffer would be addressed cross-device and fault.
 * Index by the current CUDA device id. */
static char *g_moe_compact_gate[DS4_CUDA_MAX_DEVICES];
static char *g_moe_compact_up[DS4_CUDA_MAX_DEVICES];
static char *g_moe_compact_down[DS4_CUDA_MAX_DEVICES];
static int32_t *g_moe_compact_selected_dev[DS4_CUDA_MAX_DEVICES]; /* device: remapped slot indices */
static uint64_t g_moe_compact_per_expert[DS4_CUDA_MAX_DEVICES];   /* max(gate_expert_bytes, down_expert_bytes) */
static uint32_t g_moe_compact_slots[DS4_CUDA_MAX_DEVICES];        /* allocated capacity in (token*expert) slots */

static int cuda_ok(cudaError_t err, const char *what);
static int cuda_decode_graph_enabled(void);
static int cuda_model_direct_host_access_allowed(void);
static int cuda_moe_temp_weights_enabled(void);
static int cuda_cublas_decode_enabled(void);
static int ds4_gpu_pp_init(void);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
/* SSD expert-streaming (Part 2 port): forward decl — the streaming expert
 * cache (defined below cuda_model_stage_read) calls this I/O helper, which in
 * turn calls the staging-pool helpers, so the two reference each other. */
static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;

    int dev = -1;
    cudaError_t ge = cudaGetDevice(&dev);
    if (ge != cudaSuccess || dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s: invalid current device\n",
                what ? what : "scratch");
        (void)cudaGetLastError();
        return NULL;
    }

    if (g_cuda_tmp_dev[dev] && g_cuda_tmp_bytes_dev[dev] >= bytes) {
        if (getenv("DS4_CUDA_TMP_CHECK") != NULL) {
            cudaPointerAttributes a;
            cudaError_t ae = cudaPointerGetAttributes(&a, g_cuda_tmp_dev[dev]);
            if (ae == cudaSuccess) {
                fprintf(stderr, "ds4: CUDA tmp check reuse dev=%d ptr=%p attr.device=%d type=%d\n",
                        dev, g_cuda_tmp_dev[dev], a.device, (int)a.type);
            }
        }
        return g_cuda_tmp_dev[dev];
    }

    if (g_cuda_tmp_dev[dev]) {
        (void)cudaFree(g_cuda_tmp_dev[dev]);
        g_cuda_tmp_dev[dev] = NULL;
        g_cuda_tmp_bytes_dev[dev] = 0;
    }

    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s dev=%d (%.2f MiB): %s\n",
                what ? what : "scratch", dev, (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_cuda_tmp_dev[dev] = ptr;
    g_cuda_tmp_bytes_dev[dev] = bytes;

    if (getenv("DS4_CUDA_TMP_CHECK") != NULL) {
        cudaPointerAttributes a;
        cudaError_t ae = cudaPointerGetAttributes(&a, ptr);
        if (ae == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA tmp check alloc dev=%d ptr=%p attr.device=%d type=%d what=%s\n",
                    dev, ptr, a.device, (int)a.type, what ? what : "?");
        } else {
            fprintf(stderr, "ds4: CUDA tmp check failed dev=%d ptr=%p: %s\n",
                    dev, ptr, cudaGetErrorString(ae));
            (void)cudaGetLastError();
        }
    }

    return ptr;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static int cuda_model_direct_host_access_allowed(void) {
    if (g_model_hmm_direct &&
        getenv("DS4_CUDA_WEIGHT_CACHE") == NULL &&
        getenv("DS4_CUDA_WEIGHT_PRELOAD") == NULL) {
        return 1;
    }
    const char *direct_env = getenv("DS4_CUDA_DIRECT_MODEL");
    return direct_env && direct_env[0];
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);

    /* Check per-tensor cache first. PP resident weights must override the
     * registered-model shortcut so each GPU uses its local cached copy. */
    const uint64_t end = offset + bytes;
    int current_dev = -1;
    (void)cudaGetDevice(&current_dev);
    {
        auto exact = g_model_range_by_offset.find(offset);
        if (exact != g_model_range_by_offset.end()) {
            const cuda_model_range &r = g_model_ranges[exact->second];
            if (r.host_base == model_map && end >= offset && bytes <= r.bytes &&
                r.device == current_dev && !r.host_registered) {
                return r.device_ptr;
            }
        }
    }
    const char *fallback = NULL;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            if (r.device == current_dev && !r.host_registered) {
                return r.device_ptr + (offset - r.offset);
            }
            if (g_pp_decode_active) {
                if (!fallback) fallback = r.device_ptr + (offset - r.offset);
                continue;
            }
            if (!fallback) fallback = r.device_ptr + (offset - r.offset);
        }
        if (!g_pp_decode_active &&
            r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }
    if (fallback) {
        if (g_pp_decode_active) {
            fprintf(stderr,
                    "ds4: PP weight cache miss on current device; refusing fallback "
                    "dev=%d off=%llu bytes=%llu what=%s fallback=%p\n",
                    current_dev,
                    (unsigned long long)offset,
                    (unsigned long long)bytes,
                    what ? what : "?",
                    (const void *)fallback);
            return NULL;
        }
        return fallback;
    }

    if (g_pp_decode_active) {
        fprintf(stderr,
                "ds4: PP weight cache miss dev=%d off=%llu bytes=%llu what=%s; "
                "refusing host/global fallback\n",
                current_dev,
                (unsigned long long)offset,
                (unsigned long long)bytes,
                what ? what : "?");
        return NULL;
    }

    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (cuda_model_direct_host_access_allowed()) return cuda_model_ptr(model_map, offset);

    if (getenv("DS4_CUDA_NO_FD_CACHE") == NULL) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped | cudaHostRegisterReadOnly);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0, -1});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                            what ? what : "weights",
                            (double)bytes / 1048576.0);
                }
                return dev_ptr;
            }
            fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    void *dev = NULL;
    err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0, -1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_moe_temp_weights_enabled(void) {
    if (getenv("DS4_CUDA_NO_MOE_TEMP_WEIGHTS") != NULL) return 0;
    /* PP resident mode: weights are cached on each GPU, no H2D needed.
     * But if weights are not actually cached (e.g., OOM), fall through
     * to use compact MoE which streams from host memory. */
    if (g_pp_decode_active) return 0;
    const char *env = getenv("DS4_CUDA_MOE_TEMP_WEIGHTS");
    if (env && env[0]) return env[0] != '0';
    if (cuda_decode_graph_enabled()) return 1;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    (void)free_bytes;
    return (uint64_t)total_bytes <= 24ull * 1024ull * 1024ull * 1024ull;
}

static int cuda_cublas_decode_enabled(void) {
    if (getenv("DS4_CUDA_CUBLAS_DECODE_F16") != NULL) return 1;
    if (getenv("DS4_CUDA_CUBLAS_DECODE_F32") != NULL) return 1;
    return 0;
}

static const char *cuda_model_temp_range_alloc(
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what,
        cuda_temp_model_range *tmp) {
    if (!tmp) return NULL;
    tmp->device_ptr = NULL;
    tmp->bytes = 0;
    tmp->what = what;
    if (!model_map || bytes == 0 || offset > model_size || bytes > model_size - offset) return NULL;
    if (bytes > (uint64_t)SIZE_MAX) return NULL;

    /* Use persistent scratch for MoE when available (graph-mode compatible) */
    if (g_moe_scratch_inited) {
        int slot = -1;
        if (what && strstr(what, "moe_gate") && g_moe_scratch_gate && bytes <= g_moe_scratch_gate_bytes) slot = 0;
        else if (what && strstr(what, "moe_up") && g_moe_scratch_up && bytes <= g_moe_scratch_up_bytes) slot = 1;
        else if (what && strstr(what, "moe_down") && g_moe_scratch_down && bytes <= g_moe_scratch_down_bytes) slot = 2;
        if (slot >= 0) {
            char *scratch = slot == 0 ? g_moe_scratch_gate : slot == 1 ? g_moe_scratch_up : g_moe_scratch_down;
            /* Use the mmap'd model bytes as the async copy source. The source is
             * stable across queued copies; a shared pinned staging buffer is not,
             * because the host can overwrite it before the previous H2D DMA has
             * consumed it. CUDA Graph capture is disabled for this weight policy
             * until MoE weights are device-resident or gathered inside the graph. */
            const char *src = (const char *)model_map + offset;
            cudaError_t err = cudaMemcpyAsync(scratch, src, (size_t)bytes,
                                              cudaMemcpyHostToDevice, ds4_decode_stream());
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA moe scratch copy failed for %s: %s\n",
                        what ? what : "?", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
            tmp->device_ptr = scratch;
            tmp->bytes = bytes;
            return scratch;
        }
    }

    void *dev = NULL;
    if (getenv("DS4_CUDA_MTP_TRACE_ALLOC") != NULL) {
        int cur_dev = -1;
        size_t free_b = 0, total_b = 0;
        (void)cudaGetDevice(&cur_dev);
        (void)cudaMemGetInfo(&free_b, &total_b);
        fprintf(stderr,
                "ds4: CUDA MTP trace transient alloc request dev=%d %s %.2f MiB free %.2f/%.2f MiB\n",
                cur_dev,
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)free_b / 1048576.0,
                (double)total_b / 1048576.0);
    }
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA transient model alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        const uint64_t n = (bytes - done < chunk) ? (bytes - done) : chunk;
        err = cudaMemcpyAsync((char *)dev + done, src + done, (size_t)n,
                              cudaMemcpyHostToDevice, ds4_decode_stream());
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA transient model copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaStreamSynchronize(ds4_decode_stream());
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }

    tmp->device_ptr = (char *)dev;
    tmp->bytes = bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA transient-loaded %s %.2f MiB\n",
                what ? what : "weights",
                (double)bytes / 1048576.0);
    }
    return tmp->device_ptr;
}

static void cuda_model_temp_range_free(cuda_temp_model_range *tmp) {
    if (!tmp || !tmp->device_ptr) return;
    (void)cudaFree(tmp->device_ptr);
    tmp->device_ptr = NULL;
    tmp->bytes = 0;
    tmp->what = NULL;
}

static int cuda_moe_temp_weights_release(
        cuda_temp_model_range *gate,
        cuda_temp_model_range *up,
        cuda_temp_model_range *down) {
    const int has_temp =
        (gate && gate->device_ptr) ||
        (up && up->device_ptr) ||
        (down && down->device_ptr);
    int ok = 1;
    if (has_temp) {
        /* When using persistent scratch (g_moe_scratch_inited), the buffers
         * are never freed/synced here — just reset the tracking struct. */
        if (!g_moe_scratch_inited) {
            ok = cuda_ok(cudaStreamSynchronize(ds4_decode_stream()),
                         "routed_moe transient weights sync");
        }
    }
    if (!g_moe_scratch_inited) {
        cuda_model_temp_range_free(gate);
        cuda_model_temp_range_free(up);
        cuda_model_temp_range_free(down);
    } else {
        /* Only reset tracking pointers, don't free the persistent scratch */
        if (gate) { gate->device_ptr = NULL; gate->bytes = 0; }
        if (up)   { up->device_ptr   = NULL; up->bytes   = 0; }
        if (down) { down->device_ptr = NULL; down->bytes = 0; }
    }
    return ok;
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered) return 1;

    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map &&
            r.host_registered &&
            r.registered_base &&
            r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_MB", &present);
    if (present) return limit;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err == cudaSuccess) {
        (void)free_bytes;
        if ((uint64_t)total_bytes <= 24ull * 1024ull * 1024ull * 1024ull &&
            cuda_moe_temp_weights_enabled()) {
            return 0;
        }
    } else {
        (void)cudaGetLastError();
    }
    return UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", &present);
    if (present) return reserve;

    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed && getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (getenv("DS4_CUDA_NO_Q8_F16_CACHE") != NULL) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    if (getenv("DS4_CUDA_Q8_F16_ALL") != NULL) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE") == NULL;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (getenv("DS4_CUDA_NO_ATTN_Q_B_F16_CACHE") == NULL &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_use_dp4a(void) {
    return getenv("DS4_CUDA_NO_Q8_DP4A") == NULL;
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        getenv("DS4_CUDA_ATTENTION_OUTPUT_PRELOAD") == NULL &&
        getenv("DS4_CUDA_Q8_F16_ALL") == NULL) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (getenv("DS4_CUDA_NO_Q8_F32_CACHE") != NULL) return 0;
    if (getenv("DS4_CUDA_Q8_F32_ALL") != NULL) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return getenv("DS4_CUDA_ATTN_Q_B_F32_CACHE") != NULL;
    }
    return getenv("DS4_CUDA_Q8_F32_LARGE") != NULL &&
           in_dim == 1024u && out_dim == 32768u;
}

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    return dev;
}

static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f32_by_offset.find(offset);
    if (exact != g_q8_f32_by_offset.end()) {
        const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    g_q8_f32_bytes += out_bytes;
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA cached q8 fp32 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f32_bytes / 1073741824.0);
    }
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) return 0;
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

/* SSD expert-streaming (Part 2 port): close an in-progress load progress line
 * before printing a streaming-cache warning. Ported from upstream antirez/ds4. */
static void cuda_model_load_progress_finish(void) {
    if (!g_model_load_progress_started) return;
    if (g_model_load_progress_tty) {
        fputc('\n', stderr);
        fflush(stderr);
    }
    g_model_load_progress_started = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\rds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_PREFETCH") != NULL ||
        getenv("DS4_CUDA_COPY_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (getenv("DS4_CUDA_MODEL_PREFETCH_SYNC") != NULL) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = getenv("DS4_CUDA_MODEL_COPY_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || !model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || getenv("DS4_CUDA_KEEP_MODEL_PAGES") != NULL || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA direct model read disabled: %s\n", strerror(direct_errno));
                }
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

/* =========================================================================
 * SSD expert-streaming engine (Part 2 port from upstream antirez/ds4).
 * Ported verbatim from upstream ds4_cuda.cu 1411-2247, then transformed to
 * per-CUDA-device state (indexed by cudaGetDevice) for our PP/TP multi-GPU
 * path. g_ssd_streaming_mode + g_stream_expert_budget_override stay scalar.
 * The std::vector in cuda_stream_expert_cache is reset field-by-field on
 * release (NOT memset) to avoid clobbering the vector control block.
 * ========================================================================= */
static void cuda_stream_selected_cache_invalidate(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    g_stream_selected_cache[dev].valid = 0;
}

static void cuda_stream_selected_cache_release(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (g_stream_selected_cache[dev].gate_ptr) {
        (void)cudaFree(g_stream_selected_cache[dev].gate_ptr);
    }
    if (g_stream_selected_cache[dev].up_ptr) {
        (void)cudaFree(g_stream_selected_cache[dev].up_ptr);
    }
    if (g_stream_selected_cache[dev].down_ptr) {
        (void)cudaFree(g_stream_selected_cache[dev].down_ptr);
    }
    if (g_stream_selected_cache[dev].slot_selected_ptr) {
        (void)cudaFree(g_stream_selected_cache[dev].slot_selected_ptr);
    }
    memset(&g_stream_selected_cache[dev], 0, sizeof(g_stream_selected_cache[dev]));
}

static void cuda_stream_expert_cache_release_all(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (g_stream_expert_cache[dev].gate_ptr) {
        (void)cudaFree(g_stream_expert_cache[dev].gate_ptr);
    }
    if (g_stream_expert_cache[dev].up_ptr) {
        (void)cudaFree(g_stream_expert_cache[dev].up_ptr);
    }
    if (g_stream_expert_cache[dev].down_ptr) {
        (void)cudaFree(g_stream_expert_cache[dev].down_ptr);
    }
    g_stream_expert_cache[dev].slots.clear();
    g_stream_expert_cache[dev].valid = 0;
    g_stream_expert_cache[dev].capacity = 0;
    g_stream_expert_cache[dev].count = 0;
    g_stream_expert_cache[dev].tick = 0;
    g_stream_expert_cache[dev].gate_expert_bytes = 0;
    g_stream_expert_cache[dev].down_expert_bytes = 0;
    g_stream_expert_cache[dev].gate_ptr = NULL;
    g_stream_expert_cache[dev].up_ptr = NULL;
    g_stream_expert_cache[dev].down_ptr = NULL;
    g_stream_expert_cache[dev].gate_capacity = 0;
    g_stream_expert_cache[dev].up_capacity = 0;
    g_stream_expert_cache[dev].down_capacity = 0;
    /* GPU-gather: the LRU is gone, so drop its device directory too. */
    if (g_stream_expert_slot_dir[dev]) {
        (void)cudaFree(g_stream_expert_slot_dir[dev]);
        g_stream_expert_slot_dir[dev] = NULL;
        g_stream_dir_experts[dev] = 0;
    }
}

static void cuda_stream_expert_cache_invalidate(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    for (cuda_stream_expert_cache_slot &slot : g_stream_expert_cache[dev].slots) {
        slot.valid = 0;
    }
    /* GPU-gather: every slot just became invalid -> clear the device directory. */
    if (g_stream_expert_slot_dir[dev]) {
        (void)cudaMemset(g_stream_expert_slot_dir[dev], 0xFF,
                         (size_t)DS4_STREAM_GATHER_MAX_LAYERS *
                             g_stream_dir_experts[dev] * sizeof(int32_t));
    }
    g_stream_expert_cache[dev].valid = 0;
    g_stream_expert_cache[dev].count = 0;
    g_stream_expert_cache[dev].tick = 0;
}

static uint32_t cuda_stream_expert_cache_requested_budget(void) {
    uint32_t cap = g_stream_expert_budget_override != 0 ?
        g_stream_expert_budget_override : DS4_CUDA_STREAM_EXPERT_DEFAULT;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(env, &end, 10);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0') {
            cap = v > DS4_CUDA_STREAM_EXPERT_MAX ?
                DS4_CUDA_STREAM_EXPERT_MAX : (uint32_t)v;
        }
    }
    if (cap > DS4_CUDA_STREAM_EXPERT_MAX) {
        cap = DS4_CUDA_STREAM_EXPERT_MAX;
    }
    return cap;
}

static uint32_t cuda_stream_expert_cache_configured_budget(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    uint32_t cap = cuda_stream_expert_cache_requested_budget();
    if (g_stream_expert_runtime_cap[dev] != 0 && cap > g_stream_expert_runtime_cap[dev]) {
        cap = g_stream_expert_runtime_cap[dev];
    }
    return cap;
}

static int cuda_stream_expert_cache_budget_visible_to_shared(void) {
    if (!g_ssd_streaming_mode) return 0;
    if (g_stream_expert_budget_override != 0) return 1;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) return 1;
    env = getenv("DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST");
    if (!env || !env[0]) {
        env = getenv("DS4_CUDA_STREAMING_EXPERT_HOTLIST");
    }
    return env && env[0] && strcmp(env, "0") != 0;
}

static uint64_t cuda_stream_expert_cache_reserve_bytes(void) {
    uint64_t gb = 16;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB");
    if (env && env[0]) {
        char *end = NULL;
        errno = 0;
        unsigned long long v = strtoull(env, &end, 10);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0') {
            gb = (uint64_t)v;
        }
    }
    if (gb > UINT64_MAX / 1073741824ull) return UINT64_MAX;
    return gb * 1073741824ull;
}

static uint32_t cuda_stream_expert_cache_live_budget(
        uint32_t requested,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint64_t reclaim_bytes,
        int report) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (requested == 0 ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    const uint64_t per_expert_bytes =
        gate_expert_bytes * 2ull + down_expert_bytes;
    if (per_expert_bytes == 0) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming expert cache memory query failed: %s; "
                "using direct selected loads\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    uint64_t free_bytes = (uint64_t)free_b;
    if (reclaim_bytes > UINT64_MAX - free_bytes) {
        free_bytes = UINT64_MAX;
    } else {
        free_bytes += reclaim_bytes;
    }
    uint64_t reserve = cuda_stream_expert_cache_reserve_bytes();
    const uint64_t total_bytes = (uint64_t)total_b;
    if (total_bytes != 0 && reserve > total_bytes / 2ull) {
        reserve = total_bytes / 2ull;
    }
    if (free_bytes <= reserve) {
        if (report && g_stream_expert_memory_cap_notice[dev] != requested) {
            cuda_model_load_progress_finish();
            fprintf(stderr,
                    "ds4: CUDA streaming expert cache disabled: available %.2f GiB <= reserve %.2f GiB\n",
                    (double)free_bytes / 1073741824.0,
                    (double)reserve / 1073741824.0);
            g_stream_expert_memory_cap_notice[dev] = requested;
        }
        return 0;
    }

    uint64_t usable = free_bytes - reserve;
    uint64_t max_slots64 = usable / per_expert_bytes;
    if (max_slots64 > UINT32_MAX) max_slots64 = UINT32_MAX;
    uint32_t capped = requested;
    if ((uint64_t)capped > max_slots64) capped = (uint32_t)max_slots64;
    if (report && capped != requested && g_stream_expert_memory_cap_notice[dev] != capped) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming expert cache capped from %u to %u experts "
                "(available %.2f GiB, reserve %.2f GiB, %.2f MiB/expert)\n",
                requested,
                capped,
                (double)free_bytes / 1073741824.0,
                (double)reserve / 1073741824.0,
                (double)per_expert_bytes / 1048576.0);
        g_stream_expert_memory_cap_notice[dev] = capped;
    }
    return capped;
}

static uint64_t cuda_stream_expert_cache_expert_bytes(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    return gate_expert_bytes * 2ull + down_expert_bytes;
}

static void cuda_stream_expert_cache_note_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (g_stream_expert_runtime_gate_bytes[dev] == gate_expert_bytes &&
        g_stream_expert_runtime_down_bytes[dev] == down_expert_bytes) {
        return;
    }
    g_stream_expert_runtime_gate_bytes[dev] = gate_expert_bytes;
    g_stream_expert_runtime_down_bytes[dev] = down_expert_bytes;
    g_stream_expert_runtime_cap[dev] = 0;
    g_stream_expert_memory_cap_notice[dev] = 0;
}

static uint32_t cuda_stream_expert_cache_shrunken_cap(uint32_t cap) {
    if (cap == 0) return 0;
    const uint32_t release = (cap + 9u) / 10u;
    return cap > release ? cap - release : 0;
}

static void cuda_stream_expert_cache_note_oom_cap(
        uint32_t failed_cap,
        uint32_t new_cap,
        uint64_t expert_bytes,
        const char *errstr) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (g_stream_expert_runtime_cap[dev] != 0 &&
        g_stream_expert_runtime_cap[dev] <= new_cap) {
        return;
    }
    g_stream_expert_runtime_cap[dev] = new_cap;
    const uint32_t released =
        failed_cap > new_cap ? failed_cap - new_cap : 0;
    cuda_model_load_progress_finish();
    fprintf(stderr,
            "ds4: CUDA streaming expert cache allocation failed at %u experts "
            "/ %.2f GiB%s%s\n",
            failed_cap,
            expert_bytes != 0 ?
                (double)((uint64_t)failed_cap * expert_bytes) / 1073741824.0 :
                0.0,
            errstr && errstr[0] ? ": " : "",
            errstr && errstr[0] ? errstr : "");
    if (new_cap != 0) {
        fprintf(stderr,
                "ds4:   shrinking resident cache margin by %u experts / %.2f GiB; "
                "runtime cache cap now %u experts\n",
                released,
                expert_bytes != 0 ?
                    (double)((uint64_t)released * expert_bytes) / 1073741824.0 :
                    0.0,
                new_cap);
    } else {
        fprintf(stderr,
                "ds4:   disabling resident expert cache after OOM; using direct selected loads\n");
    }
}

static int cuda_stream_expert_cache_try_alloc(
        uint32_t cap,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        char **gate_ptr,
        char **up_ptr,
        char **down_ptr,
        const char **errstr) {
    *gate_ptr = NULL;
    *up_ptr = NULL;
    *down_ptr = NULL;
    if (errstr) *errstr = NULL;
    if (cap == 0 ||
        (uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
        return 0;
    }
    const uint64_t gate_bytes = (uint64_t)cap * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)cap * down_expert_bytes;

    void *gate = NULL;
    void *up = NULL;
    void *down = NULL;
    cudaError_t err = cudaMalloc(&gate, (size_t)gate_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&up, (size_t)gate_bytes);
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&down, (size_t)down_bytes);
    if (err != cudaSuccess) goto fail;

    *gate_ptr = (char *)gate;
    *up_ptr = (char *)up;
    *down_ptr = (char *)down;
    return 1;

fail:
    if (errstr) *errstr = cudaGetErrorString(err);
    (void)cudaGetLastError();
    if (gate) (void)cudaFree(gate);
    if (up) (void)cudaFree(up);
    if (down) (void)cudaFree(down);
    return 0;
}

static void cuda_stream_selected_stage_release(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[dev][i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[dev][i]);
            g_stream_selected_stage_event[dev][i] = NULL;
        }
        if (g_stream_selected_stage_raw[dev][i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[dev][i]);
            g_stream_selected_stage_raw[dev][i] = NULL;
            g_stream_selected_stage[dev][i] = NULL;
        }
    }
    g_stream_selected_stage_bytes[dev] = 0;
    if (g_stream_selected_upload_stream[dev]) {
        (void)cudaStreamDestroy(g_stream_selected_upload_stream[dev]);
        g_stream_selected_upload_stream[dev] = NULL;
    }
}

static int cuda_stream_selected_stage_pool_alloc(uint64_t bytes) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (g_stream_selected_stage_bytes[dev] >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_stream_selected_stage_event[dev][i]) {
            (void)cudaEventDestroy(g_stream_selected_stage_event[dev][i]);
            g_stream_selected_stage_event[dev][i] = NULL;
        }
        if (g_stream_selected_stage_raw[dev][i]) {
            (void)cudaFreeHost(g_stream_selected_stage_raw[dev][i]);
            g_stream_selected_stage_raw[dev][i] = NULL;
            g_stream_selected_stage[dev][i] = NULL;
        }
    }
    g_stream_selected_stage_bytes[dev] = 0;
    if (!g_stream_selected_upload_stream[dev]) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_stream_selected_upload_stream[dev],
                                                    cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected upload stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_stream_selected_stage_raw[dev][i],
                                         (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging allocation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_stream_selected_stage[dev][i] =
            cuda_align_ptr(g_stream_selected_stage_raw[dev][i],
                           g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_stream_selected_stage_event[dev][i],
                                       cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_stream_selected_stage_bytes[dev] = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_bytes(
        char **ptr,
        uint64_t *capacity,
        uint64_t bytes,
        const char *what) {
    if (bytes == 0) return 1;
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected cache allocation failed for %s (%.2f MiB): %s\n",
                what ? what : "experts",
                (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (char *)dev;
    *capacity = bytes;
    return 1;
}

static int cuda_stream_selected_ensure_i32(
        int32_t **ptr,
        uint64_t *capacity,
        uint64_t count,
        const char *what) {
    if (count == 0 || count > UINT64_MAX / sizeof(int32_t)) return 0;
    const uint64_t bytes = count * sizeof(int32_t);
    if (*ptr && *capacity >= bytes) return 1;
    if (*ptr) {
        (void)cudaFree(*ptr);
        *ptr = NULL;
        *capacity = 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected cache allocation failed for %s (%u entries): %s\n",
                what ? what : "selected slots",
                (unsigned)count,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    *ptr = (int32_t *)dev;
    *capacity = bytes;
    return 1;
}

static cuda_stream_expert_cache *cuda_stream_expert_cache_prepare(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t target_cap) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    const uint64_t expert_bytes =
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes);
    if (expert_bytes == 0) return NULL;
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);

    const uint32_t requested_cap = cuda_stream_expert_cache_configured_budget();
    if (requested_cap == 0) return NULL;
    if (target_cap == 0 || target_cap > requested_cap) target_cap = requested_cap;
    if (target_cap == 0) return NULL;
    const int same_dims =
        g_stream_expert_cache[dev].valid &&
        g_stream_expert_cache[dev].gate_expert_bytes == gate_expert_bytes &&
        g_stream_expert_cache[dev].down_expert_bytes == down_expert_bytes;
    if (!same_dims && g_stream_expert_cache[dev].valid) {
        cuda_stream_expert_cache_release_all();
    }
    if (same_dims &&
        g_stream_expert_cache[dev].capacity != 0 &&
        g_stream_expert_cache[dev].capacity >= target_cap &&
        g_stream_expert_cache[dev].slots.size() == g_stream_expert_cache[dev].capacity) {
        return &g_stream_expert_cache[dev];
    }

    uint64_t reclaim_bytes = 0;
    if (same_dims &&
        g_stream_expert_cache[dev].capacity != 0 &&
        (uint64_t)g_stream_expert_cache[dev].capacity <= UINT64_MAX / expert_bytes) {
        reclaim_bytes = (uint64_t)g_stream_expert_cache[dev].capacity * expert_bytes;
    }
    uint32_t cap =
        cuda_stream_expert_cache_live_budget(target_cap,
                                             gate_expert_bytes,
                                             down_expert_bytes,
                                             reclaim_bytes,
                                             reclaim_bytes == 0);
    if (cap == 0) return NULL;
    if (same_dims &&
        g_stream_expert_cache[dev].capacity != 0 &&
        g_stream_expert_cache[dev].capacity >= cap &&
        g_stream_expert_cache[dev].slots.size() == g_stream_expert_cache[dev].capacity) {
        return &g_stream_expert_cache[dev];
    }

    cuda_stream_expert_cache_release_all();
    while (cap != 0) {
        if ((uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
            (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: CUDA streaming expert cache size overflow\n");
            return NULL;
        }

        char *gate_ptr = NULL;
        char *up_ptr = NULL;
        char *down_ptr = NULL;
        const char *alloc_error = NULL;
        if (!cuda_stream_expert_cache_try_alloc(cap,
                                                gate_expert_bytes,
                                                down_expert_bytes,
                                                &gate_ptr,
                                                &up_ptr,
                                                &down_ptr,
                                                &alloc_error)) {
            const uint32_t new_cap =
                cuda_stream_expert_cache_shrunken_cap(cap);
            cuda_stream_expert_cache_note_oom_cap(cap,
                                                  new_cap,
                                                  expert_bytes,
                                                  alloc_error);
            cap = new_cap;
            if (cap != 0) {
                cap = cuda_stream_expert_cache_live_budget(cap,
                                                           gate_expert_bytes,
                                                           down_expert_bytes,
                                                           0,
                                                           1);
            }
            continue;
        }

        try {
            g_stream_expert_cache[dev].slots.resize(cap);
        } catch (...) {
            fprintf(stderr, "ds4: CUDA streaming expert cache metadata allocation failed\n");
            (void)cudaFree(gate_ptr);
            (void)cudaFree(up_ptr);
            (void)cudaFree(down_ptr);
            cuda_stream_expert_cache_release_all();
            return NULL;
        }

        g_stream_expert_cache[dev].valid = 1;
        g_stream_expert_cache[dev].capacity = cap;
        g_stream_expert_cache[dev].count = 0;
        g_stream_expert_cache[dev].tick = 0;
        g_stream_expert_cache[dev].gate_expert_bytes = gate_expert_bytes;
        g_stream_expert_cache[dev].down_expert_bytes = down_expert_bytes;
        g_stream_expert_cache[dev].gate_ptr = gate_ptr;
        g_stream_expert_cache[dev].up_ptr = up_ptr;
        g_stream_expert_cache[dev].down_ptr = down_ptr;
        g_stream_expert_cache[dev].gate_capacity =
            (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache[dev].up_capacity =
            (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache[dev].down_capacity =
            (uint64_t)cap * down_expert_bytes;
        return &g_stream_expert_cache[dev];
    }
    return NULL;
}

static int cuda_stream_expert_cache_find(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cache || !cache->valid) return -1;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        const cuda_stream_expert_cache_slot &slot = cache->slots[i];
        if (slot.valid &&
            slot.model_map == model_map &&
            slot.model_size == model_size &&
            slot.layer == layer &&
            slot.n_total_expert == n_total_expert &&
            slot.expert == expert &&
            slot.gate_offset == gate_offset &&
            slot.up_offset == up_offset &&
            slot.down_offset == down_offset &&
            slot.gate_expert_bytes == gate_expert_bytes &&
            slot.down_expert_bytes == down_expert_bytes) {
            return (int)i;
        }
    }
    return -1;
}

static uint32_t cuda_stream_expert_cache_lru_slot(
        cuda_stream_expert_cache *cache) {
    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (!cache->slots[i].valid) return i;
    }
    uint32_t slot = 0;
    uint64_t best_age = cache->slots[0].age;
    for (uint32_t i = 1; i < cache->capacity; i++) {
        if (cache->slots[i].age < best_age) {
            best_age = cache->slots[i].age;
            slot = i;
        }
    }
    return slot;
}

static int cuda_stream_expert_cache_copy_to_compact(
        cuda_stream_expert_cache *cache,
        uint32_t cache_slot,
        uint32_t compact_slot,
        char *compact_gate,
        char *compact_up,
        char *compact_down) {
    const uint64_t gate_src = (uint64_t)cache_slot * cache->gate_expert_bytes;
    const uint64_t down_src = (uint64_t)cache_slot * cache->down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)compact_slot * cache->gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)compact_slot * cache->down_expert_bytes;
    return cuda_ok(cudaMemcpy(compact_gate + gate_dst,
                              cache->gate_ptr + gate_src,
                              (size_t)cache->gate_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected gate cache copy") &&
           cuda_ok(cudaMemcpy(compact_up + gate_dst,
                              cache->up_ptr + gate_src,
                              (size_t)cache->gate_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected up cache copy") &&
           cuda_ok(cudaMemcpy(compact_down + down_dst,
                              cache->down_ptr + down_src,
                              (size_t)cache->down_expert_bytes,
                              cudaMemcpyDeviceToDevice),
                   "streaming selected down cache copy");
}

static int cuda_stream_gpu_gather_enabled(void) {
    if (g_stream_gpu_gather < 0)
        g_stream_gpu_gather = (getenv("DS4_STREAM_GPU_GATHER") != NULL) ? 1 : 0;
    return g_stream_gpu_gather;
}

/* dense layer_id for a gate_offset (unique per layer); assigns on first sight. */
static uint32_t cuda_stream_layerid(uint64_t gate_offset) {
    auto it = g_stream_gate_off_layerid.find(gate_offset);
    if (it != g_stream_gate_off_layerid.end()) return it->second;
    uint32_t id = (uint32_t)g_stream_gate_off_layerid.size();
    if (id >= DS4_STREAM_GATHER_MAX_LAYERS) id = DS4_STREAM_GATHER_MAX_LAYERS - 1u;
    g_stream_gate_off_layerid[gate_offset] = id;
    return id;
}

/* ensure the per-device slot directory exists for n_total_expert; init all -1. */
static int cuda_stream_dir_ensure(int dev, uint32_t n_total_expert) {
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES || n_total_expert == 0) return 0;
    if (g_stream_expert_slot_dir[dev] && g_stream_dir_experts[dev] == n_total_expert) return 1;
    if (g_stream_expert_slot_dir[dev]) {
        (void)cudaFree(g_stream_expert_slot_dir[dev]);
        g_stream_expert_slot_dir[dev] = NULL;
    }
    const size_t n = (size_t)DS4_STREAM_GATHER_MAX_LAYERS * n_total_expert;
    if (!cuda_ok(cudaMalloc(&g_stream_expert_slot_dir[dev], n * sizeof(int32_t)),
                 "stream gather dir alloc")) {
        g_stream_expert_slot_dir[dev] = NULL;
        return 0;
    }
    (void)cudaMemset(g_stream_expert_slot_dir[dev], 0xFF, n * sizeof(int32_t)); /* -1 */
    g_stream_dir_experts[dev] = n_total_expert;
    return 1;
}

/* one-thread write: clear an old (layer,expert) entry and set the new one.
 * Indices are passed by value so there is no host-stack lifetime hazard. */
__global__ static void cuda_stream_dir_write_kernel(int32_t *dir,
                                                    int has_clear, uint32_t clear_idx,
                                                    uint32_t set_idx, int32_t set_val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (has_clear) dir[clear_idx] = -1;
        dir[set_idx] = set_val;
    }
}

/* mirror a load_slot into the device directory: clear the slot's previous
 * (gate_offset,expert) key if it held one, then point (new gate_offset,expert)
 * at this slot. Runs on the decode stream so it orders before the resolve. */
static void cuda_stream_dir_update_load(int dev,
                                        int old_valid, uint64_t old_gate_offset, uint32_t old_expert,
                                        uint64_t new_gate_offset, uint32_t new_expert,
                                        uint32_t slot) {
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES || !g_stream_expert_slot_dir[dev]) return;
    const uint32_t ne = g_stream_dir_experts[dev];
    if (new_expert >= ne) return;
    const uint32_t new_idx =
        cuda_stream_layerid(new_gate_offset) * ne + new_expert;
    int has_clear = 0;
    uint32_t clear_idx = 0;
    if (old_valid && old_expert < ne) {
        has_clear = 1;
        clear_idx = cuda_stream_layerid(old_gate_offset) * ne + old_expert;
    }
    cuda_stream_dir_write_kernel<<<1, 1, 0, ds4_decode_stream()>>>(
        g_stream_expert_slot_dir[dev], has_clear, clear_idx, new_idx, (int32_t)slot);
    (void)cudaGetLastError();
}

static int cuda_stream_expert_cache_load_slot(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t slot,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    const uint64_t gate_src =
        gate_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t up_src =
        up_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t down_src =
        down_offset + (uint64_t)expert * down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)slot * gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)slot * down_expert_bytes;
    if (!cuda_model_copy_to_device_streamed(cache->gate_ptr + gate_dst,
                                            model_map,
                                            model_size,
                                            gate_src,
                                            gate_expert_bytes,
                                            "cached moe_gate") ||
        !cuda_model_copy_to_device_streamed(cache->up_ptr + gate_dst,
                                            model_map,
                                            model_size,
                                            up_src,
                                            gate_expert_bytes,
                                            "cached moe_up") ||
        !cuda_model_copy_to_device_streamed(cache->down_ptr + down_dst,
                                            model_map,
                                            model_size,
                                            down_src,
                                            down_expert_bytes,
                                            "cached moe_down")) {
        return 0;
    }
    cuda_stream_expert_cache_slot &entry = cache->slots[slot];
    /* GPU-gather directory: capture the entry this slot previously held so we
     * can clear its key before re-pointing it at the new (gate_offset,expert). */
    const int      old_valid       = entry.valid;
    const uint64_t old_gate_offset = entry.gate_offset;
    const uint32_t old_expert      = entry.expert;
    entry.valid = 1;
    entry.model_map = model_map;
    entry.model_size = model_size;
    entry.layer = layer;
    entry.n_total_expert = n_total_expert;
    entry.expert = expert;
    entry.gate_offset = gate_offset;
    entry.up_offset = up_offset;
    entry.down_offset = down_offset;
    entry.gate_expert_bytes = gate_expert_bytes;
    entry.down_expert_bytes = down_expert_bytes;
    entry.age = ++cache->tick;
    if (cuda_stream_gpu_gather_enabled()) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
        cuda_stream_dir_ensure(dev, n_total_expert);
        cuda_stream_dir_update_load(dev, old_valid, old_gate_offset, old_expert,
                                    gate_offset, expert, slot);
    }
    return 1;
}

static int cuda_stream_expert_cache_seed_one(
        cuda_stream_expert_cache *cache,
        const void *model_map,
        uint64_t model_size,
        uint32_t layer,
        uint32_t n_total_expert,
        uint32_t expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    int cache_slot = cuda_stream_expert_cache_find(cache,
                                                   model_map,
                                                   model_size,
                                                   layer,
                                                   n_total_expert,
                                                   expert,
                                                   gate_offset,
                                                   up_offset,
                                                   down_offset,
                                                   gate_expert_bytes,
                                                   down_expert_bytes);
    if (cache_slot >= 0) {
        cache->slots[(uint32_t)cache_slot].age = ++cache->tick;
        return 1;
    }

    const uint32_t load_slot = cuda_stream_expert_cache_lru_slot(cache);
    const int append = !cache->slots[load_slot].valid;
    if (!cuda_stream_expert_cache_load_slot(cache,
                                            model_map,
                                            model_size,
                                            load_slot,
                                            layer,
                                            n_total_expert,
                                            expert,
                                            gate_offset,
                                            up_offset,
                                            down_offset,
                                            gate_expert_bytes,
                                            down_expert_bytes)) {
        return 0;
    }
    if (append && cache->count < cache->capacity) cache->count++;
    return 1;
}

static int cuda_stream_layer_expert_ranges_valid(
        uint64_t model_size,
        uint32_t n_total_expert,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        const char *what) {
    if (n_total_expert == 0 ||
        gate_expert_bytes == 0 ||
        down_expert_bytes == 0 ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr,
                "ds4: CUDA streaming %s expert size overflow\n",
                what ? what : "selected");
        return 0;
    }
    const uint64_t full_gate_bytes =
        (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes =
        (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size ||
        down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr,
                "ds4: CUDA streaming %s expert range outside model map\n",
                what ? what : "selected");
        return 0;
    }
    return 1;
}

static int cuda_model_copy_to_device_streamed(
        char *dst,
        const void *model_map,
        uint64_t model_size,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    if (!dst || !model_map || offset > model_size || bytes > model_size - offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    /* DEFAULT: copy the expert directly from the (page-cache-resident) model
     * mmap. With DS4_CUDA_KEEP_MODEL_PAGES=1 the whole GGUF lives in the 1024GB
     * host RAM page cache, so this reads expert bytes from RAM at full PCIe
     * speed and is byte-identical to the proven compact path. The streaming win
     * (only the routed experts land in VRAM, not all 256/layer) is fully
     * realized here without any pread.
     *
     * The pread/staging path below is for TRUE out-of-core streaming (model >
     * host RAM); it is opt-in via DS4_CUDA_STREAM_USE_PREAD because it currently
     * reads the wrong bytes for this fork's GGUF mmap layout (the proven-correct
     * mmap copy above produces coherent output; the pread path produces garbage)
     * — TODO root-cause the file-offset/fd mismatch before relying on it. For
     * the Pro 1.6T target (432GB < 1024GB RAM) the model is always resident, so
     * the mmap path is both correct and optimal. */
    if (g_model_fd < 0 ||
        getenv("DS4_CUDA_STREAM_USE_PREAD") == NULL ||
        (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base)) {
        return cuda_ok(cudaMemcpy(dst,
                                  (const char *)model_map + offset,
                                  (size_t)bytes,
                                  cudaMemcpyHostToDevice),
                       what ? what : "stream selected expert copy");
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_stream_selected_stage_pool_alloc(stage_bytes)) return 0;

    cudaError_t err = cudaSuccess;
    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_stream_selected_stage_event[dev][bi]);
            if (err != cudaSuccess) {
                fprintf(stderr,
                        "ds4: CUDA streaming selected staging wait failed for %s: %s\n",
                        what ? what : "expert",
                        cudaGetErrorString(err));
                (void)cudaGetLastError();
                return 0;
            }
        }

        const char *payload = NULL;
        if (!cuda_model_stage_read(g_stream_selected_stage[dev][bi],
                                   g_stream_selected_stage_bytes[dev],
                                   offset + copied,
                                   n,
                                   &payload)) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected read failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return 0;
        }
        err = cudaMemcpyAsync(dst + copied,
                              payload,
                              (size_t)n,
                              cudaMemcpyHostToDevice,
                              g_stream_selected_upload_stream[dev]);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "expert",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        err = cudaEventRecord(g_stream_selected_stage_event[dev][bi],
                              g_stream_selected_upload_stream[dev]);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected staging record failed for %s: %s\n",
                    what ? what : "expert",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, model_size, offset + copied, n);
        copied += n;
        chunk_idx++;
    }

    err = cudaStreamSynchronize(g_stream_selected_upload_stream[dev]);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "ds4: CUDA streaming selected upload sync failed for %s: %s\n",
                what ? what : "expert",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = getenv("DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
    }
    if (gb == 0) return UINT64_MAX;
    return gb * 1073741824ull;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = getenv("DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_direct_host_access_allowed()
            ? cuda_model_ptr(model_map, offset)
            : NULL;
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (getenv("DS4_CUDA_STRICT_WEIGHT_CACHE") != NULL) return NULL;
        return cuda_model_direct_host_access_allowed()
            ? cuda_model_ptr(model_map, offset)
            : NULL;
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1, -1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (getenv("DS4_CUDA_NO_MODEL_COPY") != NULL ||
        getenv("DS4_CUDA_DIRECT_MODEL") != NULL ||
        getenv("DS4_CUDA_WEIGHT_CACHE") != NULL ||
        getenv("DS4_CUDA_WEIGHT_PRELOAD") != NULL) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "ds4: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (getenv("DS4_CUDA_MODEL_COPY_VERBOSE") != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

static int cuda_decode_graph_enabled(void) {
    const char *env = getenv("DS4_CUDA_DECODE_GRAPH");
    return env && env[0] && env[0] != '0';
}

static int cublas_bind_decode_stream(void) {
    if (!g_cublas) return 1;
    if (!g_cuda_decode_stream_created) return 1;
    if (!cublas_ok(cublasSetStream(g_cublas, g_cuda_decode_stream),
                   "set decode stream")) {
        return 0;
    }
    if (g_cublas_workspace && g_cublas_workspace_bytes != 0 &&
        !cublas_ok(cublasSetWorkspace(g_cublas,
                                      g_cublas_workspace,
                                      (size_t)g_cublas_workspace_bytes),
                   "set decode workspace")) {
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);

        /* Phase 1: non-default graph stream + cuBLAS workspace */
        if (cuda_decode_graph_enabled()) {
            cudaError_t ce = cudaStreamCreateWithFlags(&g_cuda_decode_stream,
                                                       cudaStreamNonBlocking);
            if (ce != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA graph stream creation failed: %s\n",
                        cudaGetErrorString(ce));
                (void)cudaGetLastError();
                g_cuda_decode_stream = 0;
                (void)cublasDestroy(g_cublas);
                g_cublas = NULL;
                return 0;
            } else {
                g_cuda_decode_stream_created = 1;
                g_cublas_workspace_bytes = 32ull * 1024ull * 1024ull;
                ce = cudaMalloc(&g_cublas_workspace, (size_t)g_cublas_workspace_bytes);
                if (ce != cudaSuccess) {
                    fprintf(stderr, "ds4: CUDA graph workspace alloc failed: %s\n",
                            cudaGetErrorString(ce));
                    (void)cudaGetLastError();
                    (void)cudaStreamDestroy(g_cuda_decode_stream);
                    g_cuda_decode_stream = 0;
                    g_cuda_decode_stream_created = 0;
                    g_cublas_workspace = NULL;
                    g_cublas_workspace_bytes = 0;
                    (void)cublasDestroy(g_cublas);
                    g_cublas = NULL;
                    return 0;
                }
                if (!cublas_bind_decode_stream()) {
                    (void)cudaFree(g_cublas_workspace);
                    g_cublas_workspace = NULL;
                    g_cublas_workspace_bytes = 0;
                    (void)cudaStreamDestroy(g_cuda_decode_stream);
                    g_cuda_decode_stream = 0;
                    g_cuda_decode_stream_created = 0;
                    (void)cublasDestroy(g_cublas);
                    g_cublas = NULL;
                    return 0;
                }
                fprintf(stderr, "ds4: CUDA graph decode stream ready\n");
            }
        }

        g_cublas_ready = 1;
    }

    (void)ds4_gpu_pp_init();

    return 1;
}

void ds4_gpu_init_moe_scratch(uint64_t gate_bytes, uint64_t up_bytes, uint64_t down_bytes) {
    if (g_moe_scratch_inited) return;
    const char *scratch_env = getenv("DS4_CUDA_MOE_PERSISTENT_SCRATCH");
    if (!scratch_env || !scratch_env[0] || scratch_env[0] == '0') return;
    /* Limit to available VRAM: skip if total > 6 GiB.
     * Modern GPUs have 16+ GB; 3.5 GiB scratch for Q4K MTP is acceptable. */
    if (gate_bytes + up_bytes + down_bytes > 6ull * 1073741824ull) {
        fprintf(stderr, "ds4: MoE scratch too large (%.2f GiB), skipping\n",
                (double)(gate_bytes + up_bytes + down_bytes) / 1073741824.0);
        return;
    }
    cudaError_t ce;
    ce = cudaMalloc(&g_moe_scratch_gate, (size_t)gate_bytes);
    if (ce != cudaSuccess) { fprintf(stderr, "ds4: MoE scratch gate alloc failed\n"); (void)cudaGetLastError(); return; }
    ce = cudaMalloc(&g_moe_scratch_up, (size_t)up_bytes);
    if (ce != cudaSuccess) { fprintf(stderr, "ds4: MoE scratch up alloc failed\n"); (void)cudaFree(g_moe_scratch_gate); g_moe_scratch_gate = NULL; (void)cudaGetLastError(); return; }
    ce = cudaMalloc(&g_moe_scratch_down, (size_t)down_bytes);
    if (ce != cudaSuccess) { fprintf(stderr, "ds4: MoE scratch down alloc failed\n"); (void)cudaFree(g_moe_scratch_gate); (void)cudaFree(g_moe_scratch_up); g_moe_scratch_gate = g_moe_scratch_up = NULL; (void)cudaGetLastError(); return; }
    g_moe_scratch_gate_bytes = gate_bytes;
    g_moe_scratch_up_bytes = up_bytes;
    g_moe_scratch_down_bytes = down_bytes;
    g_moe_scratch_inited = 1;
    fprintf(stderr, "ds4: MoE persistent scratch allocated (gate=%.2f up=%.2f down=%.2f MiB)\n",
            (double)gate_bytes / 1048576.0, (double)up_bytes / 1048576.0, (double)down_bytes / 1048576.0);
}

static int ds4_gpu_pp_init(void) {
    const char *pp_env = getenv("DS4_CUDA_PP");
    if (!pp_env || !pp_env[0] || pp_env[0] == '0') return 1;
    g_pp_requested = 1;
    int ngpu = 0;
    if (cudaGetDeviceCount(&ngpu) != cudaSuccess || ngpu < 2) return 0;
    g_pp_ngpu = ngpu;

    int ok = 1;
    for (int i = 0; i < ngpu; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < ngpu; j++) {
            if (i == j) continue;
            int can = 0;
            cudaError_t ce = cudaDeviceCanAccessPeer(&can, i, j);
            if (ce != cudaSuccess || !can) { ok = 0; continue; }
            /* NOTE: cudaDeviceEnablePeerAccess BROKEN on Blackwell CUDA 13.2.
             * cudaMemcpyPeer works automatically without explicit enable. */
            (void)ce;
        }
    }
    if (!ok) { g_pp_ngpu = 0; return 0; }

    /* Tensor parallelism degree: g_tp ranks cooperate per layer. The model is
     * split into nstage = ngpu/g_tp pipeline stages; within a stage every rank
     * holds the SAME layer range (its own weight shards). g_tp=1 => pure PP. */
    int g_tp = 1;
    {
        const char *tp_env = getenv("DS4_CUDA_TP");
        if (tp_env && tp_env[0]) {
            int v = (int)strtol(tp_env, NULL, 10);
            if (v >= 2 && (ngpu % v) == 0) {
                g_tp = v;
            } else if (v >= 2) {
                fprintf(stderr, "ds4: DS4_CUDA_TP=%d but ngpu=%d not divisible; TP disabled\n", v, ngpu);
            }
        }
    }
    g_tp_degree = g_tp;
    const int nstage = ngpu / g_tp;

    int total_layers = g_model_n_layer;
    const char *split_env = getenv("DS4_CUDA_PP_LAYER_SPLIT");
    int stage_start[16] = {0}, stage_end[16] = {0};
    int have_split = 0;
    if (split_env && split_env[0]) {
        /* Custom split: comma-separated start layers, one per STAGE (== per GPU
         * when g_tp=1). e.g. TP2xPP3 6-GPU: "0,15,29" (3 stage starts). */
        int custom_start[16] = {0};
        int n_parsed = 0;
        const char *p = split_env;
        while (*p && n_parsed < nstage) {
            custom_start[n_parsed] = (int)strtol(p, NULL, 10);
            n_parsed++;
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        if (n_parsed == nstage) {
            for (int s = 0; s < nstage; s++) {
                stage_start[s] = custom_start[s];
                stage_end[s] = (s + 1 < nstage) ? custom_start[s + 1] - 1 : total_layers - 1;
            }
            have_split = 1;
            fprintf(stderr, "ds4: PP%s layer split (%d stages): %s\n",
                    g_tp > 1 ? "/TP" : "", nstage, split_env);
        } else {
            fprintf(stderr, "ds4: PP_LAYER_SPLIT expected %d values (stages), got %d; using default\n",
                    nstage, n_parsed);
        }
    }
    if (!have_split) {
        int per_stage = total_layers / nstage;
        int rem = total_layers % nstage;
        int il = 0;
        for (int s = 0; s < nstage; s++) {
            stage_start[s] = il;
            int n = per_stage + (s < rem ? 1 : 0);
            stage_end[s] = il + n - 1;
            il += n;
        }
    }
    for (int g = 0; g < ngpu; g++) {
        int s = g / g_tp;
        g_pp_layer_start[g] = stage_start[s];
        g_pp_layer_end[g] = stage_end[s];
    }
    for (int g = 0; g < ngpu; g++) {
        cudaSetDevice(g);
        if (cudaMalloc(&g_pp_active[g], 4 * g_model_n_embd * sizeof(float)) != cudaSuccess) {
            (void)cudaGetLastError(); ok = 0;
        }
    }
    g_pp_topology_ready = ok;
    if (ok) {
        for (int g = 0; g < ngpu; g++) {
            cudaSetDevice(g);
            if (!g_pp_stream[g]) {
                cudaError_t ce = cudaStreamCreateWithFlags(&g_pp_stream[g], cudaStreamNonBlocking);
                if (ce != cudaSuccess) { ok = 0; break; }
            }
            if (!g_pp_event[g]) {
                cudaError_t ce = cudaEventCreateWithFlags(&g_pp_event[g], cudaEventDisableTiming);
                if (ce != cudaSuccess) { ok = 0; break; }
            }
            if (!g_pp_copy_event[g]) {
                cudaError_t ce = cudaEventCreateWithFlags(&g_pp_copy_event[g], cudaEventDisableTiming);
                if (ce != cudaSuccess) { ok = 0; break; }
            }
        }
        cudaSetDevice(0);
    }
    g_pp_topology_ready = ok;
    if (ok && getenv("DS4_CUDA_PP_NCCL") != NULL) {
        ncclResult_t nr = ncclCommInitAll(g_pp_nccl_comms, ngpu, NULL);
        if (nr == ncclSuccess) {
            g_pp_nccl_ready = 1;
            fprintf(stderr, "ds4: NCCL PP communicator ready (%d GPUs)\n", ngpu);
        } else {
            fprintf(stderr, "ds4: ncclCommInitAll failed: %s\n", ncclGetErrorString(nr));
        }
        cudaSetDevice(0);
    }
    cudaSetDevice(0);
    if (ok) {
        const int delayed = getenv("DS4_CUDA_PP_DELAY_RESIDENT") != NULL;
        if (!delayed) {
            g_pp_resident_ready = 1;
            g_pp_decode_active = 1;
        }
        int n0 = g_pp_layer_end[0] - g_pp_layer_start[0] + 1;
        if (g_tp_degree > 1) {
            fprintf(stderr, "ds4: TP%dxPP%d ready (%d GPUs, %d stages, stage0/rank0: L%d-%d%s)\n",
                    g_tp_degree, nstage, ngpu, nstage, g_pp_layer_start[0], g_pp_layer_end[0],
                    delayed ? ", delayed resident" : "");
        } else {
            fprintf(stderr, "ds4: PP=%d ready (%d layers/GPU, GPU0: L%d-%d%s)\n",
                    ngpu, n0, g_pp_layer_start[0], g_pp_layer_end[0],
                    delayed ? ", delayed resident" : "");
        }
        if (getenv("DS4_CUDA_TP_SELFTEST") != NULL) (void)ds4_gpu_tp_selftest(ngpu);
    }
    return ok ? 1 : 0;
}

/* PP export API */
extern "C" void ds4_gpu_set_n_layer(uint32_t n) { if (n > 0u) g_model_n_layer = (int)n; }
extern "C" void ds4_gpu_set_n_embd(uint32_t n) { if (n > 0u) g_model_n_embd = (int)n; }
extern "C" int ds4_gpu_pp_set_device(int g) {
    cudaSetDevice(g); return 1;
}
extern "C" int ds4_gpu_pp_enabled(void) { return g_pp_decode_active; }
extern "C" int ds4_gpu_pp_requested(void) { return g_pp_requested; }
extern "C" int ds4_gpu_pp_resident_ready(void) { return g_pp_resident_ready; }
extern "C" int ds4_gpu_pp_enable_decode(void) {
    g_pp_resident_ready = 1;
    g_pp_decode_active = 1;
    return 1;
}
extern "C" int ds4_gpu_pp_work_streams_enable(int enable) {
    g_pp_work_streams_enabled = enable ? 1 : 0;
    return 1;
}
extern "C" int ds4_gpu_pp_ngpu(void) { return g_pp_ngpu; }
extern "C" int ds4_gpu_pp_layer_start(int g) { return (g >= 0 && g < g_pp_ngpu) ? g_pp_layer_start[g] : -1; }
extern "C" int ds4_gpu_pp_layer_end(int g) { return (g >= 0 && g < g_pp_ngpu) ? g_pp_layer_end[g] : -1; }

/* Tensor-Parallel topology accessors. g_tp_degree<=1 => TP off (rank 0, one
 * rank per stage, stage == logical GPU). */
extern "C" int ds4_gpu_tp_enabled(void) { return g_tp_degree > 1; }
extern "C" int ds4_gpu_tp_degree(void) { return g_tp_degree > 1 ? g_tp_degree : 1; }
extern "C" int ds4_gpu_tp_rank(int g) { return g_tp_degree > 1 ? (g % g_tp_degree) : 0; }
extern "C" int ds4_gpu_tp_stage(int g) { return g_tp_degree > 1 ? (g / g_tp_degree) : g; }
extern "C" int ds4_gpu_tp_nstage(void) { return g_tp_degree > 1 ? (g_pp_ngpu / g_tp_degree) : g_pp_ngpu; }
extern "C" int ds4_gpu_tp_stage_dev0(int s) { return g_tp_degree > 1 ? (s * g_tp_degree) : s; }
extern "C" int ds4_gpu_current_device(void) { int d = 0; (void)cudaGetDevice(&d); return d; }
extern "C" void *ds4_gpu_pp_active_ptr(int g) { return (g >= 0 && g < g_pp_ngpu) ? g_pp_active[g] : NULL; }
extern "C" int ds4_gpu_pp_p2p_copy(int dst_gpu, int src_gpu) {
    if (dst_gpu < 0 || dst_gpu >= g_pp_ngpu || src_gpu < 0 || src_gpu >= g_pp_ngpu) return 0;
    cudaError_t ce = cudaMemcpyPeer(g_pp_active[dst_gpu], dst_gpu,
                                     g_pp_active[src_gpu], src_gpu,
                                     4 * g_model_n_embd * sizeof(float));
    return ce == cudaSuccess ? 1 : 0;
}

extern "C" int ds4_gpu_pp_p2p_copy_ptr(int dst_gpu, int src_gpu,
                                         void *dst_ptr, void *src_ptr,
                                         uint64_t bytes) {
    if (!dst_ptr || !src_ptr || bytes == 0) return 0;
    if (g_pp_nccl_ready && src_gpu >= 0 && src_gpu < g_pp_ngpu && dst_gpu >= 0 && dst_gpu < g_pp_ngpu) {
        ncclResult_t nr = ncclGroupStart();
        size_t count = bytes / sizeof(float);
        if (nr == ncclSuccess) nr = ncclSend(src_ptr, count, ncclFloat, dst_gpu, g_pp_nccl_comms[src_gpu], cudaStreamDefault);
        if (nr == ncclSuccess) nr = ncclRecv(dst_ptr, count, ncclFloat, src_gpu, g_pp_nccl_comms[src_gpu], cudaStreamDefault);
        if (nr == ncclSuccess) nr = ncclGroupEnd();
        if (nr == ncclSuccess) {
            cudaSetDevice(src_gpu);
            cudaError_t ce = cudaStreamSynchronize(cudaStreamDefault);
            if (ce == cudaSuccess) return 1;
            fprintf(stderr, "ds4: NCCL P2P copy stream sync failed: %s\n", cudaGetErrorString(ce));
        } else {
            fprintf(stderr, "ds4: NCCL P2P copy failed: %s, falling back to cudaMemcpyPeer\n", ncclGetErrorString(nr));
        }
    }
    cudaError_t ce = cudaMemcpyPeer(dst_ptr, dst_gpu, src_ptr, src_gpu, (size_t)bytes);
    return ce == cudaSuccess ? 1 : 0;
}

/* Async variant: copy on a specific stream. Caller must ensure ordering via
   cudaEventRecord/cudaStreamWaitEvent. */
extern "C" int ds4_gpu_pp_p2p_copy_async(int dst_gpu, int src_gpu,
                                          void *dst_ptr, void *src_ptr,
                                          uint64_t bytes, cudaStream_t stream) {
    if (!dst_ptr || !src_ptr || bytes == 0) return 0;
    cudaSetDevice(dst_gpu);
    cudaError_t ce = cudaMemcpyPeerAsync(dst_ptr, dst_gpu, src_ptr, src_gpu, (size_t)bytes, stream);
    return ce == cudaSuccess ? 1 : 0;
}

extern "C" int ds4_gpu_pp_event_record(int gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_event[gpu]) return 0;
    cudaSetDevice(gpu);
    return cudaEventRecord(g_pp_event[gpu], ds4_decode_stream()) == cudaSuccess ? 1 : 0;
}

extern "C" int ds4_gpu_pp_stream_wait_event(int gpu, int event_gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_stream[gpu]) return 0;
    if (event_gpu < 0 || event_gpu >= DS4_CUDA_MAX_DEVICES || !g_pp_event[event_gpu]) return 0;
    cudaSetDevice(gpu);
    return cudaStreamWaitEvent(g_pp_stream[gpu], g_pp_event[event_gpu], 0) == cudaSuccess ? 1 : 0;
}

extern "C" void *ds4_gpu_pp_stream_get(int gpu) {
    if (gpu < 0 || gpu >= DS4_CUDA_MAX_DEVICES) return NULL;
    return g_pp_stream[gpu];
}

extern "C" int ds4_gpu_pp_p2p_copy_ordered_async(int dst_gpu, int src_gpu,
                                                   void *dst_ptr, void *src_ptr,
                                                   uint64_t bytes) {
    if (!dst_ptr || !src_ptr || bytes == 0) return 0;
    if (dst_gpu < 0 || dst_gpu >= DS4_CUDA_MAX_DEVICES || src_gpu < 0 || src_gpu >= DS4_CUDA_MAX_DEVICES) return 0;
    if (!g_pp_stream[dst_gpu] || !g_pp_event[src_gpu] || !g_pp_copy_event[dst_gpu]) return 0;

    cudaSetDevice(src_gpu);
    cudaStream_t src_work_stream = ds4_decode_stream();
    cudaError_t ce = cudaEventRecord(g_pp_event[src_gpu], src_work_stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP async source event record failed gpu%d: %s\n",
                src_gpu, cudaGetErrorString(ce));
        return 0;
    }

    cudaSetDevice(dst_gpu);
    cudaStream_t dst_copy_stream = g_pp_stream[dst_gpu];
    ce = cudaStreamWaitEvent(dst_copy_stream, g_pp_event[src_gpu], 0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP async copy stream wait failed gpu%d<-gpu%d: %s\n",
                dst_gpu, src_gpu, cudaGetErrorString(ce));
        return 0;
    }
    ce = cudaMemcpyPeerAsync(dst_ptr, dst_gpu, src_ptr, src_gpu, (size_t)bytes, dst_copy_stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP async cudaMemcpyPeerAsync failed gpu%d<-gpu%d: %s\n",
                dst_gpu, src_gpu, cudaGetErrorString(ce));
        return 0;
    }
    ce = cudaEventRecord(g_pp_copy_event[dst_gpu], dst_copy_stream);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP async copy event record failed gpu%d: %s\n",
                dst_gpu, cudaGetErrorString(ce));
        return 0;
    }

    cudaStream_t dst_work_stream = ds4_decode_stream();
    ce = cudaStreamWaitEvent(dst_work_stream, g_pp_copy_event[dst_gpu], 0);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: PP async destination stream wait failed gpu%d<-gpu%d: %s\n",
                dst_gpu, src_gpu, cudaGetErrorString(ce));
        return 0;
    }
    return 1;
}

/* ---- Tensor-Parallel collectives (host-staged ring all-reduce) ----
 * P2P peer-access is broken on this Blackwell driver, so cudaMemcpyPeer is
 * host-staged (shared PCIe/host bottleneck). A ring all-reduce moves the least
 * total data, so it is the right algorithm here (recursive-doubling moves more
 * and is slower on a shared link). It is issued async on per-device streams
 * ordered by events with a single sync at the end — per-step global syncs
 * otherwise dominate. Parameterized by an explicit device-id group so it works
 * for any TP degree / hybrid TPxPP layout. */
__global__ static void tp_vadd_f32_kernel(float *dst, const float *src, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

static int tp_collective_ensure(const int *devs, int n, uint32_t n_floats) {
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;
    for (int i = 0; i < n; i++) {
        int d = devs[i];
        if (d < 0 || d >= DS4_CUDA_MAX_DEVICES) { ok = 0; break; }
        if (cudaSetDevice(d) != cudaSuccess) { ok = 0; break; }
        if (!g_tp_stream[d] &&
            cudaStreamCreateWithFlags(&g_tp_stream[d], cudaStreamNonBlocking) != cudaSuccess) { ok = 0; break; }
        if (!g_tp_event[d] &&
            cudaEventCreateWithFlags(&g_tp_event[d], cudaEventDisableTiming) != cudaSuccess) { ok = 0; break; }
        if (g_tp_recv_floats[d] < n_floats) {
            if (g_tp_recv[d]) { cudaFree(g_tp_recv[d]); g_tp_recv[d] = NULL; g_tp_recv_floats[d] = 0; }
            if (cudaMalloc(&g_tp_recv[d], (size_t)n_floats * sizeof(float)) != cudaSuccess) {
                (void)cudaGetLastError(); ok = 0; break;
            }
            g_tp_recv_floats[d] = n_floats;
        }
    }
    (void)cudaSetDevice(saved);
    return ok;
}

/* All-reduce(sum) float buffers across a TP group. bufs[i] is the device
 * pointer on devs[i]; each holds n_floats. On return every device's buffer
 * holds the elementwise sum. Returns 1 on success. */
extern "C" int ds4_gpu_tp_all_reduce_f32(const int *devs, int n,
                                         float *const *bufs, uint32_t n_floats) {
    if (!devs || !bufs || n <= 0 || n_floats == 0) return 0;
    if (n == 1) return 1;
    if (!tp_collective_ensure(devs, n, n_floats)) return 0;

    const uint32_t chunk = (n_floats + (uint32_t)n - 1u) / (uint32_t)n;
#define TP_COFF(c) ((uint32_t)(c) * chunk)
#define TP_CLEN(c) ( TP_COFF(c) >= n_floats ? 0u : \
                     (n_floats - TP_COFF(c) < chunk ? n_floats - TP_COFF(c) : chunk) )
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;

    /* reduce-scatter: after N-1 steps, ring-index i holds the full sum of chunk (i+1)%N */
    for (int step = 0; ok && step < n - 1; step++) {
        for (int i = 0; i < n; i++) { cudaSetDevice(devs[i]); cudaEventRecord(g_tp_event[devs[i]], g_tp_stream[devs[i]]); }
        for (int i = 0; ok && i < n; i++) {
            int dst = (i + 1) % n;
            int c = ((i - step) % n + n) % n;
            uint32_t len = TP_CLEN(c); if (!len) continue;
            int dd = devs[dst], sd = devs[i];
            if (cudaSetDevice(dd) != cudaSuccess) { ok = 0; break; }
            cudaStreamWaitEvent(g_tp_stream[dd], g_tp_event[sd], 0);
            if (cudaMemcpyPeerAsync(g_tp_recv[dd] + TP_COFF(c), dd, bufs[i] + TP_COFF(c), sd,
                                    (size_t)len * sizeof(float), g_tp_stream[dd]) != cudaSuccess) { ok = 0; break; }
            tp_vadd_f32_kernel<<<(len + 255u) / 256u, 256, 0, g_tp_stream[dd]>>>(
                bufs[dst] + TP_COFF(c), g_tp_recv[dd] + TP_COFF(c), len);
        }
    }
    /* all-gather: rotate the reduced chunks so every device holds all of them */
    for (int step = 0; ok && step < n - 1; step++) {
        for (int i = 0; i < n; i++) { cudaSetDevice(devs[i]); cudaEventRecord(g_tp_event[devs[i]], g_tp_stream[devs[i]]); }
        for (int i = 0; ok && i < n; i++) {
            int dst = (i + 1) % n;
            int c = ((i + 1 - step) % n + n) % n;
            uint32_t len = TP_CLEN(c); if (!len) continue;
            int dd = devs[dst], sd = devs[i];
            if (cudaSetDevice(dd) != cudaSuccess) { ok = 0; break; }
            cudaStreamWaitEvent(g_tp_stream[dd], g_tp_event[sd], 0);
            if (cudaMemcpyPeerAsync(bufs[dst] + TP_COFF(c), dd, bufs[i] + TP_COFF(c), sd,
                                    (size_t)len * sizeof(float), g_tp_stream[dd]) != cudaSuccess) { ok = 0; break; }
        }
    }
    for (int i = 0; i < n; i++) { cudaSetDevice(devs[i]); if (cudaStreamSynchronize(g_tp_stream[devs[i]]) != cudaSuccess) ok = 0; }
#undef TP_COFF
#undef TP_CLEN
    (void)cudaSetDevice(saved);
    return ok;
}

/* Self-test: each device d starts filled with (d+1); after all-reduce every
 * element on every device must equal sum(1..N)=N(N+1)/2. Env-gated. */
static int ds4_gpu_tp_selftest(int ngpu) {
    int n = ngpu > DS4_CUDA_MAX_DEVICES ? DS4_CUDA_MAX_DEVICES : ngpu;
    if (n < 2) return 1;
    int devs[DS4_CUDA_MAX_DEVICES];
    float *bufs[DS4_CUDA_MAX_DEVICES] = {0};
    for (int i = 0; i < n; i++) devs[i] = i;
    const uint32_t M = 28672u; /* hc_dim */
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;
    std::vector<float> host(M);
    for (int i = 0; ok && i < n; i++) {
        if (cudaSetDevice(devs[i]) != cudaSuccess) { ok = 0; break; }
        if (cudaMalloc(&bufs[i], (size_t)M * sizeof(float)) != cudaSuccess) { (void)cudaGetLastError(); ok = 0; break; }
        for (uint32_t k = 0; k < M; k++) host[k] = (float)(devs[i] + 1);
        if (cudaMemcpy(bufs[i], host.data(), (size_t)M * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { ok = 0; break; }
    }
    double dt = 0.0;
    if (ok) {
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        ok = ds4_gpu_tp_all_reduce_f32(devs, n, bufs, M);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        dt = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    }
    int bad = 0; float sample = 0.0f;
    const float expect = (float)(n * (n + 1) / 2);
    for (int i = 0; ok && i < n; i++) {
        cudaSetDevice(devs[i]);
        if (cudaMemcpy(host.data(), bufs[i], (size_t)M * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) { ok = 0; break; }
        for (uint32_t k = 0; k < M; k++) if (host[k] != expect) { if (!bad) sample = host[k]; bad++; }
    }
    for (int i = 0; i < n; i++) if (bufs[i]) { cudaSetDevice(devs[i]); cudaFree(bufs[i]); }
    (void)cudaSetDevice(saved);
    fprintf(stderr, "ds4: TP all-reduce self-test N=%d M=%u: %s (%.4f ms, expect=%.0f%s)\n",
            n, M, (ok && !bad) ? "PASS" : "FAIL", dt, (double)expect,
            bad ? "" : "");
    if (bad) fprintf(stderr, "ds4: TP self-test mismatch count=%d sample=%.1f\n", bad, sample);
    return (ok && !bad) ? 1 : 0;
}

extern "C" void ds4_gpu_model_range_reserve(void) {
    g_model_ranges.reserve(g_model_ranges.size() + 1024);
    g_q8_f16_ranges.reserve(g_q8_f16_ranges.size() + 1024);
    g_q8_f32_ranges.reserve(g_q8_f32_ranges.size() + 1024);
}

extern "C" void ds4_gpu_release_weight_cache_for_pp(void) {
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
}

extern "C" int ds4_gpu_cache_model_range_force(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA force-cache alloc failed for %s (%.2f MiB): %s\n",
                label ? label : "pp_weight", (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        return 0;
    }
    const char *src = (const char *)model_map + offset;
    err = cudaMemcpy(dev, src, (size_t)bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA force-cache copy failed for %s: %s\n",
                label ? label : "pp_weight", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }
    int current_dev = -1;
    (void)cudaGetDevice(&current_dev);
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0, current_dev});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1;
    g_model_range_bytes += bytes;
    return 1;
}

/* ---- Tensor-Parallel resident weight shards ----
 * Resolve a TP shard for the CURRENT device by parent (model_map, offset).
 * Returns the shard's device pointer and fills *kind plus the shard's matmul
 * dims (out_dim/in_dim/blocks as stored). NULL if no shard for this device.
 * Mirrors cuda_model_range_ptr's "must match current device" rule. */
static const char *cuda_tp_shard_ptr(const void *model_map, uint64_t offset,
                                     int *kind, uint64_t *out_dim,
                                     uint64_t *in_dim, uint64_t *blocks) {
    int cur = -1; (void)cudaGetDevice(&cur);
    for (const cuda_tp_shard &s : g_tp_shards) {
        if (s.host_base == model_map && s.offset == offset && s.device == cur) {
            if (kind) *kind = s.kind;
            if (out_dim) *out_dim = s.out_dim;
            if (in_dim) *in_dim = s.in_dim;
            if (blocks) *blocks = s.blocks;
            return s.device_ptr;
        }
    }
    return NULL;
}

/* Cache a COLUMN-parallel shard (Q8_0 [in_dim->out_dim]) on the current device:
 * rank r owns the contiguous output-row slice [r*rpr,(r+1)*rpr), rpr=out_dim/k.
 * That slice is a contiguous byte sub-range of the host tensor, so a plain H2D
 * copy suffices (matches ds4_gpu_cache_model_range_force, just offset+length). */
extern "C" int ds4_gpu_cache_col_shard(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t in_dim, uint64_t out_dim,
        int rank, int k, const char *label) {
    if (!model_map || in_dim == 0 || out_dim == 0 || k < 1 || rank < 0 || rank >= k) return 0;
    if ((out_dim % (uint64_t)k) != 0) {
        fprintf(stderr, "ds4: col-shard %s: out_dim(%llu) not divisible by k=%d\n",
                label ? label : "?", (unsigned long long)out_dim, k);
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t rpr = out_dim / (uint64_t)k;
    const uint64_t row_bytes = blocks * 34;
    const uint64_t shard_bytes = rpr * row_bytes;
    const uint64_t src_off = offset + (uint64_t)rank * shard_bytes;
    if (src_off > model_size || shard_bytes > model_size - src_off) return 0;
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)shard_bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: col-shard %s alloc failed (%.2f MiB): %s\n",
                label ? label : "?", (double)shard_bytes / 1048576.0, cudaGetErrorString(err));
        return 0;
    }
    err = cudaMemcpy(dev, (const char *)model_map + src_off, (size_t)shard_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: col-shard %s copy failed: %s\n", label ? label : "?", cudaGetErrorString(err));
        (void)cudaFree(dev); (void)cudaGetLastError();
        return 0;
    }
    int cur = -1; (void)cudaGetDevice(&cur);
    cuda_tp_shard s;
    s.host_base = model_map; s.offset = offset; s.rank = rank; s.k = k;
    s.kind = DS4_TP_SHARD_COL; s.device = cur; s.device_ptr = (char *)dev; s.bytes = shard_bytes;
    s.out_dim = rpr; s.in_dim = in_dim; s.blocks = blocks;
    g_tp_shards.push_back(s);
    g_tp_shard_bytes += shard_bytes;
    return 1;
}

/* Cache a ROW-parallel shard (Q8_0 [in_dim->out_dim]) on the current device:
 * rank r owns the input-block slice [r*bpr,(r+1)*bpr), bpr=blocks/k, for EVERY
 * output row. That is a strided gather, so pack it with cudaMemcpy2D (H2D) into
 * a dense [out_dim x bpr] Q8_0 buffer — the same pack the TP jig validated, but
 * straight from the host map (no resident full copy needed). The packed buffer
 * is then a normal Q8_0 weight of in=bpr*32, out=out_dim, blocks=bpr. */
extern "C" int ds4_gpu_cache_row_shard(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t in_dim, uint64_t out_dim,
        int rank, int k, const char *label) {
    if (!model_map || in_dim == 0 || out_dim == 0 || k < 1 || rank < 0 || rank >= k) return 0;
    const uint64_t blocks = (in_dim + 31) / 32;
    if ((blocks % (uint64_t)k) != 0) {
        fprintf(stderr, "ds4: row-shard %s: blocks(%llu) not divisible by k=%d\n",
                label ? label : "?", (unsigned long long)blocks, k);
        return 0;
    }
    const uint64_t bpr = blocks / (uint64_t)k;
    const uint64_t full_bytes = out_dim * blocks * 34;
    if (offset > model_size || full_bytes > model_size - offset) return 0;
    const uint64_t dst_pitch = bpr * 34;
    const uint64_t src_pitch = blocks * 34;
    const uint64_t shard_bytes = out_dim * dst_pitch;
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)shard_bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: row-shard %s alloc failed (%.2f MiB): %s\n",
                label ? label : "?", (double)shard_bytes / 1048576.0, cudaGetErrorString(err));
        return 0;
    }
    const char *src = (const char *)model_map + offset + (uint64_t)rank * dst_pitch;
    err = cudaMemcpy2D(dev, (size_t)dst_pitch, src, (size_t)src_pitch,
                       (size_t)dst_pitch, (size_t)out_dim, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: row-shard %s pack failed: %s\n", label ? label : "?", cudaGetErrorString(err));
        (void)cudaFree(dev); (void)cudaGetLastError();
        return 0;
    }
    int cur = -1; (void)cudaGetDevice(&cur);
    cuda_tp_shard s;
    s.host_base = model_map; s.offset = offset; s.rank = rank; s.k = k;
    s.kind = DS4_TP_SHARD_ROW; s.device = cur; s.device_ptr = (char *)dev; s.bytes = shard_bytes;
    s.out_dim = out_dim; s.in_dim = bpr * 32; s.blocks = bpr;
    g_tp_shards.push_back(s);
    g_tp_shard_bytes += shard_bytes;
    return 1;
}

/* NOTE: routed-expert (MoE) sharding does NOT use g_tp_shards. The owned expert
 * subset [r*ec,(r+1)*ec) is one contiguous byte range, so the TP cache loop
 * force-caches it in g_model_ranges at its natural sub-offset (gate_offset +
 * e0*expert_bytes). The UNCHANGED routed_moe then addresses experts as
 * gate_offset + gid*expert_bytes and transparently resolves owned experts within
 * that cached sub-range; the decode encode filters the router selection to the
 * owned id-range (non-owned remapped to the first owned expert with weight 0). */

extern "C" void ds4_gpu_tp_shards_release_all(void) {
    int saved = -1; (void)cudaGetDevice(&saved);
    for (const cuda_tp_shard &s : g_tp_shards) {
        if (s.device_ptr) { (void)cudaSetDevice(s.device); (void)cudaFree(s.device_ptr); }
    }
    if (saved >= 0) (void)cudaSetDevice(saved);
    g_tp_shards.clear();
    g_tp_shard_bytes = 0;
}

extern "C" void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    {
        int saved = 0;
        (void)cudaGetDevice(&saved);
        for (int d = 0; d < DS4_CUDA_MAX_DEVICES; ++d) {
            if (!g_cuda_tmp_dev[d]) continue;
            if (cudaSetDevice(d) == cudaSuccess) {
                (void)cudaFree(g_cuda_tmp_dev[d]);
            } else {
                (void)cudaGetLastError();
            }
            g_cuda_tmp_dev[d] = NULL;
            g_cuda_tmp_bytes_dev[d] = 0;
        }
        (void)cudaSetDevice(saved);
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
    if (g_cuda_decode_stream_created && g_cuda_decode_stream) {
        (void)cudaStreamDestroy(g_cuda_decode_stream);
        g_cuda_decode_stream = 0;
        g_cuda_decode_stream_created = 0;
    }
    if (g_cublas_workspace) {
        (void)cudaFree(g_cublas_workspace);
        g_cublas_workspace = NULL;
        g_cublas_workspace_bytes = 0;
    }
    if (g_cuda_decode_graph_exec) {
        (void)cudaGraphExecDestroy(g_cuda_decode_graph_exec);
        g_cuda_decode_graph_exec = NULL;
    }
    if (g_cuda_decode_graph) {
        (void)cudaGraphDestroy(g_cuda_decode_graph);
        g_cuda_decode_graph = NULL;
    }
    g_cuda_decode_graph_captured = 0;
    for (int g = 0; g < DS4_CUDA_MAX_DEVICES; g++) {
        if (g_pp_chunk_graph_exec[g]) {
            (void)cudaGraphExecDestroy(g_pp_chunk_graph_exec[g]);
            g_pp_chunk_graph_exec[g] = NULL;
        }
        if (g_pp_chunk_graph[g]) {
            (void)cudaGraphDestroy(g_pp_chunk_graph[g]);
            g_pp_chunk_graph[g] = NULL;
        }
        g_pp_chunk_graph_ready[g] = 0;
    }
    for (int p = 0; p < 2; p++) for (int i = 0; i < DS4_GPU_MAX_LAYER; i++) {
        if (g_sub_graph_exec[p][i]) { (void)cudaGraphExecDestroy(g_sub_graph_exec[p][i]); g_sub_graph_exec[p][i] = NULL; }
        g_sub_graph_exec_ready[p][i] = 0;
    }
    if (g_moe_scratch_gate) { (void)cudaFree(g_moe_scratch_gate); g_moe_scratch_gate = NULL; }
    if (g_moe_scratch_up)   { (void)cudaFree(g_moe_scratch_up);   g_moe_scratch_up = NULL; }
    if (g_moe_scratch_down) { (void)cudaFree(g_moe_scratch_down); g_moe_scratch_down = NULL; }
    g_moe_scratch_gate_bytes = 0;
    g_moe_scratch_up_bytes = 0;
    g_moe_scratch_down_bytes = 0;
    g_moe_scratch_inited = 0;
    for (int cd = 0; cd < DS4_CUDA_MAX_DEVICES; cd++) {
        if (g_moe_compact_gate[cd]) { (void)cudaFree(g_moe_compact_gate[cd]); g_moe_compact_gate[cd] = NULL; }
        if (g_moe_compact_up[cd])   { (void)cudaFree(g_moe_compact_up[cd]);   g_moe_compact_up[cd] = NULL; }
        if (g_moe_compact_down[cd]) { (void)cudaFree(g_moe_compact_down[cd]); g_moe_compact_down[cd] = NULL; }
        if (g_moe_compact_selected_dev[cd]) { (void)cudaFree(g_moe_compact_selected_dev[cd]); g_moe_compact_selected_dev[cd] = NULL; }
        g_moe_compact_per_expert[cd] = 0;
        g_moe_compact_slots[cd] = 0;
    }
    if (g_pp_topology_ready) {
        if (g_pp_nccl_ready) {
            for (int g = 0; g < g_pp_ngpu; g++) {
                if (g_pp_nccl_comms[g]) {
                    ncclCommDestroy(g_pp_nccl_comms[g]);
                    g_pp_nccl_comms[g] = NULL;
                }
            }
            g_pp_nccl_ready = 0;
        }
        for (int g = 0; g < g_pp_ngpu; g++) {
            cudaSetDevice(g);
            if (g_pp_stream[g]) { (void)cudaStreamDestroy(g_pp_stream[g]); g_pp_stream[g] = NULL; }
            if (g_pp_event[g]) { (void)cudaEventDestroy(g_pp_event[g]); g_pp_event[g] = NULL; }
            if (g_pp_copy_event[g]) { (void)cudaEventDestroy(g_pp_copy_event[g]); g_pp_copy_event[g] = NULL; }
            (void)cudaFree(g_pp_active[g]);
            g_pp_active[g] = NULL;
        }
        memset(g_pp_layer_start, 0, sizeof(g_pp_layer_start));
        memset(g_pp_layer_end, 0, sizeof(g_pp_layer_end));
        g_pp_ngpu = 0;
        g_pp_topology_ready = 0;
        g_pp_decode_active = 0;
        g_pp_resident_ready = 0;
        g_pp_requested = 0;
    }
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" void ds4_gpu_debug_tensor_ptr(const char *name, ds4_gpu_tensor *t) {
    if (!t) { fprintf(stderr, "ds4: %s is NULL\n", name); return; }
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, t->ptr);
    if (err == cudaSuccess) {
        fprintf(stderr, "ds4: %s ptr=%p device=%d type=%d bytes=%lu\n",
                name, t->ptr, attr.device, (int)attr.type, (unsigned long)t->bytes);
    } else {
        fprintf(stderr, "ds4: %s ptr=%p attr failed: %s\n", name, t->ptr, cudaGetErrorString(err));
    }
}

static void ds4_debug_ptr_attr_raw(const char *name, const void *p) {
    int cur = -1;
    cudaGetDevice(&cur);
    cudaPointerAttributes a;
    cudaError_t e = cudaPointerGetAttributes(&a, p);
    if (e == cudaSuccess) {
        fprintf(stderr,
                "ds4: %s ptr=%p current_dev=%d attr.device=%d attr.type=%d\n",
                name, p, cur, a.device, (int)a.type);
    } else {
        fprintf(stderr,
                "ds4: %s ptr=%p current_dev=%d attr failed: %s\n",
                name, p, cur, cudaGetErrorString(e));
        cudaGetLastError();
    }
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

/* No-sync variant for PP hot path where we only need the device pointer
   (not reading tensor data on host). */
extern "C" void *ds4_gpu_tensor_device_ptr(ds4_gpu_tensor *tensor) {
    return tensor ? tensor->ptr : NULL;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256, 0, ds4_decode_stream()>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

extern "C" int ds4_gpu_begin_commands(void) { return 1; }
extern "C" int ds4_gpu_flush_commands(void) {
    if (g_cuda_decode_stream_created) {
        cudaStreamCaptureStatus status;
        cudaError_t ce = cudaStreamIsCapturing(g_cuda_decode_stream, &status);
        if (ce == cudaSuccess && status == cudaStreamCaptureStatusActive) return 1;
        (void)cudaGetLastError();
    }
    return cuda_ok(cudaDeviceSynchronize(), "flush");
}
extern "C" int ds4_gpu_end_commands(void) {
    if (g_cuda_decode_stream_created) {
        cudaStreamCaptureStatus status;
        cudaError_t ce = cudaStreamIsCapturing(g_cuda_decode_stream, &status);
        if (ce == cudaSuccess && status == cudaStreamCaptureStatusActive) return 1;
        (void)cudaGetLastError();
    }
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
extern "C" int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }

extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
        g_model_device_owned = 0;
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
        g_model_registered = 0;
    }
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    const char *copy_env = getenv("DS4_CUDA_COPY_MODEL");
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (getenv("DS4_CUDA_COPY_MODEL_CHUNKED") != NULL &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    g_model_fd = fd;
    g_model_fd_host_base = model_map;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (getenv("DS4_CUDA_NO_DIRECT_IO") == NULL) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    return ds4_gpu_set_model_fd_for_map(fd, g_model_host_base);
}

/* =========================================================================
 * SSD expert-streaming public API (Part 2 port from upstream antirez/ds4,
 * UPSTREAM ds4_cuda.cu 2757-3345). Per-device transform applied: bodies that
 * touch the per-device streaming globals resolve the slot via cudaGetDevice.
 * NOTE (step 5): set_ssd_streaming / set_streaming_expert_cache_budget /
 * release_resident currently tear down only the CURRENT device; the multi-GPU
 * teardown + seed/budget device loop is gap (c) handled in the routed-moe step.
 * ========================================================================= */
extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    g_ssd_streaming_mode = enabled ? 1 : 0;
    g_stream_expert_runtime_cap[dev] = 0;
    g_stream_expert_runtime_gate_bytes[dev] = 0;
    g_stream_expert_runtime_down_bytes[dev] = 0;
    g_stream_expert_memory_cap_notice[dev] = 0;
    if (!g_ssd_streaming_mode) {
        cuda_stream_selected_cache_release();
        cuda_stream_expert_cache_release_all();
    }
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    g_stream_expert_budget_override = experts;
    g_stream_expert_runtime_cap[dev] = 0;
    g_stream_expert_runtime_gate_bytes[dev] = 0;
    g_stream_expert_runtime_down_bytes[dev] = 0;
    g_stream_expert_memory_cap_notice[dev] = 0;
    cuda_stream_selected_cache_invalidate();
    cuda_stream_expert_cache_release_all();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) {
    return 0;
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared()) return 0;
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    return g_stream_expert_cache[dev].count;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {
}

extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    cuda_stream_expert_cache_release_all();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!cuda_stream_expert_cache_budget_visible_to_shared() ||
        cuda_stream_expert_cache_expert_bytes(gate_expert_bytes,
                                              down_expert_bytes) == 0) {
        return 0;
    }
    cuda_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);
    return cuda_stream_expert_cache_configured_budget();
}

extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !selected_ids || n_selected == 0 ||
        n_selected > n_total_expert ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed selected")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_selected);
    if (!cache) return 1;
    for (uint32_t i = 0; i < n_selected; i++) {
        if (selected_ids[i] < 0 || (uint32_t)selected_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming seed selected expert id %d is outside 0..%u at layer %u\n",
                    selected_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)selected_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    return 1;
}

static int cuda_stream_selected_cache_begin_compact_load(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       n_total_expert,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            strict_failure,
        int            allow_global_cache) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= DS4_CUDA_MAX_DEVICES) dev = 0;
    cuda_stream_selected_cache_invalidate();
    cuda_model_load_progress_finish();

    if (!g_ssd_streaming_mode) return 1;
    if (!model_map || !compact_ids || !slot_ids ||
        n_total_expert == 0 ||
        compact_count == 0 || compact_count > n_total_expert ||
        slot_count == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: CUDA streaming selected expert size overflow\n");
        return 0;
    }

    const uint64_t full_gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t compact_gate_bytes = (uint64_t)compact_count * gate_expert_bytes;
    const uint64_t compact_down_bytes = (uint64_t)compact_count * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: CUDA streaming selected expert range outside model map\n");
        return 0;
    }

    if (!allow_global_cache) {
        cuda_stream_expert_cache_release_all();
    }

    if (!cuda_stream_selected_ensure_bytes(&g_stream_selected_cache[dev].gate_ptr,
                                           &g_stream_selected_cache[dev].gate_capacity,
                                           compact_gate_bytes,
                                           "selected gate experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache[dev].up_ptr,
                                           &g_stream_selected_cache[dev].up_capacity,
                                           compact_gate_bytes,
                                           "selected up experts") ||
        !cuda_stream_selected_ensure_bytes(&g_stream_selected_cache[dev].down_ptr,
                                           &g_stream_selected_cache[dev].down_capacity,
                                           compact_down_bytes,
                                           "selected down experts") ||
        !cuda_stream_selected_ensure_i32(&g_stream_selected_cache[dev].slot_selected_ptr,
                                         &g_stream_selected_cache[dev].slot_selected_capacity,
                                         slot_count,
                                         "selected expert slots")) {
        return strict_failure ? 0 : 1;
    }

    if (allow_global_cache) {
        cuda_stream_expert_cache_note_size(gate_expert_bytes,
                                           down_expert_bytes);
    }
    const uint32_t configured_cache_budget =
        cuda_stream_expert_cache_configured_budget();
    const int use_global_cache =
        allow_global_cache &&
        configured_cache_budget != 0;
    cuda_stream_expert_cache *expert_cache = use_global_cache ?
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         configured_cache_budget) :
        NULL;
    int expert_cache_disabled = expert_cache == NULL;
    const uint32_t cache_count_before =
        expert_cache && expert_cache->valid ? expert_cache->count : 0;
    uint32_t cache_hits = 0;
    uint32_t cache_misses = 0;
    uint32_t direct_loads = 0;

    for (uint32_t i = 0; i < compact_count; i++) {
        if (compact_ids[i] < 0 || (uint32_t)compact_ids[i] >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    compact_ids[i],
                    n_total_expert,
                    layer);
            return 0;
        }

        const uint64_t expert = (uint64_t)(uint32_t)compact_ids[i];
        const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
        int copied_from_global_cache = 0;

        if (!expert_cache_disabled) {
            int cache_slot =
                cuda_stream_expert_cache_find(expert_cache,
                                              model_map,
                                              model_size,
                                              layer,
                                              n_total_expert,
                                              (uint32_t)expert,
                                              gate_offset,
                                              up_offset,
                                              down_offset,
                                              gate_expert_bytes,
                                              down_expert_bytes);
            if (cache_slot >= 0) {
                cache_hits++;
                expert_cache->slots[(uint32_t)cache_slot].age =
                    ++expert_cache->tick;
            } else {
                cache_misses++;
                const uint32_t load_slot =
                    cuda_stream_expert_cache_lru_slot(expert_cache);
                const int append = !expert_cache->slots[load_slot].valid;
                if (cuda_stream_expert_cache_load_slot(expert_cache,
                                                       model_map,
                                                       model_size,
                                                       load_slot,
                                                       layer,
                                                       n_total_expert,
                                                       (uint32_t)expert,
                                                       gate_offset,
                                                       up_offset,
                                                       down_offset,
                                                       gate_expert_bytes,
                                                       down_expert_bytes)) {
                    if (append && expert_cache->count < expert_cache->capacity) {
                        expert_cache->count++;
                    }
                    cache_slot = (int)load_slot;
                } else {
                    cuda_stream_expert_cache_invalidate();
                    expert_cache_disabled = 1;
                    cache_slot = -1;
                }
            }

            if (cache_slot >= 0) {
                copied_from_global_cache =
                    cuda_stream_expert_cache_copy_to_compact(
                            expert_cache,
                            (uint32_t)cache_slot,
                            i,
                            g_stream_selected_cache[dev].gate_ptr,
                            g_stream_selected_cache[dev].up_ptr,
                            g_stream_selected_cache[dev].down_ptr);
                if (!copied_from_global_cache) {
                    cuda_stream_expert_cache_invalidate();
                    expert_cache_disabled = 1;
                }
            }
        }

        if (!copied_from_global_cache) {
            const uint64_t gate_src = gate_offset + expert * gate_expert_bytes;
            const uint64_t up_src = up_offset + expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + expert * down_expert_bytes;
            direct_loads++;
            if (!cuda_model_copy_to_device_streamed(g_stream_selected_cache[dev].gate_ptr + gate_dst,
                                                    model_map,
                                                    model_size,
                                                    gate_src,
                                                    gate_expert_bytes,
                                                    "selected moe_gate") ||
                !cuda_model_copy_to_device_streamed(g_stream_selected_cache[dev].up_ptr + gate_dst,
                                                    model_map,
                                                    model_size,
                                                    up_src,
                                                    gate_expert_bytes,
                                                    "selected moe_up") ||
                !cuda_model_copy_to_device_streamed(g_stream_selected_cache[dev].down_ptr + down_dst,
                                                    model_map,
                                                    model_size,
                                                    down_src,
                                                    down_expert_bytes,
                                                    "selected moe_down")) {
                cuda_stream_selected_cache_invalidate();
                return strict_failure ? 0 : 1;
            }
        }
    }

    if (!cuda_ok(cudaMemcpy(g_stream_selected_cache[dev].slot_selected_ptr,
                            slot_ids,
                            (size_t)slot_count * sizeof(slot_ids[0]),
                            cudaMemcpyHostToDevice),
                 "streaming selected slot upload")) {
        cuda_stream_selected_cache_invalidate();
        return strict_failure ? 0 : 1;
    }

    g_stream_selected_cache[dev].model_map = model_map;
    g_stream_selected_cache[dev].layer = layer;
    g_stream_selected_cache[dev].n_total_expert = n_total_expert;
    g_stream_selected_cache[dev].n_selected = slot_count;
    g_stream_selected_cache[dev].slot_count = slot_count;
    g_stream_selected_cache[dev].compact_count = compact_count;
    g_stream_selected_cache[dev].gate_offset = gate_offset;
    g_stream_selected_cache[dev].up_offset = up_offset;
    g_stream_selected_cache[dev].down_offset = down_offset;
    g_stream_selected_cache[dev].gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache[dev].down_expert_bytes = down_expert_bytes;
    g_stream_selected_cache[dev].slot_selected_tensor.ptr =
        g_stream_selected_cache[dev].slot_selected_ptr;
    g_stream_selected_cache[dev].slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache[dev].slot_selected_tensor.owner = 0;
    g_stream_selected_cache[dev].valid = 1;

    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        cuda_model_load_progress_finish();
        fprintf(stderr,
                "ds4: CUDA streaming selected layer=%u slots=%u compact=%u global_budget=%u before=%u after=%u hits=%u misses=%u direct=%u gate/up %.2f MiB down %.2f MiB\n",
                layer,
                slot_count,
                compact_count,
                expert_cache && expert_cache->valid ? expert_cache->capacity : 0,
                cache_count_before,
                expert_cache && expert_cache->valid ? expert_cache->count : 0,
                cache_hits,
                cache_misses,
                direct_loads,
                (double)compact_gate_bytes / 1048576.0,
                (double)compact_down_bytes / 1048576.0);
    }
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table || !selected_ids || n_selected == 0) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    std::vector<int32_t> slot_ids(n_selected);
    compact_ids.reserve(n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < n_selected; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }
    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            n_selected,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            0,
            1);
}

extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_tokens,
        uint32_t                           n_selected) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table ||
        !selected_ids ||
        table->n_total_expert == 0 ||
        n_selected == 0 ||
        n_tokens == 0 ||
        (uint64_t)n_tokens > UINT32_MAX / (uint64_t)n_selected) {
        return 0;
    }
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;

    std::vector<int32_t> expert_to_slot(n_total_expert, -1);
    std::vector<int32_t> compact_ids;
    const uint32_t slot_count = n_tokens * n_selected;
    std::vector<int32_t> slot_ids(slot_count);
    compact_ids.reserve(slot_count < n_total_expert ? slot_count : n_total_expert);

    for (uint32_t i = 0; i < slot_count; i++) {
        const int32_t expert_i = selected_ids[i];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming batch selected expert id %d is outside 0..%u at layer %u\n",
                    expert_i,
                    n_total_expert,
                    layer);
            return 0;
        }
        expert_to_slot[(uint32_t)expert_i] = -2;
    }
    for (uint32_t e = 0; e < n_total_expert; e++) {
        if (expert_to_slot[e] != -2) continue;
        expert_to_slot[e] = (int32_t)compact_ids.size();
        compact_ids.push_back((int32_t)e);
    }
    for (uint32_t i = 0; i < slot_count; i++) {
        slot_ids[i] = expert_to_slot[(uint32_t)selected_ids[i]];
    }

    if (compact_ids.empty() || compact_ids.size() > UINT32_MAX) return 0;
    return cuda_stream_selected_cache_begin_compact_load(
            model_map,
            model_size,
            layer,
            compact_ids.data(),
            slot_ids.data(),
            n_total_expert,
            (uint32_t)compact_ids.size(),
            slot_count,
            gate_offset,
            up_offset,
            down_offset,
            gate_expert_bytes,
            down_expert_bytes,
            1,
            1 /* allow_global_cache: use the per-device resident expert LRU so
                * decode reuses hot experts across tokens and skips the PCIe
                * RAM->VRAM copy that the profile showed is ~67% of decode time.
                * Was 0 (direct copy every token); see streaming decode profiling. */);
}

extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts) {
    if (!g_ssd_streaming_mode) return 1;
    if (!table) return 0;
    const void *model_map = table->model_map;
    const uint64_t model_size = table->model_size;
    const uint32_t layer = table->layer;
    const uint32_t n_total_expert = table->n_total_expert;
    const uint64_t gate_offset = table->gate_offset;
    const uint64_t up_offset = table->up_offset;
    const uint64_t down_offset = table->down_offset;
    const uint64_t gate_expert_bytes = table->gate_expert_bytes;
    const uint64_t down_expert_bytes = table->down_expert_bytes;
    if (!model_map || !expert_ids || n_experts == 0 ||
        !cuda_stream_layer_expert_ranges_valid(model_size,
                                               n_total_expert,
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes,
                                               "seed hotlist")) {
        return 0;
    }

    cuda_stream_expert_cache *cache =
        cuda_stream_expert_cache_prepare(gate_expert_bytes,
                                         down_expert_bytes,
                                         n_experts);
    if (!cache || cache->capacity == 0) return 1;

    const uint32_t layer_seed_cap =
        n_experts < cache->capacity ? n_experts : cache->capacity;
    std::vector<uint32_t> chosen;
    try {
        chosen.reserve(layer_seed_cap);
    } catch (...) {
        return 1;
    }

    for (uint32_t i = 0; i < n_experts; i++) {
        const int32_t expert = expert_ids[i];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr,
                    "ds4: CUDA streaming hotlist seed expert id %d is outside 0..%u at layer %u\n",
                    expert,
                    n_total_expert,
                    layer);
            return 0;
        }
        const uint32_t priority =
            expert_priorities ? expert_priorities[i] : (n_experts - i);
        uint32_t pos = 0;
        while (pos < chosen.size()) {
            const uint32_t other = chosen[pos];
            const uint32_t other_priority =
                expert_priorities ? expert_priorities[other] :
                                    (n_experts - other);
            if (priority > other_priority) break;
            pos++;
        }
        if (chosen.size() < layer_seed_cap) {
            chosen.insert(chosen.begin() + pos, i);
        } else if (pos < chosen.size()) {
            chosen.insert(chosen.begin() + pos, i);
            chosen.pop_back();
        }
    }

    const uint32_t n = (uint32_t)chosen.size();
    for (uint32_t ri = 0; ri < n; ri++) {
        const uint32_t i = chosen[n - 1u - ri];
        if (!cuda_stream_expert_cache_seed_one(cache,
                                               model_map,
                                               model_size,
                                               layer,
                                               n_total_expert,
                                               (uint32_t)expert_ids[i],
                                               gate_offset,
                                               up_offset,
                                               down_offset,
                                               gate_expert_bytes,
                                               down_expert_bytes)) {
            cuda_stream_expert_cache_invalidate();
            return 1;
        }
    }
    if (getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "ds4: CUDA streaming hotlist seeded layer=%u requested=%u cached=%u cap=%u\n",
                layer,
                n_experts,
                n,
                cache->capacity);
    }
    return 1;
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (!cuda_model_range_ptr(model_map, offset, bytes, label ? label : "model_tensor")) return 0;
    return cuda_model_range_is_cached(model_map, offset, bytes);
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (getenv("DS4_CUDA_Q8_F32_PRELOAD") != NULL &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
        optional_q8_preload_disabled = 1;
        return 1;
    }
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
    optional_q8_preload_disabled = 1;
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    if (g_cublas_ready) {
        const cublasMath_t math_mode =
            (g_quality_mode || getenv("DS4_CUDA_NO_TF32") != NULL)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_vocab, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = (uint32_t)(i % n_embd);
    int32_t tok = ds4_cuda_params_active ? (int32_t)ds4_cuda_dev_params.token : (int32_t)token;
    uint32_t t = tok < 0 ? 0u : (uint32_t)tok;
    if (t >= n_vocab) t = 0u;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)t * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    __shared__ float partial[32];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = k0; i < k1; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    partial[tid] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) total += partial[i];
        out[tok * out_dim + row] = total;
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    uint64_t row = (uint64_t)blockIdx.x;
    if (row >= out0_dim && row >= out1_dim) return;

    __shared__ float partial0[32];
    __shared__ float partial1[32];
    const uint32_t tid = threadIdx.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const uint64_t chunk = (in_dim + 31u) / 32u;
    const uint64_t k0 = (uint64_t)tid * chunk;
    uint64_t k1 = k0 + chunk;
    if (k1 > in_dim) k1 = in_dim;
    const __half *wr0 = row < out0_dim ? w0 + row * in_dim : w0;
    const __half *wr1 = row < out1_dim ? w1 + row * in_dim : w1;
    for (uint64_t i = k0; i < k1; i++) {
        const float xv = x[i];
        if (row < out0_dim) sum0 += __half2float(wr0[i]) * xv;
        if (row < out1_dim) sum1 += __half2float(wr1[i]) * xv;
    }
    partial0[tid] = sum0;
    partial1[tid] = sum1;
    __syncthreads();
    if (tid == 0) {
        float total0 = 0.0f;
        float total1 = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) {
            total0 += partial0[i];
            total1 += partial1[i];
        }
        if (row < out0_dim) out0[row] = total0;
        if (row < out1_dim) out1[row] = total1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

/* Row-parallel (Tensor-Parallel) variant: each rank accumulates only over the
 * input-block sub-range [blocks_start, blocks_end). Writing partials from every
 * rank and summing them (all-reduce) reproduces the full dot product. Splitting
 * by whole 32-element Q8_0 blocks keeps quantization intact. add_to!=0 adds into
 * out (so a rank's own partial can accumulate without a separate buffer). */
__global__ static void matmul_q8_0_preq_warp8_brange_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        uint64_t blocks_start,
        uint64_t blocks_end,
        int add_to,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = blocks_start + lane; b < blocks_end; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = add_to ? out[row] + acc : acc;
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        uint32_t p = ds4_cuda_params_active ? ds4_cuda_dev_params.pos : pos0;
        float theta_extrap = (float)(p + t) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)((ds4_cuda_params_active ? ds4_cuda_dev_params.pos : pos0) + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__device__ static float rope_yarn_ramp_cpu_equiv_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static DS4_CUDA_UNUSED void rope_tail_one_dev(float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = fmaxf(0.0f, floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom));
        corr1 = fminf((float)(n_rot - 1), ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom));
    }
    for (uint32_t i = 0; i < n_rot; i += 2) {
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float mix = rope_yarn_ramp_cpu_equiv_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float x0 = x[n_nope + i];
        float x1 = x[n_nope + i + 1];
        x[n_nope + i] = x0 * c - x1 * s;
        x[n_nope + i + 1] = x0 * s + x1 * c;
    }
}

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    if (threadIdx.x == 0) {
        float den = expf(sinks[h] - max_s);
        for (uint32_t r = 0; r < raw_count; r++) {
            scores[r] = expf(scores[r] - max_s);
            den += scores[r];
        }
        denom = den;
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = NULL;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                }
                float s = -INFINITY;
                if (kvrow) {
                    float dot = 0.0f;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    for (uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
        int32_t c = topk[(uint64_t)t * top_k + i];
        if (c >= 0 && (uint32_t)c < visible_comp) {
            uint32_t slot = atomicAdd(&comp_count, 1u);
            if (slot < 512u) comp_rows[slot] = (uint32_t)c;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (comp_count > 512u) comp_count = 512u;
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const uint32_t comp_idx = sr < raw_count
                ? 0u
                : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_idx * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}

__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    int32_t token = ds4_cuda_params_active ? ds4_cuda_dev_params.token : token_scalar;
    const float *log = logits + (uint64_t)t * n_expert;
    float *prob = probs + (uint64_t)t * n_expert;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;

    for (uint32_t i = 0; i < n_expert; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * n_expert_used;
        for (uint32_t i = 0; i < n_expert_used; i++) sel[i] = row[i];
    } else {
        for (uint32_t i = 0; i < n_expert_used; i++) sel[i] = -1;
        for (uint32_t i = 0; i < n_expert; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (uint32_t j = 0; j < n_expert_used; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (uint32_t k = n_expert_used - 1; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = (int32_t)i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_expert_used; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < (int32_t)n_expert) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t i = 0; i < n_expert_used; i++) w[i] = w[i] / sum * expert_weight_scale;
}

__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= n_expert) return;
    const float *log = logits + (uint64_t)t * n_expert;
    float *prob = probs + (uint64_t)t * n_expert;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;
    __shared__ float sprob[384];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * n_expert_used;
        for (uint32_t j = 0; j < n_expert_used; j++) sel[j] = row[j];
    } else {
        for (uint32_t j = 0; j < n_expert_used; j++) sel[j] = -1;
        for (uint32_t e = 0; e < n_expert; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (uint32_t j = 0; j < n_expert_used; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (uint32_t k = n_expert_used - 1; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = (int32_t)e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < (int32_t)n_expert) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (uint32_t j = 0; j < n_expert_used; j++) w[j] = w[j] / sum * expert_weight_scale;
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode,
        uint32_t n_expert,
        uint32_t n_expert_used,
        float expert_weight_scale) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * n_expert;
    float *prob = probs + (uint64_t)t * n_expert;
    int32_t *sel = selected + (uint64_t)t * n_expert_used;
    float *w = weights + (uint64_t)t * n_expert_used;
    __shared__ float sprob[4][384];
    float local_prob[12];
    float local_score[12];

    for (uint32_t j = 0; j * 32u < n_expert; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < n_expert) {
            const float p = sqrtf(softplus_dev(log[e]));
            local_prob[j] = p;
            local_score[j] = p + (has_bias ? bias[e] : 0.0f);
            sprob[row_in_block][e] = p;
            prob[e] = p;
        } else {
            local_prob[j] = 0.0f;
            local_score[j] = -INFINITY;
        }
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
        int32_t tok = ds4_cuda_params_active ? ds4_cuda_dev_params.token :
                        (tokens ? tokens[t] : token_scalar);
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * n_expert_used;
            float sum = 0.0f;
            for (uint32_t j = 0; j < n_expert_used; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < (int32_t)n_expert) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            for (uint32_t j = 0; j < n_expert_used; j++) w[j] = w[j] / sum * expert_weight_scale;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    for (uint32_t k = 0; k < n_expert_used; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        for (uint32_t j = 0; j * 32u < n_expert; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        for (uint32_t j = 0; j * 32u < n_expert; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < n_expert_used; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        for (uint32_t j = 0; j < n_expert_used; j++) w[j] = w[j] / sum * expert_weight_scale;
    }
}

__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}

__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__global__ static void indexer_scores_wmma_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    if (tid >= 32u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
                const uint32_t r = i >> 4u;
                const uint32_t c = i & 15u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[16 * 128];
    __shared__ float c_sh[16 * 16];
    __shared__ float acc_sh[16 * 16];

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
            const uint32_t r = i >> 4u;
            const uint32_t token = tile_t + r;
            if (token < n_tokens) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
        const uint32_t r = i >> 4u;
        const uint32_t c = i & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 64u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 32u; i += 64u) {
                const uint32_t r = i >> 5u;
                const uint32_t c = i & 31u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[32 * 128];
    __shared__ float c_sh[2 * 16 * 16];
    __shared__ float acc_sh[2 * 16 * 16];

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 32u * 128u; i += 64u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 64u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma64_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 64u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 128u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 64u; i += 128u) {
                const uint32_t r = i >> 6u;
                const uint32_t c = i & 63u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[64 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float acc_sh[4 * 16 * 16];

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 64u * 128u; i += 128u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 128u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens ? weights[(uint64_t)token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void argmax_stage1_kernel(
        float *block_vals,
        uint32_t *block_idxs,
        const float *scores,
        uint32_t n_comp) {
    uint32_t tid = threadIdx.x;
    float best_v = -1.0e30f;
    uint32_t best_i = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + tid;
         i < n_comp;
         i += blockDim.x * gridDim.x) {
        const float v = scores[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }

    __shared__ float vals[256];
    __shared__ uint32_t idxs[256];
    vals[tid] = best_v;
    idxs[tid] = best_i;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1u; stride > 0; stride >>= 1u) {
        if (tid < stride) {
            const float ov = vals[tid + stride];
            const uint32_t oi = idxs[tid + stride];
            if (topk_score_better(ov, oi, vals[tid], idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_vals[blockIdx.x] = vals[0];
        block_idxs[blockIdx.x] = idxs[0];
    }
}

__global__ static void argmax_stage2_kernel(
        uint32_t *selected,
        const float *block_vals,
        const uint32_t *block_idxs,
        uint32_t n_blocks) {
    uint32_t tid = threadIdx.x;
    float best_v = -1.0e30f;
    uint32_t best_i = 0;
    for (uint32_t i = tid; i < n_blocks; i += blockDim.x) {
        const float v = block_vals[i];
        const uint32_t idx = block_idxs[i];
        if (topk_score_better(v, idx, best_v, best_i)) {
            best_v = v;
            best_i = idx;
        }
    }

    __shared__ float vals[256];
    __shared__ uint32_t idxs[256];
    vals[tid] = best_v;
    idxs[tid] = best_i;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1u; stride > 0; stride >>= 1u) {
        if (tid < stride) {
            const float ov = vals[tid + stride];
            const uint32_t oi = idxs[tid + stride];
            if (topk_score_better(ov, oi, vals[tid], idxs[tid])) {
                vals[tid] = ov;
                idxs[tid] = oi;
            }
        }
        __syncthreads();
    }

    if (tid == 0) selected[0] = idxs[0];
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

static int ds4_gpu_embed_token_hc_cpu_fallback(
        ds4_gpu_tensor *out_hc,
        const char *wptr,
        uint32_t token,
        uint32_t n_vocab,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!out_hc || !wptr || n_embd == 0 || n_hc == 0) return 0;
    if (token >= n_vocab) token = 0;

    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    uint16_t *row = (uint16_t *)malloc((size_t)row_bytes);
    float *out = (float *)malloc((size_t)((uint64_t)n_embd * n_hc * sizeof(float)));
    if (!row || !out) {
        free(row);
        free(out);
        return 0;
    }

    const uint64_t row_off = (uint64_t)token * n_embd * sizeof(uint16_t);
    cudaError_t ce = cudaMemcpy(row, wptr + row_off, (size_t)row_bytes,
                                cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: CPU embed row read failed: %s\n", cudaGetErrorString(ce));
        cudaGetLastError();
        free(row);
        free(out);
        return 0;
    }

    for (uint32_t h = 0; h < n_hc; ++h) {
        for (uint32_t e = 0; e < n_embd; ++e) {
            __half hv;
            memcpy(&hv, &row[e], sizeof(hv));
            out[(uint64_t)h * n_embd + e] = __half2float(hv);
        }
    }

    ce = cudaMemcpy(out_hc->ptr, out,
                    (size_t)((uint64_t)n_embd * n_hc * sizeof(float)),
                    cudaMemcpyHostToDevice);

    free(row);
    free(out);

    if (ce != cudaSuccess) {
        fprintf(stderr, "ds4: CPU embed output upload failed: %s\n", cudaGetErrorString(ce));
        cudaGetLastError();
        return 0;
    }
    return 1;
}

extern "C" int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (token >= n_vocab) token = 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    if (getenv("DS4_CUDA_PP_EMBED_DEBUG") != NULL) {
        ds4_debug_ptr_attr_raw("embed token wptr", wptr);
        ds4_debug_ptr_attr_raw("embed token out_hc", out_hc->ptr);
        /* Read actual row accessed by kernel (token * n_embd) */
        const uint64_t tok = token < n_vocab ? token : 0;
        const uint64_t row_bytes = tok * (uint64_t)n_embd * sizeof(uint16_t);
        __half v0, v_last;
        cudaError_t ce = cudaMemcpy(&v0, wptr + row_bytes, sizeof(v0), cudaMemcpyDeviceToHost);
        cudaError_t ce_last = cudaMemcpy(&v_last,
                wptr + row_bytes + (uint64_t)(n_embd - 1) * sizeof(uint16_t),
                sizeof(v_last), cudaMemcpyDeviceToHost);
        fprintf(stderr,
                "ds4: embed row read token=%u row_off=%llu first=%s/%g last=%s/%g\n",
                token,
                (unsigned long long)row_bytes,
                cudaGetErrorString(ce), ce == cudaSuccess ? __half2float(v0) : 0.0f,
                cudaGetErrorString(ce_last), ce_last == cudaSuccess ? __half2float(v_last) : 0.0f);
        if (ce != cudaSuccess || ce_last != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
    }
    if (getenv("DS4_CUDA_PP_CPU_EMBED") != NULL) {
        return ds4_gpu_embed_token_hc_cpu_fallback(
            out_hc, wptr, token, n_vocab, n_embd, n_hc);
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    embed_token_hc_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
        (float *)out_hc->ptr, (const unsigned short *)wptr,
        token, n_vocab, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_DIRECT_ONE") == NULL) {
        indexer_score_one_direct_kernel<<<n_comp, 128, 0, ds4_decode_stream()>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u &&
        getenv("DS4_CUDA_NO_INDEXER_WMMA") == NULL) {
        if (getenv("DS4_CUDA_NO_INDEXER_WMMA128") == NULL) {
            dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma128_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, n_tokens, pos0, n_head,
                                                         head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA64") == NULL) {
            dim3 grid((n_comp + 63u) / 64u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma64_kernel<<<grid, 128, 0, ds4_decode_stream()>>>((float *)scores->ptr,
                                                        (const float *)q->ptr,
                                                        (const float *)weights->ptr,
                                                        (const float *)index_comp->ptr,
                                                        n_comp, n_tokens, pos0, n_head,
                                                        head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma64 launch");
        } else if (getenv("DS4_CUDA_NO_INDEXER_WMMA32") == NULL) {
            dim3 grid((n_comp + 31u) / 32u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma32_kernel<<<grid, 64, 0, ds4_decode_stream()>>>((float *)scores->ptr,
                                                       (const float *)q->ptr,
                                                       (const float *)weights->ptr,
                                                       (const float *)index_comp->ptr,
                                                       n_comp, n_tokens, pos0, n_head,
                                                       head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma32 launch");
        } else {
            dim3 grid((n_comp + 15u) / 16u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma_kernel<<<grid, 32, 0, ds4_decode_stream()>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma launch");
        }
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 512u && n_comp <= 1024u &&
        getenv("DS4_CUDA_NO_TOPK1024") == NULL) {
        indexer_topk_1024_kernel<<<n_tokens, 1024, 0, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024, 0, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024, 0, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK8192") == NULL) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024, 0, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 512u && getenv("DS4_CUDA_NO_TOPK2048") == NULL &&
        getenv("DS4_CUDA_NO_TOPK_CHUNKED") == NULL) {
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024, 0, ds4_decode_stream()>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024, 0, ds4_decode_stream()>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    DS4_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024, 0, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1, 0, ds4_decode_stream()>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp) {
    if (!selected || !scores || n_comp == 0 ||
        scores->bytes < (uint64_t)n_comp * sizeof(float) ||
        selected->bytes < sizeof(uint32_t)) {
        return 0;
    }

    const uint32_t threads = 256u;
    uint32_t blocks = (n_comp + threads - 1u) / threads;
    if (blocks == 0) blocks = 1;
    if (blocks > 1024u) blocks = 1024u;

    const uint64_t vals_bytes = (uint64_t)blocks * sizeof(float);
    const uint64_t idx_offset = (vals_bytes + 15u) & ~15ull;
    const uint64_t idx_bytes = (uint64_t)blocks * sizeof(uint32_t);
    uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(idx_offset + idx_bytes, "argmax");
    if (!scratch) return 0;

    float *block_vals = (float *)scratch;
    uint32_t *block_idxs = (uint32_t *)(scratch + idx_offset);
    cudaStream_t s = ds4_decode_stream();
    argmax_stage1_kernel<<<blocks, threads, 0, s>>>(
            block_vals,
            block_idxs,
            (const float *)scores->ptr,
            n_comp);
    if (!cuda_ok(cudaGetLastError(), "argmax stage1 launch")) return 0;

    argmax_stage2_kernel<<<1, threads, 0, s>>>(
            (uint32_t *)selected->ptr,
            block_vals,
            block_idxs,
            blocks);
    return cuda_ok(cudaGetLastError(), "argmax stage2 launch");
}

extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256, 0, ds4_decode_stream()>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}
static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(g_cublas,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256, 0, ds4_decode_stream()>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: cuBLAS q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, ds4_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 warp launch");
    }
    if (getenv("DS4_CUDA_NO_Q8_BATCH_WARP") == NULL && blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256, 0, ds4_decode_stream()>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

/* Row-parallel (TP) Q8_0 matmul: accumulate only over input blocks
 * [block_start, block_end) of a full Q8_0 weight. Decode-only (n_tok==1).
 * add_to!=0 sums into out. Summing every rank's [b0,b1) partial via all-reduce
 * reproduces the full result. block_start/end are 32-element-block indices, so
 * the input split stays quantization-aligned. */
extern "C" int ds4_gpu_matmul_q8_0_brange_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        uint64_t block_start, uint64_t block_end, int add_to,
        const ds4_gpu_tensor *x) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (block_end > blocks || block_start > block_end) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < in_dim * sizeof(float) || out->bytes < out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0_brange");
    if (!wptr) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 brange prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, 1u, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, ds4_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_brange quantize launch")) return 0;
    matmul_q8_0_preq_warp8_brange_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
            (float *)out->ptr, reinterpret_cast<const unsigned char *>(wptr),
            xq, xscale, in_dim, out_dim, blocks, block_start, block_end, add_to, use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_brange launch");
}

/* ---- Tensor-Parallel shard matmuls (read the resident g_tp_shards entry) ----
 * The TP decode encode calls these instead of ds4_gpu_matmul_q8_0_tensor for the
 * sharded weights: they resolve the CURRENT device's shard for (model_map,
 * weight_offset) and run the validated Q8_0 kernel with the shard's own dims.
 * Underlying kernels + the shard accessor are already jig-validated; these are
 * the thin tensor-level plumbing the encode needs. */

/* Column-parallel: out[out_dim/k] = colshard(W) . x (full input, in_dim). Each
 * rank produces its own output-row slice; no all-reduce (the slices are disjoint
 * outputs that feed a row-parallel matmul next, Megatron-style). */
extern "C" int ds4_gpu_tp_col_matmul_tensor(
        ds4_gpu_tensor *out, const void *model_map,
        uint64_t weight_offset, const ds4_gpu_tensor *x) {
    if (!out || !x || !model_map) return 0;
    int kind = -1; uint64_t out_rows = 0, in_dim = 0, blocks = 0;
    const char *wptr = cuda_tp_shard_ptr(model_map, weight_offset, &kind, &out_rows, &in_dim, &blocks);
    if (!wptr || kind != DS4_TP_SHARD_COL) return 0;
    if (x->bytes < in_dim * sizeof(float) || out->bytes < out_rows * sizeof(float)) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    void *tmp = cuda_tmp_alloc(scale_offset + blocks * sizeof(float), "tp col prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp; float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, ds4_decode_stream()>>>(
            xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "tp_col quantize")) return 0;
    matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_rows + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
            (float *)out->ptr, reinterpret_cast<const unsigned char *>(wptr),
            xq, xscale, in_dim, out_rows, blocks, use_dp4a);
    return cuda_ok(cudaGetLastError(), "tp_col matmul");
}

/* Row-parallel: out[out_dim] = rowshard(W) . x_slice (this rank's input slice,
 * in_dim/k = shard.in_dim elements). add_to!=0 accumulates into out. The caller
 * all-reduces the per-rank partials to get the full result. */
extern "C" int ds4_gpu_tp_row_matmul_tensor(
        ds4_gpu_tensor *out, const void *model_map,
        uint64_t weight_offset, int add_to, const ds4_gpu_tensor *x) {
    if (!out || !x || !model_map) return 0;
    int kind = -1; uint64_t out_dim = 0, in_slice = 0, blocks = 0;
    const char *wptr = cuda_tp_shard_ptr(model_map, weight_offset, &kind, &out_dim, &in_slice, &blocks);
    if (!wptr || kind != DS4_TP_SHARD_ROW) return 0;
    if (x->bytes < in_slice * sizeof(float) || out->bytes < out_dim * sizeof(float)) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    void *tmp = cuda_tmp_alloc(scale_offset + blocks * sizeof(float), "tp row prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp; float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, ds4_decode_stream()>>>(
            xq, xscale, (const float *)x->ptr, in_slice, blocks);
    if (!cuda_ok(cudaGetLastError(), "tp_row quantize")) return 0;
    /* the packed row-shard is a dense [out_dim x blocks] Q8_0 weight; a full
     * matmul over its blocks (with optional add_to) gives this rank's partial */
    matmul_q8_0_preq_warp8_brange_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
            (float *)out->ptr, reinterpret_cast<const unsigned char *>(wptr),
            xq, xscale, in_slice, out_dim, blocks, 0, blocks, add_to, use_dp4a);
    return cuda_ok(cudaGetLastError(), "tp_row matmul");
}

/* ---- Tensor-Parallel validation/benchmark jig (Q8_0 matmul) ----
 * Reusable, TP-degree-parameterized harness: for a real Q8_0 weight + input,
 * compute the full single-GPU result (golden), then both column-parallel and
 * row-parallel TP versions across k GPUs, comparing numerically (rel error vs
 * golden) and timing each. Reshards weights to the group GPUs the way the real
 * integration will (contiguous output-row shards for column-parallel; packed
 * input-block shards for row-parallel) so the jig exercises the actual data
 * movement. Same harness serves TP2, TP4, etc. Env-gated. */
static double tp_jig_now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0+ts.tv_nsec/1e6; }

extern "C" int ds4_gpu_tp_matmul_jig(
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, int k) {
    int ndev = 0; (void)cudaGetDeviceCount(&ndev);
    if (k < 2 || k > ndev || k > 8) { fprintf(stderr, "ds4: TP jig: need 2..%d GPUs (k=%d)\n", ndev, k); return 0; }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t wbytes = out_dim * blocks * 34;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    if ((out_dim % (uint64_t)k) != 0 || (blocks % (uint64_t)k) != 0) {
        fprintf(stderr, "ds4: TP jig: out_dim(%llu) or blocks(%llu) not divisible by k=%d; skipping\n",
                (unsigned long long)out_dim, (unsigned long long)blocks, k);
        return 0;
    }
    const int dp4a = cuda_q8_use_dp4a();
    const int iters = 30;
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;

    /* dev0: input ramp, prequant, resident full weight, golden full matmul + timing */
    if (cudaSetDevice(0) != cudaSuccess) return 0;
    const char *wfull = cuda_model_range_ptr(model_map, weight_offset, wbytes, "tp_jig");
    if (!wfull) { fprintf(stderr, "ds4: TP jig: weight not resident on dev0\n"); return 0; }
    float *x = NULL, *outF = NULL, *outChk = NULL; int8_t *xqA = NULL; float *xsA = NULL;
    std::vector<float> hx(in_dim);
    for (uint64_t i = 0; i < in_dim; i++) hx[i] = sinf((float)i * 0.011f) * 0.5f;
    ok = ok && cudaMalloc(&x, in_dim*sizeof(float))==cudaSuccess && cudaMalloc(&xqA, blocks*32)==cudaSuccess
            && cudaMalloc(&xsA, blocks*sizeof(float))==cudaSuccess && cudaMalloc(&outF, out_dim*sizeof(float))==cudaSuccess
            && cudaMalloc(&outChk, out_dim*sizeof(float))==cudaSuccess;
    if (ok) ok = cudaMemcpy(x, hx.data(), in_dim*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess;
    if (ok) { quantize_q8_0_f32_kernel<<<(unsigned)blocks,32>>>(xqA, xsA, x, in_dim, blocks); ok = cudaGetLastError()==cudaSuccess; }
    double ms_full = 0.0;
    if (ok) {
        cudaDeviceSynchronize();
        double t0 = tp_jig_now_ms();
        for (int it=0; it<iters && ok; it++) {
            matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim+7u)/8u,256>>>(outF,(const unsigned char*)wfull,xqA,xsA,in_dim,out_dim,blocks,dp4a);
        }
        ok = cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess;
        ms_full = (tp_jig_now_ms()-t0)/iters;
    }
    std::vector<float> hF(out_dim);
    if (ok) ok = cudaMemcpy(hF.data(), outF, out_dim*sizeof(float), cudaMemcpyDeviceToHost)==cudaSuccess;

    /* per-rank scratch (k<=8) */
    char *wsh[8]={0}; int8_t *xqr[8]={0}; float *xsr[8]={0}; float *outr[8]={0}; cudaStream_t st[8]={0};

    /* ===== COLUMN-PARALLEL: rank r owns output rows [r*rpr,(r+1)*rpr) ===== */
    const uint64_t rpr = out_dim / (uint64_t)k;
    double ms_col = 0.0, rel_col = 0.0;
    if (ok) {
        for (int r=0; r<k && ok; r++) {
            ok = cudaSetDevice(r)==cudaSuccess;
            ok = ok && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && cudaMalloc(&wsh[r], rpr*blocks*34)==cudaSuccess
                    && cudaMalloc(&xqr[r], blocks*32)==cudaSuccess
                    && cudaMalloc(&xsr[r], blocks*sizeof(float))==cudaSuccess
                    && cudaMalloc(&outr[r], rpr*sizeof(float))==cudaSuccess;
            /* reshard: contiguous weight rows + replicated input (one-time) */
            ok = ok && cudaMemcpyPeer(wsh[r], r, wfull + (uint64_t)r*rpr*blocks*34, 0, rpr*blocks*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xqr[r], r, xqA, 0, blocks*32)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xsr[r], r, xsA, 0, blocks*sizeof(float))==cudaSuccess;
        }
        for (int r=0;r<k;r++){ cudaSetDevice(r); cudaDeviceSynchronize(); }
        double t0 = tp_jig_now_ms();
        for (int it=0; it<iters && ok; it++)
            for (int r=0;r<k;r++){ cudaSetDevice(r);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)rpr+7u)/8u,256,0,st[r]>>>(outr[r],(const unsigned char*)wsh[r],xqr[r],xsr[r],in_dim,rpr,blocks,dp4a); }
        for (int r=0;r<k;r++){ cudaSetDevice(r); if (cudaStreamSynchronize(st[r])!=cudaSuccess) ok=0; }
        ms_col = (tp_jig_now_ms()-t0)/iters;
        /* gather + compare */
        cudaSetDevice(0);
        for (int r=0;r<k && ok;r++) ok = cudaMemcpyPeer(outChk + (uint64_t)r*rpr, 0, outr[r], r, rpr*sizeof(float))==cudaSuccess;
        cudaDeviceSynchronize();
        std::vector<float> hC(out_dim);
        if (ok) ok = cudaMemcpy(hC.data(), outChk, out_dim*sizeof(float), cudaMemcpyDeviceToHost)==cudaSuccess;
        double maxabs=0, scale=0; for (uint64_t i=0;i<out_dim;i++){ double a=fabs((double)hF[i]); if(a>scale)scale=a; double d=fabs((double)hF[i]-(double)hC[i]); if(d>maxabs)maxabs=d; }
        if (scale<1e-12) scale=1e-12; rel_col = maxabs/scale;
        for (int r=0;r<k;r++){ cudaSetDevice(r); if(wsh[r])cudaFree(wsh[r]); if(xqr[r])cudaFree(xqr[r]); if(xsr[r])cudaFree(xsr[r]); if(outr[r])cudaFree(outr[r]); if(st[r])cudaStreamDestroy(st[r]); wsh[r]=0;xqr[r]=0;xsr[r]=0;outr[r]=0;st[r]=0; }
    }

    /* ===== ROW-PARALLEL: rank r owns input blocks [r*bpr,(r+1)*bpr), all rows ===== */
    const uint64_t bpr = blocks / (uint64_t)k;
    double ms_row = 0.0, rel_row = 0.0;
    char *packtmp = NULL;
    int devs[8]; for (int r=0;r<k;r++) devs[r]=r;
    if (ok) {
        cudaSetDevice(0);
        ok = cudaMalloc(&packtmp, out_dim*bpr*34)==cudaSuccess; /* dev0 staging for the 2D pack */
        for (int r=0; r<k && ok; r++) {
            /* pack rank r's block-slice of every row on dev0, then peer-copy to dev r */
            ok = cudaMemcpy2D(packtmp, bpr*34, wfull + (uint64_t)r*bpr*34, blocks*34, bpr*34, out_dim, cudaMemcpyDeviceToDevice)==cudaSuccess;
            ok = ok && cudaSetDevice(r)==cudaSuccess;
            ok = ok && cudaMalloc(&wsh[r], out_dim*bpr*34)==cudaSuccess
                    && cudaMalloc(&xqr[r], bpr*32)==cudaSuccess
                    && cudaMalloc(&xsr[r], bpr*sizeof(float))==cudaSuccess
                    && cudaMalloc(&outr[r], out_dim*sizeof(float))==cudaSuccess;
            ok = ok && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && cudaMemcpyPeer(wsh[r], r, packtmp, 0, out_dim*bpr*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xqr[r], r, xqA + (uint64_t)r*bpr*32, 0, bpr*32)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xsr[r], r, xsA + (uint64_t)r*bpr, 0, bpr*sizeof(float))==cudaSuccess;
            cudaSetDevice(0);
        }
        for (int r=0;r<k;r++){ cudaSetDevice(r); cudaDeviceSynchronize(); }
        double t0 = tp_jig_now_ms();
        for (int it=0; it<iters && ok; it++) {
            for (int r=0;r<k;r++){ cudaSetDevice(r);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim+7u)/8u,256,0,st[r]>>>(outr[r],(const unsigned char*)wsh[r],xqr[r],xsr[r],bpr*32,out_dim,bpr,dp4a); }
            for (int r=0;r<k;r++){ cudaSetDevice(r); cudaStreamSynchronize(st[r]); }
            /* sum partials across ranks via the TP all-reduce (result on every rank) */
            ok = ds4_gpu_tp_all_reduce_f32(devs, k, outr, (uint32_t)out_dim);
        }
        ms_row = (tp_jig_now_ms()-t0)/iters;
        cudaSetDevice(0);
        std::vector<float> hR(out_dim);
        if (ok) ok = cudaMemcpy(hR.data(), outr[0], out_dim*sizeof(float), cudaMemcpyDeviceToHost)==cudaSuccess;
        double maxabs=0, scale=0; for (uint64_t i=0;i<out_dim;i++){ double a=fabs((double)hF[i]); if(a>scale)scale=a; double d=fabs((double)hF[i]-(double)hR[i]); if(d>maxabs)maxabs=d; }
        if (scale<1e-12) scale=1e-12; rel_row = maxabs/scale;
        cudaSetDevice(0); if (packtmp) cudaFree(packtmp);
        for (int r=0;r<k;r++){ cudaSetDevice(r); if(wsh[r])cudaFree(wsh[r]); if(xqr[r])cudaFree(xqr[r]); if(xsr[r])cudaFree(xsr[r]); if(outr[r])cudaFree(outr[r]); if(st[r])cudaStreamDestroy(st[r]); }
    }

    fprintf(stderr,
        "ds4: TP jig k=%d in=%llu out=%llu | full=%.4f ms | col %s rel=%.2e %.4f ms (%.2fx) | row %s rel=%.2e %.4f ms (%.2fx)\n",
        k, (unsigned long long)in_dim, (unsigned long long)out_dim, ms_full,
        rel_col<1e-3?"OK":"BAD", rel_col, ms_col, ms_col>0?ms_full/ms_col:0.0,
        rel_row<1e-3?"OK":"BAD", rel_row, ms_row, ms_row>0?ms_full/ms_row:0.0);

    cudaSetDevice(0); cudaFree(x); cudaFree(xqA); cudaFree(xsA); cudaFree(outF); cudaFree(outChk);
    (void)cudaSetDevice(saved);
    return (ok && rel_col<1e-3 && rel_row<1e-3) ? 1 : 0;
}

/* TP jig: a full SwiGLU MLP block (the shared-expert FFN) split across k GPUs vs
 * the single-GPU golden. This is the Megatron MLP pattern end-to-end:
 *   gate,up = column-parallel matmul (each rank owns ff_dim/k output cols)
 *   mid     = swiglu(gate,up) elementwise on each rank's slice (no comm)
 *   down    = row-parallel matmul (each rank's mid slice = its input blocks)
 *   out     = ONE all-reduce of the n_embd partials.
 * Validates that the col-parallel output feeds row-parallel with zero intermediate
 * comm. All three weights are Q8_0. Reports rel error vs golden + timing. */
extern "C" int ds4_gpu_tp_ffn_jig(
        const void *model_map, uint64_t model_size,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        uint64_t in_dim, uint64_t ff_dim, float clamp, int k) {
    int ndev = 0; (void)cudaGetDeviceCount(&ndev);
    if (k < 2 || k > ndev || k > 8) { fprintf(stderr, "ds4: TP FFN jig: need 2..%d GPUs (k=%d)\n", ndev, k); return 0; }
    const uint64_t inb = (in_dim + 31) / 32;       /* gate/up input blocks (n_embd) */
    const uint64_t ffb = (ff_dim + 31) / 32;       /* down input blocks (ff_dim)    */
    const uint64_t gate_wb = ff_dim * inb * 34, down_wb = in_dim * ffb * 34;
    if ((ff_dim % (uint64_t)k) != 0 || (ffb % (uint64_t)k) != 0) {
        fprintf(stderr, "ds4: TP FFN jig: ff_dim(%llu)/blocks(%llu) not divisible by k=%d; skip\n",
                (unsigned long long)ff_dim, (unsigned long long)ffb, k); return 0;
    }
    const int dp4a = cuda_q8_use_dp4a();
    const int iters = 30;
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;

    if (cudaSetDevice(0) != cudaSuccess) return 0;
    const char *gW = cuda_model_range_ptr(model_map, gate_off, gate_wb, "tp_ffn_gate");
    const char *uW = cuda_model_range_ptr(model_map, up_off,   gate_wb, "tp_ffn_up");
    const char *dW = cuda_model_range_ptr(model_map, down_off, down_wb, "tp_ffn_down");
    if (!gW || !uW || !dW) { fprintf(stderr, "ds4: TP FFN jig: weights not resident on dev0\n"); return 0; }

    /* golden full FFN on dev0 */
    float *x=NULL,*g=NULL,*u=NULL,*mid=NULL,*outF=NULL; int8_t *xq=NULL,*midq=NULL; float *xs=NULL,*mids=NULL;
    std::vector<float> hx(in_dim); for (uint64_t i=0;i<in_dim;i++) hx[i]=sinf((float)i*0.009f)*0.4f;
    ok = ok && cudaMalloc(&x,in_dim*4)==cudaSuccess && cudaMalloc(&xq,inb*32)==cudaSuccess && cudaMalloc(&xs,inb*4)==cudaSuccess
            && cudaMalloc(&g,ff_dim*4)==cudaSuccess && cudaMalloc(&u,ff_dim*4)==cudaSuccess && cudaMalloc(&mid,ff_dim*4)==cudaSuccess
            && cudaMalloc(&midq,ffb*32)==cudaSuccess && cudaMalloc(&mids,ffb*4)==cudaSuccess && cudaMalloc(&outF,in_dim*4)==cudaSuccess;
    if (ok) ok = cudaMemcpy(x,hx.data(),in_dim*4,cudaMemcpyHostToDevice)==cudaSuccess;
    if (ok) { quantize_q8_0_f32_kernel<<<(unsigned)inb,32>>>(xq,xs,x,in_dim,inb);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)ff_dim+7u)/8u,256>>>(g,(const unsigned char*)gW,xq,xs,in_dim,ff_dim,inb,dp4a);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)ff_dim+7u)/8u,256>>>(u,(const unsigned char*)uW,xq,xs,in_dim,ff_dim,inb,dp4a);
              swiglu_kernel<<<((unsigned)ff_dim+255u)/256u,256>>>(mid,g,u,(uint32_t)ff_dim,clamp,1.0f);
              quantize_q8_0_f32_kernel<<<(unsigned)ffb,32>>>(midq,mids,mid,ff_dim,ffb);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)in_dim+7u)/8u,256>>>(outF,(const unsigned char*)dW,midq,mids,ff_dim,in_dim,ffb,dp4a);
              ok = cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess; }
    std::vector<float> hF(in_dim);
    if (ok) ok = cudaMemcpy(hF.data(),outF,in_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess;

    /* TP across k GPUs */
    const uint64_t ffpr = ff_dim/(uint64_t)k, bpr = ffb/(uint64_t)k;
    char *gsh[8]={0},*ush[8]={0},*dsh[8]={0}; int8_t *xqr[8]={0},*mqr[8]={0}; float *xsr[8]={0},*msr[8]={0};
    float *gr[8]={0},*ur[8]={0},*mr[8]={0},*pr[8]={0}; cudaStream_t st[8]={0};
    char *packtmp=NULL; int devs[8]; for(int r=0;r<k;r++)devs[r]=r;
    double ms_tp=0.0, rel=1.0;
    if (ok) {
        cudaSetDevice(0); ok = cudaMalloc(&packtmp,in_dim*bpr*34)==cudaSuccess;
        for (int r=0;r<k && ok;r++) {
            /* down row-shard: pack input blocks [r*bpr,(r+1)*bpr) of every out row on dev0, peer-copy */
            ok = cudaMemcpy2D(packtmp,bpr*34,dW + (uint64_t)r*bpr*34,ffb*34,bpr*34,in_dim,cudaMemcpyDeviceToDevice)==cudaSuccess;
            ok = ok && cudaSetDevice(r)==cudaSuccess && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && cudaMalloc(&gsh[r],ffpr*inb*34)==cudaSuccess && cudaMalloc(&ush[r],ffpr*inb*34)==cudaSuccess
                    && cudaMalloc(&dsh[r],in_dim*bpr*34)==cudaSuccess
                    && cudaMalloc(&xqr[r],inb*32)==cudaSuccess && cudaMalloc(&xsr[r],inb*4)==cudaSuccess
                    && cudaMalloc(&gr[r],ffpr*4)==cudaSuccess && cudaMalloc(&ur[r],ffpr*4)==cudaSuccess && cudaMalloc(&mr[r],ffpr*4)==cudaSuccess
                    && cudaMalloc(&mqr[r],bpr*32)==cudaSuccess && cudaMalloc(&msr[r],bpr*4)==cudaSuccess && cudaMalloc(&pr[r],in_dim*4)==cudaSuccess;
            /* col-shards of gate/up (contiguous output rows), replicated input, down row-shard */
            ok = ok && cudaMemcpyPeer(gsh[r],r,gW + (uint64_t)r*ffpr*inb*34,0,ffpr*inb*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(ush[r],r,uW + (uint64_t)r*ffpr*inb*34,0,ffpr*inb*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(dsh[r],r,packtmp,0,in_dim*bpr*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xqr[r],r,xq,0,inb*32)==cudaSuccess && cudaMemcpyPeer(xsr[r],r,xs,0,inb*4)==cudaSuccess;
            cudaSetDevice(0);
        }
        for(int r=0;r<k;r++){cudaSetDevice(r);cudaDeviceSynchronize();}
        double t0=tp_jig_now_ms();
        for (int it=0; it<iters && ok; it++) {
            for (int r=0;r<k;r++){ cudaSetDevice(r);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)ffpr+7u)/8u,256,0,st[r]>>>(gr[r],(const unsigned char*)gsh[r],xqr[r],xsr[r],in_dim,ffpr,inb,dp4a);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)ffpr+7u)/8u,256,0,st[r]>>>(ur[r],(const unsigned char*)ush[r],xqr[r],xsr[r],in_dim,ffpr,inb,dp4a);
                swiglu_kernel<<<((unsigned)ffpr+255u)/256u,256,0,st[r]>>>(mr[r],gr[r],ur[r],(uint32_t)ffpr,clamp,1.0f);
                quantize_q8_0_f32_kernel<<<(unsigned)bpr,32,0,st[r]>>>(mqr[r],msr[r],mr[r],ffpr,bpr);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)in_dim+7u)/8u,256,0,st[r]>>>(pr[r],(const unsigned char*)dsh[r],mqr[r],msr[r],ffpr,in_dim,bpr,dp4a);
            }
            for(int r=0;r<k;r++){cudaSetDevice(r);cudaStreamSynchronize(st[r]);}
            ok = ds4_gpu_tp_all_reduce_f32(devs,k,pr,(uint32_t)in_dim);
        }
        ms_tp=(tp_jig_now_ms()-t0)/iters;
        cudaSetDevice(0); std::vector<float> hT(in_dim);
        if (ok) ok = cudaMemcpy(hT.data(),pr[0],in_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess;
        double maxabs=0,scale=0; for(uint64_t i=0;i<in_dim;i++){double a=fabs((double)hF[i]);if(a>scale)scale=a;double d=fabs((double)hF[i]-(double)hT[i]);if(d>maxabs)maxabs=d;}
        if(scale<1e-12)scale=1e-12; rel=maxabs/scale;
        cudaSetDevice(0); if(packtmp)cudaFree(packtmp);
        for(int r=0;r<k;r++){cudaSetDevice(r);
            if(gsh[r])cudaFree(gsh[r]);if(ush[r])cudaFree(ush[r]);if(dsh[r])cudaFree(dsh[r]);
            if(xqr[r])cudaFree(xqr[r]);if(xsr[r])cudaFree(xsr[r]);if(gr[r])cudaFree(gr[r]);if(ur[r])cudaFree(ur[r]);
            if(mr[r])cudaFree(mr[r]);if(mqr[r])cudaFree(mqr[r]);if(msr[r])cudaFree(msr[r]);if(pr[r])cudaFree(pr[r]);
            if(st[r])cudaStreamDestroy(st[r]);}
    }
    fprintf(stderr, "ds4: TP FFN-block jig k=%d in=%llu ff=%llu | %s rel=%.2e tp=%.4f ms (1 all-reduce/block)\n",
            k, (unsigned long long)in_dim, (unsigned long long)ff_dim, rel<1e-3?"PASS":"FAIL", rel, ms_tp);
    cudaSetDevice(0); cudaFree(x);cudaFree(xq);cudaFree(xs);cudaFree(g);cudaFree(u);cudaFree(mid);cudaFree(midq);cudaFree(mids);cudaFree(outF);
    (void)cudaSetDevice(saved);
    return (ok && rel<1e-3) ? 1 : 0;
}

/* TP grouped O-projection jig: de-risks the attention-TP O-proj split before
 * integration. The MLA O-proj is grouped-LoRA: output_a maps each of n_groups
 * groups' group_dim heads -> a rank-vector (attn_low=[n_groups,rank]); output_b
 * maps [n_groups*rank -> n_embd]. The TP split gives each rank a CONTIGUOUS set of
 * gpr=n_groups/k groups: output_a is column-sharded (its owned output rows
 * [r*gpr*rank,(r+1)*gpr*rank), contiguous) and output_b is row-sharded (its owned
 * input-block range), so rank r computes its groups' attn_low + a partial n_embd;
 * the partials all-reduce to the full O-proj output. This is the same col->row->
 * all-reduce pattern as the FFN-block jig, specialized to the grouped output_a. */
extern "C" int ds4_gpu_tp_oproj_jig(
        const void *model_map, uint64_t model_size,
        uint64_t out_a_off, uint64_t out_b_off,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t n_embd, int k) {
    int ndev = 0; (void)cudaGetDeviceCount(&ndev);
    if (k < 2 || k > ndev || k > 8) { fprintf(stderr, "ds4: TP O-proj jig: need 2..%d GPUs (k=%d)\n", ndev, k); return 0; }
    if ((n_groups % (uint32_t)k) != 0) {
        fprintf(stderr, "ds4: TP O-proj jig: n_groups(%u) not divisible by k=%d; skip\n", n_groups, k); return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;   /* output_a input blocks (group_dim) */
    const uint64_t blocks_b = (low_dim + 31) / 32;     /* output_b input blocks (low_dim)    */
    if ((blocks_b % (uint64_t)k) != 0) {
        fprintf(stderr, "ds4: TP O-proj jig: low_dim blocks(%llu) not divisible by k=%d; skip\n",
                (unsigned long long)blocks_b, k); return 0;
    }
    const uint64_t out_a_wb = low_dim * blocks_a * 34;
    const uint64_t out_b_wb = n_embd * blocks_b * 34;
    const int dp4a = cuda_q8_use_dp4a();
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;
    if (cudaSetDevice(0) != cudaSuccess) return 0;
    const char *aW = cuda_model_range_ptr(model_map, out_a_off, out_a_wb, "tp_oproj_a");
    const char *bW = cuda_model_range_ptr(model_map, out_b_off, out_b_wb, "tp_oproj_b");
    if (!aW || !bW) { fprintf(stderr, "ds4: TP O-proj jig: weights not resident on dev0\n"); return 0; }

    /* golden full O-proj on dev0: heads -> grouped output_a -> attn_low -> output_b -> out */
    float *heads=NULL,*low=NULL,*outF=NULL; int8_t *xq=NULL,*lowq=NULL; float *xs=NULL,*lows=NULL;
    const uint64_t heads_n = (uint64_t)n_groups * group_dim;
    std::vector<float> hh(heads_n); for (uint64_t i=0;i<heads_n;i++) hh[i]=sinf((float)i*0.007f)*0.5f;
    ok = ok && cudaMalloc(&heads,heads_n*4)==cudaSuccess
            && cudaMalloc(&xq,(uint64_t)n_groups*blocks_a*32)==cudaSuccess && cudaMalloc(&xs,(uint64_t)n_groups*blocks_a*4)==cudaSuccess
            && cudaMalloc(&low,low_dim*4)==cudaSuccess
            && cudaMalloc(&lowq,blocks_b*32)==cudaSuccess && cudaMalloc(&lows,blocks_b*4)==cudaSuccess
            && cudaMalloc(&outF,n_embd*4)==cudaSuccess;
    if (ok) ok = cudaMemcpy(heads,hh.data(),heads_n*4,cudaMemcpyHostToDevice)==cudaSuccess;
    if (ok) { dim3 qg((unsigned)blocks_a,(unsigned)n_groups,1);
              quantize_q8_0_f32_kernel<<<qg,32>>>(xq,xs,heads,group_dim,blocks_a);
              grouped_q8_0_a_preq_warp8_kernel<<<((unsigned)low_dim+7u)/8u,256>>>(low,(const unsigned char*)aW,xq,xs,group_dim,rank,n_groups,1,blocks_a,dp4a);
              quantize_q8_0_f32_kernel<<<(unsigned)blocks_b,32>>>(lowq,lows,low,low_dim,blocks_b);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)n_embd+7u)/8u,256>>>(outF,(const unsigned char*)bW,lowq,lows,low_dim,n_embd,blocks_b,dp4a);
              ok = cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess; }
    std::vector<float> hF(n_embd);
    if (ok) ok = cudaMemcpy(hF.data(),outF,n_embd*4,cudaMemcpyDeviceToHost)==cudaSuccess;

    /* TP across k GPUs: gpr groups/rank; output_a col-shard (owned output rows),
       output_b row-shard (owned input-block range), heads sub-range per rank. */
    const uint32_t gpr = n_groups/(uint32_t)k;
    const uint64_t low_pr = (uint64_t)gpr * rank;       /* owned attn_low rows */
    const uint64_t bpr = blocks_b/(uint64_t)k;          /* owned output_b input blocks */
    char *ash[8]={0},*bsh[8]={0}; float *hr[8]={0}; int8_t *xqr[8]={0},*lqr[8]={0}; float *xsr[8]={0},*lsr[8]={0};
    float *lowr[8]={0},*pr[8]={0}; cudaStream_t st[8]={0};
    char *packtmp=NULL; int devs[8]; for(int r=0;r<k;r++)devs[r]=r;
    double rel=1.0;
    if (ok) {
        cudaSetDevice(0); ok = cudaMalloc(&packtmp,n_embd*bpr*34)==cudaSuccess;
        for (int r=0;r<k && ok;r++) {
            /* output_b row-shard: pack owned input blocks [r*bpr,(r+1)*bpr) of every out row */
            ok = cudaMemcpy2D(packtmp,bpr*34,bW + (uint64_t)r*bpr*34,blocks_b*34,bpr*34,n_embd,cudaMemcpyDeviceToDevice)==cudaSuccess;
            ok = ok && cudaSetDevice(r)==cudaSuccess && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && cudaMalloc(&ash[r],low_pr*blocks_a*34)==cudaSuccess && cudaMalloc(&bsh[r],n_embd*bpr*34)==cudaSuccess
                    && cudaMalloc(&hr[r],(uint64_t)gpr*group_dim*4)==cudaSuccess
                    && cudaMalloc(&xqr[r],(uint64_t)gpr*blocks_a*32)==cudaSuccess && cudaMalloc(&xsr[r],(uint64_t)gpr*blocks_a*4)==cudaSuccess
                    && cudaMalloc(&lowr[r],low_pr*4)==cudaSuccess
                    && cudaMalloc(&lqr[r],bpr*32)==cudaSuccess && cudaMalloc(&lsr[r],bpr*4)==cudaSuccess
                    && cudaMalloc(&pr[r],n_embd*4)==cudaSuccess;
            /* output_a col-shard: owned output rows [r*gpr*rank,...) are contiguous */
            ok = ok && cudaMemcpyPeer(ash[r],r,aW + (uint64_t)r*low_pr*blocks_a*34,0,low_pr*blocks_a*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(bsh[r],r,packtmp,0,n_embd*bpr*34)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(hr[r],r,heads + (uint64_t)r*gpr*group_dim,0,(uint64_t)gpr*group_dim*4)==cudaSuccess;
            cudaSetDevice(0);
        }
        for(int r=0;r<k;r++){cudaSetDevice(r);cudaDeviceSynchronize();}
        for (int r=0;r<k && ok;r++){ cudaSetDevice(r);
            dim3 qg((unsigned)blocks_a,(unsigned)gpr,1);
            quantize_q8_0_f32_kernel<<<qg,32,0,st[r]>>>(xqr[r],xsr[r],hr[r],group_dim,blocks_a);
            grouped_q8_0_a_preq_warp8_kernel<<<((unsigned)low_pr+7u)/8u,256,0,st[r]>>>(lowr[r],(const unsigned char*)ash[r],xqr[r],xsr[r],group_dim,rank,gpr,1,blocks_a,dp4a);
            quantize_q8_0_f32_kernel<<<(unsigned)bpr,32,0,st[r]>>>(lqr[r],lsr[r],lowr[r],low_pr,bpr);
            matmul_q8_0_preq_warp8_kernel<<<((unsigned)n_embd+7u)/8u,256,0,st[r]>>>(pr[r],(const unsigned char*)bsh[r],lqr[r],lsr[r],low_pr,n_embd,bpr,dp4a);
        }
        for(int r=0;r<k;r++){cudaSetDevice(r);cudaStreamSynchronize(st[r]);}
        if (ok) ok = ds4_gpu_tp_all_reduce_f32(devs,k,pr,(uint32_t)n_embd);
        cudaSetDevice(0); std::vector<float> hT(n_embd);
        if (ok) ok = cudaMemcpy(hT.data(),pr[0],n_embd*4,cudaMemcpyDeviceToHost)==cudaSuccess;
        double maxabs=0,scale=0; for(uint64_t i=0;i<n_embd;i++){double a=fabs((double)hF[i]);if(a>scale)scale=a;double d=fabs((double)hF[i]-(double)hT[i]);if(d>maxabs)maxabs=d;}
        if(scale<1e-12)scale=1e-12; rel=maxabs/scale;
        cudaSetDevice(0); if(packtmp)cudaFree(packtmp);
        for(int r=0;r<k;r++){cudaSetDevice(r);
            if(ash[r])cudaFree(ash[r]);if(bsh[r])cudaFree(bsh[r]);if(hr[r])cudaFree(hr[r]);
            if(xqr[r])cudaFree(xqr[r]);if(xsr[r])cudaFree(xsr[r]);if(lowr[r])cudaFree(lowr[r]);
            if(lqr[r])cudaFree(lqr[r]);if(lsr[r])cudaFree(lsr[r]);if(pr[r])cudaFree(pr[r]);
            if(st[r])cudaStreamDestroy(st[r]);}
    }
    fprintf(stderr, "ds4: TP grouped-O-proj jig k=%d groups=%u(gpr=%u) group_dim=%llu rank=%llu n_embd=%llu | %s rel=%.2e (1 all-reduce)\n",
            k, n_groups, gpr, (unsigned long long)group_dim, (unsigned long long)rank, (unsigned long long)n_embd,
            rel<1e-3?"PASS":"FAIL", rel);
    cudaSetDevice(0); cudaFree(heads);cudaFree(xq);cudaFree(xs);cudaFree(low);cudaFree(lowq);cudaFree(lows);cudaFree(outF);
    (void)cudaSetDevice(saved);
    return (ok && rel<1e-3) ? 1 : 0;
}

/* TP RESIDENT-SHARD jig: validates the build-time shard-cache path the decode
 * will actually use. Unlike ds4_gpu_tp_matmul_jig (which reshards with transient
 * peer-copies inside the jig), this calls ds4_gpu_cache_col_shard /
 * ds4_gpu_cache_row_shard to populate the resident g_tp_shards registry, then
 * runs the matmul reading the shard back through cuda_tp_shard_ptr — proving the
 * registry plumbing (build -> accessor -> matmul) reproduces the single-GPU golden. */
extern "C" int ds4_gpu_tp_shard_jig(
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, int k) {
    int ndev = 0; (void)cudaGetDeviceCount(&ndev);
    if (k < 2 || k > ndev || k > 8) { fprintf(stderr, "ds4: TP shard jig: need 2..%d GPUs (k=%d)\n", ndev, k); return 0; }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t wbytes = out_dim * blocks * 34;
    if (weight_offset > model_size || wbytes > model_size - weight_offset) return 0;
    if ((out_dim % (uint64_t)k) != 0 || (blocks % (uint64_t)k) != 0) {
        fprintf(stderr, "ds4: TP shard jig: out_dim(%llu)/blocks(%llu) not divisible by k=%d; skip\n",
                (unsigned long long)out_dim, (unsigned long long)blocks, k); return 0;
    }
    const int dp4a = cuda_q8_use_dp4a();
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;

    /* dev0: input ramp, prequant, single-GPU golden full matmul */
    if (cudaSetDevice(0) != cudaSuccess) return 0;
    const char *wfull = cuda_model_range_ptr(model_map, weight_offset, wbytes, "tp_shard_jig");
    if (!wfull) { fprintf(stderr, "ds4: TP shard jig: weight not resident on dev0\n"); return 0; }
    float *x=NULL,*outF=NULL; int8_t *xqA=NULL; float *xsA=NULL;
    std::vector<float> hx(in_dim); for (uint64_t i=0;i<in_dim;i++) hx[i]=sinf((float)i*0.011f)*0.5f;
    ok = ok && cudaMalloc(&x,in_dim*4)==cudaSuccess && cudaMalloc(&xqA,blocks*32)==cudaSuccess
            && cudaMalloc(&xsA,blocks*4)==cudaSuccess && cudaMalloc(&outF,out_dim*4)==cudaSuccess;
    if (ok) ok = cudaMemcpy(x,hx.data(),in_dim*4,cudaMemcpyHostToDevice)==cudaSuccess;
    if (ok) { quantize_q8_0_f32_kernel<<<(unsigned)blocks,32>>>(xqA,xsA,x,in_dim,blocks);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim+7u)/8u,256>>>(outF,(const unsigned char*)wfull,xqA,xsA,in_dim,out_dim,blocks,dp4a);
              ok = cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess; }
    std::vector<float> hF(out_dim);
    if (ok) ok = cudaMemcpy(hF.data(),outF,out_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess;

    int8_t *xqr[8]={0}; float *xsr[8]={0}; float *outr[8]={0}; cudaStream_t st[8]={0};
    int devs[8]; for(int r=0;r<k;r++)devs[r]=r;
    const uint64_t rpr = out_dim/(uint64_t)k;
    const uint64_t bpr = blocks/(uint64_t)k;

    /* ===== COLUMN-PARALLEL via the resident col-shard cache ===== */
    double rel_col = 1.0;
    if (ok) {
        for (int r=0;r<k && ok;r++) {
            ok = cudaSetDevice(r)==cudaSuccess && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && ds4_gpu_cache_col_shard(model_map, model_size, weight_offset, in_dim, out_dim, r, k, "shardjig_col")!=0;
            ok = ok && cudaMalloc(&xqr[r],blocks*32)==cudaSuccess && cudaMalloc(&xsr[r],blocks*4)==cudaSuccess
                    && cudaMalloc(&outr[r],rpr*4)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xqr[r],r,xqA,0,blocks*32)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xsr[r],r,xsA,0,blocks*4)==cudaSuccess;
        }
        for (int r=0;r<k && ok;r++) {
            ok = cudaSetDevice(r)==cudaSuccess;
            int kind=-1; uint64_t so=0, si=0, sb=0;
            const char *sh = cuda_tp_shard_ptr(model_map, weight_offset, &kind, &so, &si, &sb);
            ok = ok && sh!=NULL && kind==DS4_TP_SHARD_COL && so==rpr && si==in_dim && sb==blocks;
            if (ok) matmul_q8_0_preq_warp8_kernel<<<((unsigned)rpr+7u)/8u,256,0,st[r]>>>(outr[r],(const unsigned char*)sh,xqr[r],xsr[r],si,so,sb,dp4a);
        }
        float *outChk=NULL; std::vector<float> hC(out_dim);
        if (ok) { cudaSetDevice(0); ok = cudaMalloc(&outChk,out_dim*4)==cudaSuccess; }
        for (int r=0;r<k && ok;r++){ cudaSetDevice(r); ok = cudaStreamSynchronize(st[r])==cudaSuccess; }
        for (int r=0;r<k && ok;r++){ cudaSetDevice(0); ok = cudaMemcpyPeer(outChk+(uint64_t)r*rpr,0,outr[r],r,rpr*4)==cudaSuccess; }
        if (ok) { cudaSetDevice(0); cudaDeviceSynchronize(); ok = cudaMemcpy(hC.data(),outChk,out_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess; }
        if (ok) { double maxabs=0,scale=0; for(uint64_t i=0;i<out_dim;i++){double a=fabs((double)hF[i]);if(a>scale)scale=a;double d=fabs((double)hF[i]-(double)hC[i]);if(d>maxabs)maxabs=d;} if(scale<1e-12)scale=1e-12; rel_col=maxabs/scale; }
        cudaSetDevice(0); if(outChk)cudaFree(outChk);
        for (int r=0;r<k;r++){ cudaSetDevice(r); if(xqr[r])cudaFree(xqr[r]); if(xsr[r])cudaFree(xsr[r]); if(outr[r])cudaFree(outr[r]); if(st[r])cudaStreamDestroy(st[r]); xqr[r]=0;xsr[r]=0;outr[r]=0;st[r]=0; }
        ds4_gpu_tp_shards_release_all();
    }

    /* ===== ROW-PARALLEL via the resident row-shard cache ===== */
    double rel_row = 1.0;
    if (ok) {
        for (int r=0;r<k && ok;r++) {
            ok = cudaSetDevice(r)==cudaSuccess && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && ds4_gpu_cache_row_shard(model_map, model_size, weight_offset, in_dim, out_dim, r, k, "shardjig_row")!=0;
            ok = ok && cudaMalloc(&xqr[r],bpr*32)==cudaSuccess && cudaMalloc(&xsr[r],bpr*4)==cudaSuccess
                    && cudaMalloc(&outr[r],out_dim*4)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xqr[r],r,xqA+(uint64_t)r*bpr*32,0,bpr*32)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xsr[r],r,xsA+(uint64_t)r*bpr,0,bpr*4)==cudaSuccess;
        }
        for (int r=0;r<k && ok;r++) {
            ok = cudaSetDevice(r)==cudaSuccess;
            int kind=-1; uint64_t so=0, si=0, sb=0;
            const char *sh = cuda_tp_shard_ptr(model_map, weight_offset, &kind, &so, &si, &sb);
            ok = ok && sh!=NULL && kind==DS4_TP_SHARD_ROW && so==out_dim && si==bpr*32 && sb==bpr;
            if (ok) matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim+7u)/8u,256,0,st[r]>>>(outr[r],(const unsigned char*)sh,xqr[r],xsr[r],si,so,sb,dp4a);
        }
        for (int r=0;r<k && ok;r++){ cudaSetDevice(r); ok = cudaStreamSynchronize(st[r])==cudaSuccess; }
        if (ok) ok = ds4_gpu_tp_all_reduce_f32(devs,k,outr,(uint32_t)out_dim)!=0;
        std::vector<float> hR(out_dim);
        if (ok) { cudaSetDevice(0); ok = cudaMemcpy(hR.data(),outr[0],out_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess; }
        if (ok) { double maxabs=0,scale=0; for(uint64_t i=0;i<out_dim;i++){double a=fabs((double)hF[i]);if(a>scale)scale=a;double d=fabs((double)hF[i]-(double)hR[i]);if(d>maxabs)maxabs=d;} if(scale<1e-12)scale=1e-12; rel_row=maxabs/scale; }
        for (int r=0;r<k;r++){ cudaSetDevice(r); if(xqr[r])cudaFree(xqr[r]); if(xsr[r])cudaFree(xsr[r]); if(outr[r])cudaFree(outr[r]); if(st[r])cudaStreamDestroy(st[r]); xqr[r]=0;xsr[r]=0;outr[r]=0;st[r]=0; }
        ds4_gpu_tp_shards_release_all();
    }

    fprintf(stderr, "ds4: TP shard jig k=%d in=%llu out=%llu | col(resident) %s rel=%.2e | row(resident) %s rel=%.2e\n",
            k, (unsigned long long)in_dim, (unsigned long long)out_dim,
            rel_col<1e-3?"PASS":"FAIL", rel_col, rel_row<1e-3?"PASS":"FAIL", rel_row);

    cudaSetDevice(0); if(x)cudaFree(x); if(xqA)cudaFree(xqA); if(xsA)cudaFree(xsA); if(outF)cudaFree(outF);
    (void)cudaSetDevice(saved);
    return (ok && rel_col<1e-3 && rel_row<1e-3) ? 1 : 0;
}

/* Validate the RESIDENT shared-FFN shards built by the TP-aware cache loop:
 * recompute the full SwiGLU MLP on stage_dev0 from the HOST weights (golden),
 * then run it across the stage's k ranks reading their resident col/row shards
 * via cuda_tp_shard_ptr, all-reduce, and compare. Proves the load-path sharding
 * is numerically correct end-to-end (the data half of the TP integration).
 * Returns 1 on PASS. */
extern "C" int ds4_gpu_tp_resident_ffn_check(
        const void *model_map, uint64_t model_size,
        uint64_t gate_off, uint64_t up_off, uint64_t down_off,
        uint64_t in_dim, uint64_t ff_dim, float clamp, int stage_dev0, int k) {
    int ndev = 0; (void)cudaGetDeviceCount(&ndev);
    if (k < 2 || k > 8 || stage_dev0 < 0 || stage_dev0 + k > ndev) return 0;
    const uint64_t inb = (in_dim + 31) / 32, ffb = (ff_dim + 31) / 32;
    const uint64_t gate_wb = ff_dim * inb * 34, down_wb = in_dim * ffb * 34;
    if (gate_off + gate_wb > model_size || up_off + gate_wb > model_size || down_off + down_wb > model_size) return 0;
    if ((ff_dim % (uint64_t)k) != 0 || (ffb % (uint64_t)k) != 0) return 0;
    const int dp4a = cuda_q8_use_dp4a();
    int saved = 0; (void)cudaGetDevice(&saved);
    int ok = 1;

    /* ---- golden full FFN on stage_dev0 (weights read straight from host) ---- */
    if (cudaSetDevice(stage_dev0) != cudaSuccess) return 0;
    char *gW=NULL,*uW=NULL,*dW=NULL;
    ok = ok && cudaMalloc(&gW,gate_wb)==cudaSuccess && cudaMalloc(&uW,gate_wb)==cudaSuccess && cudaMalloc(&dW,down_wb)==cudaSuccess;
    ok = ok && cudaMemcpy(gW,(const char*)model_map+gate_off,gate_wb,cudaMemcpyHostToDevice)==cudaSuccess;
    ok = ok && cudaMemcpy(uW,(const char*)model_map+up_off,  gate_wb,cudaMemcpyHostToDevice)==cudaSuccess;
    ok = ok && cudaMemcpy(dW,(const char*)model_map+down_off,down_wb,cudaMemcpyHostToDevice)==cudaSuccess;
    float *x=NULL,*g=NULL,*u=NULL,*mid=NULL,*outF=NULL; int8_t *xq=NULL,*midq=NULL; float *xs=NULL,*mids=NULL;
    std::vector<float> hx(in_dim); for (uint64_t i=0;i<in_dim;i++) hx[i]=sinf((float)i*0.009f)*0.4f;
    ok = ok && cudaMalloc(&x,in_dim*4)==cudaSuccess && cudaMalloc(&xq,inb*32)==cudaSuccess && cudaMalloc(&xs,inb*4)==cudaSuccess
            && cudaMalloc(&g,ff_dim*4)==cudaSuccess && cudaMalloc(&u,ff_dim*4)==cudaSuccess && cudaMalloc(&mid,ff_dim*4)==cudaSuccess
            && cudaMalloc(&midq,ffb*32)==cudaSuccess && cudaMalloc(&mids,ffb*4)==cudaSuccess && cudaMalloc(&outF,in_dim*4)==cudaSuccess;
    if (ok) ok = cudaMemcpy(x,hx.data(),in_dim*4,cudaMemcpyHostToDevice)==cudaSuccess;
    if (ok) { quantize_q8_0_f32_kernel<<<(unsigned)inb,32>>>(xq,xs,x,in_dim,inb);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)ff_dim+7u)/8u,256>>>(g,(const unsigned char*)gW,xq,xs,in_dim,ff_dim,inb,dp4a);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)ff_dim+7u)/8u,256>>>(u,(const unsigned char*)uW,xq,xs,in_dim,ff_dim,inb,dp4a);
              swiglu_kernel<<<((unsigned)ff_dim+255u)/256u,256>>>(mid,g,u,(uint32_t)ff_dim,clamp,1.0f);
              quantize_q8_0_f32_kernel<<<(unsigned)ffb,32>>>(midq,mids,mid,ff_dim,ffb);
              matmul_q8_0_preq_warp8_kernel<<<((unsigned)in_dim+7u)/8u,256>>>(outF,(const unsigned char*)dW,midq,mids,ff_dim,in_dim,ffb,dp4a);
              ok = cudaDeviceSynchronize()==cudaSuccess && cudaGetLastError()==cudaSuccess; }
    std::vector<float> hF(in_dim);
    if (ok) ok = cudaMemcpy(hF.data(),outF,in_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess;
    if(gW)cudaFree(gW); if(uW)cudaFree(uW); if(dW)cudaFree(dW);

    /* ---- TP across the stage's k ranks reading RESIDENT shards ---- */
    const uint64_t ffpr = ff_dim/(uint64_t)k, bpr = ffb/(uint64_t)k;
    int8_t *xqr[8]={0},*mqr[8]={0}; float *xsr[8]={0},*msr[8]={0};
    float *gr[8]={0},*ur[8]={0},*mr[8]={0},*pr[8]={0}; cudaStream_t st[8]={0};
    int devs[8]; for(int r=0;r<k;r++)devs[r]=stage_dev0+r;
    double rel=1.0;
    if (ok) {
        for (int r=0;r<k && ok;r++) {
            int dv=stage_dev0+r;
            ok = cudaSetDevice(dv)==cudaSuccess && cudaStreamCreate(&st[r])==cudaSuccess;
            ok = ok && cudaMalloc(&xqr[r],inb*32)==cudaSuccess && cudaMalloc(&xsr[r],inb*4)==cudaSuccess
                    && cudaMalloc(&gr[r],ffpr*4)==cudaSuccess && cudaMalloc(&ur[r],ffpr*4)==cudaSuccess && cudaMalloc(&mr[r],ffpr*4)==cudaSuccess
                    && cudaMalloc(&mqr[r],bpr*32)==cudaSuccess && cudaMalloc(&msr[r],bpr*4)==cudaSuccess && cudaMalloc(&pr[r],in_dim*4)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xqr[r],dv,xq,stage_dev0,inb*32)==cudaSuccess;
            ok = ok && cudaMemcpyPeer(xsr[r],dv,xs,stage_dev0,inb*4)==cudaSuccess;
        }
        for (int r=0;r<k && ok;r++) {
            int dv=stage_dev0+r;
            ok = cudaSetDevice(dv)==cudaSuccess;
            int gk=-1,uk=-1,dk=-1; uint64_t go=0,gi=0,gb=0,uo=0,ui=0,ub=0,doo=0,di=0,db=0;
            const char *gsh = cuda_tp_shard_ptr(model_map, gate_off, &gk, &go, &gi, &gb);
            const char *ush = cuda_tp_shard_ptr(model_map, up_off,   &uk, &uo, &ui, &ub);
            const char *dsh = cuda_tp_shard_ptr(model_map, down_off, &dk, &doo,&di, &db);
            ok = ok && gsh && ush && dsh && gk==DS4_TP_SHARD_COL && uk==DS4_TP_SHARD_COL && dk==DS4_TP_SHARD_ROW
                    && go==ffpr && uo==ffpr && doo==in_dim && db==bpr;
            if (ok) {
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)ffpr+7u)/8u,256,0,st[r]>>>(gr[r],(const unsigned char*)gsh,xqr[r],xsr[r],in_dim,ffpr,inb,dp4a);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)ffpr+7u)/8u,256,0,st[r]>>>(ur[r],(const unsigned char*)ush,xqr[r],xsr[r],in_dim,ffpr,inb,dp4a);
                swiglu_kernel<<<((unsigned)ffpr+255u)/256u,256,0,st[r]>>>(mr[r],gr[r],ur[r],(uint32_t)ffpr,clamp,1.0f);
                quantize_q8_0_f32_kernel<<<(unsigned)bpr,32,0,st[r]>>>(mqr[r],msr[r],mr[r],ffpr,bpr);
                matmul_q8_0_preq_warp8_kernel<<<((unsigned)in_dim+7u)/8u,256,0,st[r]>>>(pr[r],(const unsigned char*)dsh,mqr[r],msr[r],ffpr,in_dim,bpr,dp4a);
            }
        }
        for (int r=0;r<k && ok;r++){ cudaSetDevice(stage_dev0+r); ok = cudaStreamSynchronize(st[r])==cudaSuccess; }
        if (ok) ok = ds4_gpu_tp_all_reduce_f32(devs,k,pr,(uint32_t)in_dim)!=0;
        std::vector<float> hT(in_dim);
        if (ok) { cudaSetDevice(stage_dev0); ok = cudaMemcpy(hT.data(),pr[0],in_dim*4,cudaMemcpyDeviceToHost)==cudaSuccess; }
        if (ok) { double maxabs=0,scale=0; for(uint64_t i=0;i<in_dim;i++){double a=fabs((double)hF[i]);if(a>scale)scale=a;double d=fabs((double)hF[i]-(double)hT[i]);if(d>maxabs)maxabs=d;} if(scale<1e-12)scale=1e-12; rel=maxabs/scale; }
        for (int r=0;r<k;r++){ cudaSetDevice(stage_dev0+r);
            if(xqr[r])cudaFree(xqr[r]);if(xsr[r])cudaFree(xsr[r]);if(gr[r])cudaFree(gr[r]);if(ur[r])cudaFree(ur[r]);
            if(mr[r])cudaFree(mr[r]);if(mqr[r])cudaFree(mqr[r]);if(msr[r])cudaFree(msr[r]);if(pr[r])cudaFree(pr[r]);
            if(st[r])cudaStreamDestroy(st[r]); }
    }
    fprintf(stderr, "ds4: TP resident-FFN check stage_dev0=%d k=%d in=%llu ff=%llu | %s rel=%.2e\n",
            stage_dev0, k, (unsigned long long)in_dim, (unsigned long long)ff_dim,
            (ok && rel<1e-3)?"PASS":"FAIL", rel);
    cudaSetDevice(stage_dev0);
    if(x)cudaFree(x);if(xq)cudaFree(xq);if(xs)cudaFree(xs);if(g)cudaFree(g);if(u)cudaFree(u);
    if(mid)cudaFree(mid);if(midq)cudaFree(midq);if(mids)cudaFree(mids);if(outF)cudaFree(outF);
    (void)cudaSetDevice(saved);
    return (ok && rel<1e-3) ? 1 : 0;
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, ds4_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32, 0, ds4_decode_stream()>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) return 0;
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

extern "C" int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        n_tok == 1u &&
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") == NULL;
    if (!serial_f16 && g_cublas_ready && n_tok > 1) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256, 0, ds4_decode_stream()>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1, 0, ds4_decode_stream()>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (ordered_router) {
        matmul_f16_ordered_chunks_kernel<<<grid, 32, 0, ds4_decode_stream()>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1 ||
        getenv("DS4_CUDA_NO_F16_PAIR_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_F16_MATMUL") != NULL ||
        getenv("DS4_CUDA_SERIAL_ROUTER") != NULL ||
        getenv("DS4_CUDA_NO_ORDERED_F16_MATMUL") != NULL) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    matmul_f16_pair_ordered_chunks_kernel<<<(unsigned)out_dim, 32, 0, ds4_decode_stream()>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        out_dim);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_ordered_chunks launch");
}

extern "C" int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

extern "C" int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}

extern "C" int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<1, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    rms_norm_plain_kernel<<<rows, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
extern "C" int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<1, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_weight_kernel<<<rows, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (getenv("DS4_CUDA_DISABLE_QKV_RMS_FUSED") == NULL) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const float *q_w = (const float *)cuda_model_range_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), "q_rms_weight");
        const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256, 0, ds4_decode_stream()>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}
extern "C" int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256, 0, ds4_decode_stream()>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256, 0, ds4_decode_stream()>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}
extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64, 0, ds4_decode_stream()>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128, 0, ds4_decode_stream()>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
extern "C" int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim);
extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}
extern "C" int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
extern "C" int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256, 0, ds4_decode_stream()>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256, 0, ds4_decode_stream()>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256, 0, ds4_decode_stream()>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}
extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256, 0, ds4_decode_stream()>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_f16, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    if (n_tokens > 1u && top_k == 512u &&
        getenv("DS4_CUDA_NO_INDEXED_TOPK_SORT") == NULL) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512, 0, ds4_decode_stream()>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        getenv("DS4_CUDA_NO_INDEXED_HEADS8") == NULL) {
        if (getenv("DS4_CUDA_INDEXED_TWOPASS") == NULL) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_WINDOW_ATTENTION") == NULL &&
        (getenv("DS4_CUDA_WINDOW_ATTENTION") != NULL || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION") == NULL) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256, 0, ds4_decode_stream()>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256, 0, ds4_decode_stream()>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_b_offset, out_b_bytes, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = getenv("DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN");
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        getenv("DS4_CUDA_NO_CUBLAS_ATTENTION_OUTPUT_A") == NULL) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        attention_pack_group_heads_f16_kernel<<<(heads_h_count + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256, 0, ds4_decode_stream()>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32, 0, ds4_decode_stream()>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
        grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256, 0, ds4_decode_stream()>>>((float *)low->ptr,
                                                          out_a,
                                                          xq,
                                                          xscale,
                                                          group_dim,
                                                          rank,
                                                          n_groups,
                                                          n_tokens,
                                                          blocks_a,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    (void)out_b;
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}
extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, ds4_decode_stream()>>>(xq,
                                            xscale,
                                            (const float *)heads->ptr,
                                            group_dim,
                                            blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q8 prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, 1, 1);
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256, 0, ds4_decode_stream()>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      1,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(), "attention_output_low_q8 launch");
}
/* Tensor-Parallel grouped O-proj output_a: like ds4_gpu_attention_output_low_q8_tensor
 * but reads this rank's COL-shard of output_a from the resident g_tp_shards registry
 * (owned groups' contiguous output rows) instead of the full model_map tensor. The
 * shard holds out_rows = gpr*rank output rows for gpr = out_rows/rank owned groups;
 * `heads` is this rank's owned heads compacted at [0, gpr*group_dim). Produces this
 * rank's partial attn_low (gpr*rank); the caller row-parallels output_b + all-reduces. */
extern "C" int ds4_gpu_tp_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                out_a_parent_offset,
        uint64_t                rank,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || rank == 0) return 0;
    int kind = -1; uint64_t out_rows = 0, group_dim = 0, blocks_a = 0;
    const char *out_a = cuda_tp_shard_ptr(model_map, out_a_parent_offset, &kind, &out_rows, &group_dim, &blocks_a);
    if (!out_a || kind != DS4_TP_SHARD_COL || group_dim == 0 || (out_rows % rank) != 0) return 0;
    const uint32_t n_groups = (uint32_t)(out_rows / rank);   /* owned groups (gpr) */
    if (n_groups == 0) return 0;
    if (heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < out_rows * sizeof(float)) {
        return 0;
    }
    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    void *tmp = cuda_tmp_alloc(scale_offset + x_rows * blocks_a * sizeof(float), "tp attn out low prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32, 0, ds4_decode_stream()>>>(xq, xscale,
                                            (const float *)heads->ptr, group_dim, blocks_a);
    if (!cuda_ok(cudaGetLastError(), "tp attn out low prequant launch")) return 0;
    grouped_q8_0_a_preq_warp8_kernel<<<((unsigned)out_rows + 7u) / 8u, 256, 0, ds4_decode_stream()>>>(
            (float *)low->ptr, reinterpret_cast<const unsigned char *>(out_a),
            xq, xscale, group_dim, rank, n_groups, 1, blocks_a, use_dp4a);
    return cuda_ok(cudaGetLastError(), "tp attn out low launch");
}
extern "C" int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (getenv("DS4_CUDA_DISABLE_SHARED_GATE_UP_PAIR") == NULL) {
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}
extern "C" int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}
extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth, 0, ds4_decode_stream()>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
extern "C" int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits) {
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if ((n_expert != 256u && n_expert != 384u) || n_expert_used != 6u) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < (uint64_t)n_expert * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, (uint64_t)n_expert * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
            getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block, 0, ds4_decode_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode, n_expert, n_expert_used, expert_weight_scale);
        } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
            router_select_parallel_kernel<<<1, 384, 0, ds4_decode_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode, n_expert, n_expert_used, expert_weight_scale);
        } else {
            router_select_kernel<<<1, 1, 0, ds4_decode_stream()>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode, n_expert, n_expert_used, expert_weight_scale);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
extern "C" int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens) {
    if ((n_expert != 256u && n_expert != 384u) || n_expert_used != 6u) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < (uint64_t)n_expert * sizeof(float)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, (uint64_t)n_expert * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    if (getenv("DS4_CUDA_NO_WARP_ROUTER_SELECT") == NULL &&
        getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block, 0, ds4_decode_stream()>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode, n_expert, n_expert_used, expert_weight_scale);
    } else if (getenv("DS4_CUDA_NO_PARALLEL_ROUTER_SELECT") == NULL) {
        router_select_parallel_kernel<<<n_tokens, 384, 0, ds4_decode_stream()>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode, n_expert, n_expert_used, expert_weight_scale);
    } else {
        router_select_kernel<<<n_tokens, 1, 0, ds4_decode_stream()>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode, n_expert, n_expert_used, expert_weight_scale);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum = 0;
    int summs = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        uint8_t sc, m;
        dev_q4_K_get_scale_min(j, x->scales, &sc, &m);
        summs += (int)m * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int shift = (j & 1u) ? 4 : 0;
        isum += (int)sc * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, shift);
    }
    return y->d * xd * (float)isum - y->d * xmin * (float)summs;
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            int d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2, q8, shift);
            d = sc[is++] & 0x0f;
            isum += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_expert; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[n_expert] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_expert) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < n_expert; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[n_expert] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m,
        uint32_t n_expert) {
    uint32_t e = threadIdx.x;
    if (e >= n_expert) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    /* LUT must be loaded into shared mem UNCONDITIONALLY: the shared sxq copy is
     * only valid when xq_blocks<=16 (sxq is [8][16]), but s_iq2_grid/s_iq2_signs
     * are consumed by dev_dot_..._deq_lut regardless of xq_blocks. Loading them
     * inside the <=16 guard left them uninitialized for wide-embd models (Pro:
     * xq_blocks=28) -> garbage dequant. Flash (xq_blocks=16) is unchanged. */
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    /* LUT must be loaded into shared mem UNCONDITIONALLY: the shared sxq copy is
     * only valid when xq_blocks<=16 (sxq is [8][16]), but s_iq2_grid/s_iq2_signs
     * are consumed by dev_dot_..._deq_lut regardless of xq_blocks. Loading them
     * inside the <=16 guard left them uninitialized for wide-embd models (Pro:
     * xq_blocks=28) -> garbage dequant. Flash (xq_blocks=16) is unchanged. */
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    /* LUT must be loaded into shared mem UNCONDITIONALLY: the shared sxq copy is
     * only valid when xq_blocks<=16 (sxq is [8][16]), but s_iq2_grid/s_iq2_signs
     * are consumed by dev_dot_..._deq_lut regardless of xq_blocks. Loading them
     * inside the <=16 guard left them uninitialized for wide-embd models (Pro:
     * xq_blocks=28) -> garbage dequant. Flash (xq_blocks=16) is unchanged. */
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
    }
    __syncthreads();
    if (xq_blocks <= 16u) {
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* Decode (n_tokens==1) direct down-projection + sum over the token's n_expert
 * selected experts (n_expert in 1..6), writing the summed result straight to out
 * (no per-expert scratch, no separate sum pass). n_expert==6 is the dense decode
 * case; n_expert<6 is the TP expert-compacted case (each rank owns a subset).
 * Generalized from the validated 6-expert kernel by making the loop bound the
 * runtime n_expert (the grid is per-row, independent of n_expert). */
__global__ static void moe_down_sumN_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sumN_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

/* IQ2_XXS down-projection variant of the n_tokens==1 / N-expert decode kernel.
 * Mirrors moe_down_sumN_qwarp32_kernel but reads IQ2_XXS down weights. */
__global__ static void moe_down_sumN_iq2xxs_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    for (uint32_t slot = 0; slot < n_expert; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_iq2_xxs_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

/* IQ2_XXS down-projection variant of the batched tile8 path (n_tokens>1).
 * Mirrors moe_down_expert_tile8_row32_kernel but reads IQ2_XXS down weights. */
__global__ static void moe_down_expert_tile8_row32_iq2xxs_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                    xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                    xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens,
        uint32_t layer_index) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    /* Supported expert quant combos: (gate Q4_K, down Q4_K), (gate IQ2_XXS,
     * down Q2_K), or (gate IQ2_XXS, down IQ2_XXS). */
    const int down_iq2xxs = (down_type == 16u);
    if (!q4k_path && (gate_type != 16u || (down_type != 10u && down_type != 16u))) return 0;
    if (q4k_path && (n_tokens != 1u || n_expert != 6u)) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    cuda_temp_model_range gate_tmp = {NULL, 0, NULL};
    cuda_temp_model_range up_tmp = {NULL, 0, NULL};
    cuda_temp_model_range down_tmp = {NULL, 0, NULL};
    const int use_temp_weights =
        cuda_moe_temp_weights_enabled() &&
        !g_model_device_owned &&
        !g_model_registered &&
        !cuda_model_direct_host_access_allowed();
    if (getenv("DS4_CUDA_MTP_TRACE_ALLOC") != NULL) {
        int cur_dev = -1;
        size_t free_b = 0, total_b = 0;
        (void)cudaGetDevice(&cur_dev);
        (void)cudaMemGetInfo(&free_b, &total_b);
        fprintf(stderr,
                "ds4: CUDA MTP trace routed_moe dev=%d n_tokens=%u n_expert=%u/%u q4k=%d use_temp=%d gate=%.2f MiB up=%.2f MiB down=%.2f MiB free %.2f/%.2f MiB\n",
                cur_dev,
                n_tokens,
                n_expert,
                n_total_expert,
                q4k_path,
                use_temp_weights,
                (double)gate_bytes / 1048576.0,
                (double)gate_bytes / 1048576.0,
                (double)down_bytes / 1048576.0,
                (double)free_b / 1048576.0,
                (double)total_b / 1048576.0);
    }

    /* Selected-expert compact copy: for decode (1 token, 6 experts), copy only
     * the 6 selected experts into a pre-allocated scratch instead of all 256.
     * This replaces 1.7 GB/layer H2D with ~42 MB/layer. */
    const char *gate_w = NULL;
    const char *up_w = NULL;
    const char *down_w = NULL;
    int use_compact = 0;
    int use_stream = 0; /* SSD streaming: weights came from g_stream_selected_cache[dev] */
    /* max n_expert=6, max n_tokens=3 (MTP draft=3), so 18 slots */
    int32_t compact_remap[18];
    uint64_t compact_exp_bytes =
        gate_expert_bytes > down_expert_bytes ? gate_expert_bytes : down_expert_bytes;

    /* PP mode also benefits from compact MoE: even though weights are
       resident-cached, the full 256-expert tensor is large and may not fit
       in the persistent scratch buffer. Use compact copy for decode.
       MTP draft=2/3 produces n_tokens=2/3; allow up to 3 tokens so compact
       MoE still applies (scratch becomes 6*3=18 experts, ~121MB). Each
       (token,expert) pair gets its own scratch slot so the batch MoE kernels
       address per-token weights via compact_remap (slot index). */
    int compact_dev = 0;
    (void)cudaGetDevice(&compact_dev);
    if (compact_dev < 0 || compact_dev >= DS4_CUDA_MAX_DEVICES) compact_dev = 0;
    /* Use compact MoE only where it is actually needed:
     *  - use_temp_weights: non-PP path that would otherwise alloc the full
     *    256-expert tensor transiently (the original compact case).
     *  - PP batch verify (n_tokens>1): the MTP suffix verify would otherwise
     *    OOM trying to materialize the full expert tensor; compact avoids it.
     * In PP single-token decode (n_tokens==1) the per-GPU resident weights are
     * already on device, so we keep the resident path and skip the ~1.7 GiB/pass
     * host->device compact copy that would otherwise dominate the decode step. */
    const int compact_pp_batch = g_pp_decode_active && n_tokens > 1u;
    if ((use_temp_weights || compact_pp_batch) && n_tokens <= 3 && n_expert <= 6) {
        /* Allocate compact scratch on first use, per device. PP runs each layer
         * on a different GPU, so each device keeps its own scratch. It is shared
         * across all layers and both models (target + MTP) on that device, so it
         * must be sized for the largest per-expert weight AND the largest slot
         * count (n_expert * n_tokens) seen so far on that device. */
        const uint32_t need_slots = n_expert * n_tokens;
        if ((!g_moe_compact_gate[compact_dev] ||
             g_moe_compact_per_expert[compact_dev] < compact_exp_bytes ||
             g_moe_compact_slots[compact_dev] < need_slots) && n_expert > 0) {
            /* Grow to the max of both dimensions seen so far to avoid
             * reallocation thrashing when target and MTP models alternate. */
            uint64_t max_per = gate_expert_bytes > down_expert_bytes ? gate_expert_bytes : down_expert_bytes;
            if (max_per < g_moe_compact_per_expert[compact_dev]) max_per = g_moe_compact_per_expert[compact_dev];
            uint32_t max_slots = need_slots > g_moe_compact_slots[compact_dev] ? need_slots : g_moe_compact_slots[compact_dev];
            if (g_moe_compact_gate[compact_dev]) cudaFree(g_moe_compact_gate[compact_dev]);
            if (g_moe_compact_up[compact_dev])   cudaFree(g_moe_compact_up[compact_dev]);
            if (g_moe_compact_down[compact_dev]) cudaFree(g_moe_compact_down[compact_dev]);
            if (g_moe_compact_selected_dev[compact_dev]) cudaFree(g_moe_compact_selected_dev[compact_dev]);
            g_moe_compact_gate[compact_dev] = g_moe_compact_up[compact_dev] = g_moe_compact_down[compact_dev] = NULL;
            g_moe_compact_selected_dev[compact_dev] = NULL;
            g_moe_compact_per_expert[compact_dev] = 0;
            g_moe_compact_slots[compact_dev] = 0;
            uint64_t total = (uint64_t)max_slots * max_per;
            if (cudaMalloc(&g_moe_compact_gate[compact_dev], (size_t)total) == cudaSuccess &&
                cudaMalloc(&g_moe_compact_up[compact_dev],   (size_t)total) == cudaSuccess &&
                cudaMalloc(&g_moe_compact_down[compact_dev], (size_t)total) == cudaSuccess &&
                cudaMalloc(&g_moe_compact_selected_dev[compact_dev], (size_t)max_slots * 4) == cudaSuccess) {
                g_moe_compact_per_expert[compact_dev] = max_per;
                g_moe_compact_slots[compact_dev] = max_slots;
            } else {
                cudaFree(g_moe_compact_gate[compact_dev]); g_moe_compact_gate[compact_dev] = NULL;
                cudaFree(g_moe_compact_up[compact_dev]);   g_moe_compact_up[compact_dev] = NULL;
                cudaFree(g_moe_compact_down[compact_dev]); g_moe_compact_down[compact_dev] = NULL;
                cudaFree(g_moe_compact_selected_dev[compact_dev]); g_moe_compact_selected_dev[compact_dev] = NULL;
                (void)cudaGetLastError();
            }
        }
        if (g_moe_compact_gate[compact_dev] &&
            g_moe_compact_per_expert[compact_dev] >= compact_exp_bytes &&
            g_moe_compact_slots[compact_dev] >= need_slots) {
            int capture_active = 0;
            if (g_cuda_decode_stream_created) {
                cudaStreamCaptureStatus st;
                if (cudaSuccess == cudaStreamIsCapturing(ds4_decode_stream(), &st))
                    capture_active = (st == cudaStreamCaptureStatusActive);
            }
            if (capture_active) {
                /* SEAM 2: no host work allowed during graph capture. Point at
                 * buffers a prior non-captured prime pass populated. Streamed
                 * layers SHOULD have capture disabled (ds4_gpu_decode_graph_can_capture
                 * returns 0 when g_ssd_streaming_mode); this is the belt-and-
                 * suspenders fallback if a streamed layer is ever captured. */
                use_compact = 1;
                if (g_ssd_streaming_mode && g_stream_selected_cache[compact_dev].valid) {
                    use_stream = 1;
                    gate_w = (const char *)g_stream_selected_cache[compact_dev].gate_ptr;
                    up_w   = (const char *)g_stream_selected_cache[compact_dev].up_ptr;
                    down_w = (const char *)g_stream_selected_cache[compact_dev].down_ptr;
                } else {
                    gate_w = g_moe_compact_gate[compact_dev];
                    up_w   = g_moe_compact_up[compact_dev];
                    down_w = g_moe_compact_down[compact_dev];
                }
            } else {
            /* Read selected indices from device (6 int32 = 24 bytes, fast).
             * With the graph decode stream enabled, router kernels run on a
             * non-default stream, so synchronize it before this host read. */
            int32_t host_selected[18]; /* max 6 experts * 3 tokens */
            cudaStream_t decode_stream = ds4_decode_stream();
            cudaError_t ce = decode_stream
                ? cudaStreamSynchronize(decode_stream)
                : cudaSuccess;
            if (ce == cudaSuccess) {
                ce = cudaMemcpy(host_selected, selected->ptr,
                                (size_t)n_expert * n_tokens * 4, cudaMemcpyDeviceToHost);
            }
            if (ce == cudaSuccess && g_ssd_streaming_mode) {
                /* STREAMING gather: the per-device GPU-LRU g_stream_expert_cache[dev]
                 * (backed by OS page cache + Optane SSD) replaces the direct
                 * mmap->H2D loop below. Dedup-packs unique experts into
                 * g_stream_selected_cache[dev].{gate,up,down}_ptr and writes the
                 * per-pair slot map into slot_selected_ptr. UPSTREAM entry is
                 * ds4_gpu_stream_expert_cache_prepare_selected_batch (UP 3176). */
                ds4_gpu_stream_expert_table table;
                table.model_map         = model_map;
                table.model_size        = model_size;
                table.layer             = layer_index;
                table.n_total_expert    = n_total_expert;
                table.gate_offset       = gate_offset;
                table.up_offset         = up_offset;
                table.down_offset       = down_offset;
                table.gate_expert_bytes = gate_expert_bytes;
                table.down_expert_bytes = down_expert_bytes;
                if (ds4_gpu_stream_expert_cache_prepare_selected_batch(
                        &table, host_selected, n_tokens, n_expert) &&
                    g_stream_selected_cache[compact_dev].valid) {
                    use_compact = 1;
                    use_stream  = 1;
                    gate_w = (const char *)g_stream_selected_cache[compact_dev].gate_ptr;
                    up_w   = (const char *)g_stream_selected_cache[compact_dev].up_ptr;
                    down_w = (const char *)g_stream_selected_cache[compact_dev].down_ptr;
                }
                /* if prepare failed (returned 0 / !valid), use_stream stays 0 and
                 * we drop into the existing g_moe_compact_* fill below (safety net). */
            }
            if (ce == cudaSuccess && !use_stream) {
                use_compact = 1;
                /* Copy each selected expert's weights to compact scratch per token */
                for (uint32_t ti = 0; ti < n_tokens; ti++) {
                    for (uint32_t ei = 0; ei < n_expert; ei++) {
                        int ge = host_selected[ti * n_expert + ei];
                        if (ge < 0) ge = 0;
                        if ((uint32_t)ge >= n_total_expert) ge = 0;
                        uint32_t uge = (uint32_t)ge;
                        /* The batch MoE kernels read weights at
                         * base + selected[pair]*expert_bytes and derive the
                         * input token from pair/n_expert. Store the compact
                         * scratch slot index (ti*n_expert+ei) here so each
                         * (token,expert) pair reads its own copied weight. */
                        compact_remap[ti * n_expert + ei] = (int32_t)(ti * n_expert + ei);
                        uint64_t slot_off = ((uint64_t)ti * n_expert + ei) * gate_expert_bytes;
                        cudaMemcpyAsync(g_moe_compact_gate[compact_dev] + slot_off,
                                        (const char*)model_map + gate_offset + (uint64_t)uge * gate_expert_bytes,
                                        gate_expert_bytes, cudaMemcpyHostToDevice, ds4_decode_stream());
                        cudaMemcpyAsync(g_moe_compact_up[compact_dev]   + slot_off,
                                        (const char*)model_map + up_offset   + (uint64_t)uge * gate_expert_bytes,
                                        gate_expert_bytes, cudaMemcpyHostToDevice, ds4_decode_stream());
                        cudaMemcpyAsync(g_moe_compact_down[compact_dev] + ((uint64_t)ti * n_expert + ei) * down_expert_bytes,
                                        (const char*)model_map + down_offset + (uint64_t)uge * down_expert_bytes,
                                        down_expert_bytes, cudaMemcpyHostToDevice, ds4_decode_stream());
                    }
                }
                cudaMemcpyAsync(g_moe_compact_selected_dev[compact_dev], compact_remap,
                                (size_t)n_expert * n_tokens * 4, cudaMemcpyHostToDevice, ds4_decode_stream());
                gate_w = g_moe_compact_gate[compact_dev];
                up_w   = g_moe_compact_up[compact_dev];
                down_w = g_moe_compact_down[compact_dev];
                if (!g_moe_compact_notice_printed) {
                    fprintf(stderr, "ds4: CUDA MoE compact selected-expert copy (%u experts, %.2f MiB/layer)\n",
                            n_expert, (double)(n_expert * (gate_expert_bytes * 2 + down_expert_bytes)) / 1048576.0);
                    g_moe_compact_notice_printed = 1;
                }
            } else {
                (void)cudaGetLastError();
            }
            } /* !capture_active */
        }
    }

    if (!use_compact) {
        if (use_temp_weights) {
            if (!g_moe_temp_weights_notice_printed ||
                getenv("DS4_CUDA_WEIGHT_CACHE_VERBOSE") != NULL) {
                fprintf(stderr,
                        "ds4: CUDA MoE using transient per-layer weights "
                        "(gate %.2f MiB, up %.2f MiB, down %.2f MiB)\n",
                        (double)gate_bytes / 1048576.0,
                        (double)gate_bytes / 1048576.0,
                        (double)down_bytes / 1048576.0);
                g_moe_temp_weights_notice_printed = 1;
            }
        }
        gate_w = use_temp_weights
            ? cuda_model_temp_range_alloc(model_map, model_size, gate_offset,
                                          gate_bytes, "moe_gate", &gate_tmp)
            : cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
        up_w = use_temp_weights
            ? cuda_model_temp_range_alloc(model_map, model_size, up_offset,
                                          gate_bytes, "moe_up", &up_tmp)
            : cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
        down_w = use_temp_weights
            ? cuda_model_temp_range_alloc(model_map, model_size, down_offset,
                                          down_bytes, "moe_down", &down_tmp)
            : cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    }

    if (!gate_w || !up_w || !down_w) {
        (void)cuda_moe_temp_weights_release(&gate_tmp, &up_tmp, &down_tmp);
        return 0;
    }

    int ok = 1;
    const int32_t *selected_ptr =
        use_stream  ? g_stream_selected_cache[compact_dev].slot_selected_ptr :
        use_compact ? g_moe_compact_selected_dev[compact_dev] :
                      (const int32_t *)selected->ptr;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = getenv("DS4_CUDA_MOE_PROFILE") != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_sorted_pairs = n_tokens > 1u;
        const uint32_t use_expert_tiles = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_EXPERT_TILES") == NULL;
        /* IQ2_XXS down only has tile8 + sum6 variants, so force tile8 (not tile4/16). */
        const uint32_t expert_tile_m = (getenv("DS4_CUDA_MOE_TILE4") && !down_iq2xxs) ? 4u : 8u;
        const uint32_t write_gate_up = getenv("DS4_CUDA_MOE_WRITE_GATE_UP") != NULL;
        const uint32_t use_p2_sorted = use_sorted_pairs && getenv("DS4_CUDA_MOE_NO_P2") == NULL;
        const uint32_t use_atomic_down = use_expert_tiles &&
            (getenv("DS4_CUDA_MOE_ATOMIC_DOWN") != NULL ||
             (n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == NULL));
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_GATE_ROW128") != NULL ||
             (n_tokens >= 128u &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_GATE_ROW128") == NULL));
        const uint32_t use_down_tile16 = use_atomic_down && expert_tile_m == 8u && !down_iq2xxs &&
            n_tokens >= 128u && getenv("DS4_CUDA_MOE_NO_DOWN_TILE16") == NULL;
        const uint32_t use_decode_lut_gate =
            n_tokens == 1u && xq_blocks <= 16u &&
            getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL;
        const uint32_t gate_row_span =
            getenv("DS4_CUDA_MOE_GATE_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_GATE_ROW2048") != NULL ? 2048u : 1024u;
        const uint32_t down_row_span =
            getenv("DS4_CUDA_MOE_DOWN_ROW512") != NULL ? 512u :
            getenv("DS4_CUDA_MOE_DOWN_ROW1024") != NULL ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_atomic_down && expert_tile_m == 8u && !down_iq2xxs &&
            (getenv("DS4_CUDA_MOE_DOWN_ROW2048") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW256") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW128") != NULL ||
             getenv("DS4_CUDA_MOE_DOWN_ROW64") != NULL ||
             (use_down_tile16 &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW2048") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW256") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW128") == NULL &&
              getenv("DS4_CUDA_MOE_NO_DOWN_ROW64") == NULL));
        /* Decode direct down+sum: n_tokens==1 with 1..6 selected experts. Covers
         * dense decode (n_expert==6) and TP expert-compacted decode (n_expert<6).
         * This is the validated path; the generic moe_down + moe_sum fallback is
         * only used for multi-token / >6-expert cases. */
        const uint32_t use_direct_down_sumN =
            n_tokens == 1u && n_expert >= 1u && n_expert <= 6u &&
            getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") == NULL;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256, 0, ds4_decode_stream()>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
            const uint64_t offsets_bytes = ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t);
            const uint64_t cursors_bytes = (uint64_t)n_total_expert * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + 256u;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + 256u) : 0u;
            const uint64_t tile_offsets_bytes = ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? ((uint64_t)n_total_expert + 1ull) * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                ok = cuda_ok(cudaMemset(counts, 0, counts_bytes), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256, 0, ds4_decode_stream()>>>(
                        counts,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1, 0, ds4_decode_stream()>>>(offsets, cursors, counts, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256, 0, ds4_decode_stream()>>>(
                        sorted_pairs,
                        cursors,
                        selected_ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, ds4_decode_stream()>>>(tile_offsets, tile_total, counts, expert_tile_m, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<1, 384, 0, ds4_decode_stream()>>>(tile_experts, tile_starts, tile_offsets, counts, expert_tile_m, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1, 0, ds4_decode_stream()>>>(tile16_offsets, tile16_total, counts, 16u, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<1, 384, 0, ds4_decode_stream()>>>(tile16_experts, tile16_starts, tile16_offsets, counts, 16u, n_total_expert);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            if (ok && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256, 0, ds4_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256, 0, ds4_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile8_row32_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256, 0, ds4_decode_stream()>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256, 0, ds4_decode_stream()>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    selected_ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (use_decode_lut_gate && q4k_path) {
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        selected_ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256, 0, ds4_decode_stream()>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            if (use_direct_down_sumN) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sumN_qwarp32_kernel<<<sgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else if (down_iq2xxs) {
                    moe_down_sumN_iq2xxs_qwarp32_kernel<<<sgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                } else {
                    moe_down_sumN_qwarp32_kernel<<<sgrid, 256, 0, ds4_decode_stream()>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        selected_ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim,
                        n_expert);
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sumN) {
                /* The direct decode kernel writes the final token row. */
            } else if (sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256, 0, ds4_decode_stream()>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256, 0, ds4_decode_stream()>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u && down_iq2xxs) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_iq2xxs_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256, 0, ds4_decode_stream()>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256, 0, ds4_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256, 0, ds4_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                moe_down_qwarp32_kernel<<<dgrid, 256, 0, ds4_decode_stream()>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    selected_ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sumN) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
        ok = cuda_moe_temp_weights_release(&gate_tmp, &up_tmp, &down_tmp) && ok;
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256, 0, ds4_decode_stream()>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            selected_ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256, 0, ds4_decode_stream()>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            selected_ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    ok = cuda_moe_temp_weights_release(&gate_tmp, &up_tmp, &down_tmp) && ok;
    return ok;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x) {
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, 1,
                             0u /* one-tensor path has no layer id; streaming LRU
                                 * disambiguates layers by per-layer gate/up/down
                                 * offsets in cuda_stream_expert_cache_find, so a
                                 * constant layer here is correct (SEAM 3). */);
}
extern "C" int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, n_tokens,
                             layer_index /* SEAM 3: real layer id for streaming LRU */);
}
extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256, 0, ds4_decode_stream()>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256, 0, ds4_decode_stream()>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256, 0, ds4_decode_stream()>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256, 0, ds4_decode_stream()>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (getenv("DS4_CUDA_DISABLE_HC_SPLIT_NORM_FUSED") == NULL) {
        if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
            n_embd == 0 || n_hc != 4) {
            return 0;
        }
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t mix_bytes = mix_hc * sizeof(float);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
            norm_out->bytes < out->bytes ||
            scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size ||
            (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
            return 0;
        }
        uint64_t n_rows = out->bytes / out_row_bytes;
        if (n_rows == 1) {
            if (mix->bytes < n_rows * mix_bytes ||
                split->bytes < n_rows * mix_bytes ||
                residual_hc->bytes < n_rows * residual_row_bytes) {
                return 0;
            }
            const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
                    3ull * sizeof(float), "hc_scale");
            const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
                    mix_bytes, "hc_base");
            const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                    (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
            if (!scale || !base || !norm_w) return 0;
            hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256, 0, ds4_decode_stream()>>>(
                    (float *)out->ptr,
                    (float *)norm_out->ptr,
                    (float *)split->ptr,
                    (const float *)mix->ptr,
                    (const float *)residual_hc->ptr,
                    scale,
                    base,
                    norm_w,
                    n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
        }
    }
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out, out, model_map, model_size,
                                            norm_weight_offset, n_embd, norm_eps);
}
extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256, 0, ds4_decode_stream()>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
extern "C" int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
extern "C" int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}
extern "C" int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256, 0, ds4_decode_stream()>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}
extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (getenv("DS4_CUDA_DISABLE_Q8_HC_EXPAND_FUSED") == NULL) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "q8_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}
