// Unloaded global-memory latency and the latency-vs-MLP curve.
//
// reports/subpipe_transfer_model.md section 5H leaves the second equation
// open on one quantity: the exposed latency per request is L / MLP, and no
// counter gives either factor separately.  Section 5N rejects three ways of
// getting MLP out of the existing counters.  What is left is to measure the
// machine half, L, directly, so that the 0.07 unloaded-latency share in
// analysis/subpipe_predict.py stops being a two-point fit.
//
// Method: a pointer chase over a buffer far larger than L2, so every step
// misses to DRAM, with CHAINS independent chases per thread.  CHAINS = 1 is
// the unloaded latency (MLP = 1 by construction).  Raising CHAINS raises the
// memory-level parallelism of one warp and the per-access cost falls until
// the machine runs out of misses in flight, which is the same curve the DG
// kernels sit on.
//
// Out of role for the benchmark (AGENTS.md): this is a machine
// characterisation, not a DG kernel.  It is not in production dispatch.
//
// Build and run (login GPU is fine; this is not a profiler):
//   nvcc -O3 -arch=sm_100 -o global_latency_probe jobs/global_latency_probe.cu
//   ./global_latency_probe
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)

template <int CHAINS>
__global__ void chase(const uint32_t* __restrict__ idx, uint32_t n,
                      uint32_t steps, uint32_t* out, uint32_t stride)
{
    uint32_t p[CHAINS];
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) p[c] = (threadIdx.x + c * stride) % n;
    for (uint32_t s = 0; s < steps; ++s) {
#pragma unroll
        for (int c = 0; c < CHAINS; ++c) p[c] = idx[p[c]];
    }
    uint32_t acc = 0;
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) acc += p[c];
    // Not `if (threadIdx.x == 1024)`: nvcc knows threadIdx.x < 1024 and drops
    // the whole chase as dead code (measured 1.5 ns/access, i.e. nothing).
    if (acc == 0xFFFFFFFFu) out[0] = acc;
}

template <int CHAINS>
static double run(const uint32_t* d_idx, uint32_t n, uint32_t* d_out, uint32_t steps)
{
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    chase<CHAINS><<<1, 32>>>(d_idx, n, 64, d_out, n / (CHAINS + 1));   // warm
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    chase<CHAINS><<<1, 32>>>(d_idx, n, steps, d_out, n / (CHAINS + 1));
    cudaEventRecord(b);
    cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1e6 / (double)(steps * CHAINS);          // ns per access
}

int main()
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    // cudaDeviceProp::clockRate is gone in CUDA 13; ask the attribute instead.
    int khz = 0; CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    // 512 MiB of indices: 128x the largest L2 here (126 MB on GB200).
    const uint32_t n = 128u * 1024u * 1024u;
    const uint32_t stride = 4096u / 4u;          // 4 KiB apart: one page, one sector
    std::vector<uint32_t> h(n);
    // A single cycle with a large stride, so consecutive steps cannot share a
    // sector, a line, or a TLB entry that the previous step just brought in.
    uint32_t cur = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t nxt = (uint32_t)(((uint64_t)cur + stride * 7 + 4099) % n);
        h[cur] = nxt; cur = nxt;
    }
    uint32_t *d_idx, *d_out;
    CK(cudaMalloc(&d_idx, (size_t)n * 4)); CK(cudaMalloc(&d_out, 4));
    CK(cudaMemcpy(d_idx, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    const uint32_t steps = 4096;
    printf("device %s  SM %d  clock %.3f GHz  L2 %.1f MB\n", pr.name,
           pr.multiProcessorCount, khz / 1e6, pr.l2CacheSize / 1e6);
    printf("%8s %12s %12s\n", "chains", "ns/access", "SM cycles");
    double f = khz / 1e6;               // GHz (nominal boost)
#define ONE(C) { double t = run<C>(d_idx, n, d_out, steps); \
                 printf("%8d %12.1f %12.1f\n", C, t, t * f); }
    ONE(1) ONE(2) ONE(3) ONE(4) ONE(6) ONE(8) ONE(12) ONE(16) ONE(24) ONE(32)
#undef ONE
    cudaFree(d_idx); cudaFree(d_out);
    return 0;
}
