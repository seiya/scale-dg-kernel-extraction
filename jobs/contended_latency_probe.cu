// Global-memory latency as a function of how busy DRAM is: L(rho).
//
// reports/subpipe_transfer_model.md 5N.10 fits one constant, QUEUE_K = 1.20,
// to two points (A100 1.17, H100 1.23), and 5P showed that a clock change
// cannot give a third: dropping the SM clock moves the unloaded-latency ratio
// and the bandwidth-pressure ratio by the SAME factor, so it tests the product
// and not the factors.
//
// The way out is to move the DEMAND rather than the supply.  A second stream
// streams memory at a controlled rate while the pointer chase of
// jobs/global_latency_probe.cu runs at MLP = 1 on one SM.  That sweeps the DRAM
// utilisation rho on ONE machine with the clock, the SM count, the ISA and the
// chase itself all fixed, and measures the queue inflation
//
//     Q(rho) = L(rho) / L(0)
//
// directly -- the quantity 5N.10 could only infer as a ratio between machines,
// and the one no ncu counter separates from MLP (5N.1).  With Q(rho) measured
// per machine, QUEUE_K stops being a fitted constant: a kernel's own measured
// dram__throughput picks its Q off this curve.
//
// Out of role (AGENTS.md): machine characterisation, not a DG kernel.
//
//   nvcc -O3 -arch=sm_100 -o contended_latency_probe jobs/contended_latency_probe.cu
//   ./contended_latency_probe
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <nvml.h>

#define CK(x) do { cudaError_t e=(x); if(e){printf("CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;} } while(0)

__global__ void chase(const uint32_t* __restrict__ idx, uint32_t n,
                      uint32_t steps, uint32_t* out)
{
    uint32_t p = threadIdx.x % n;
    for (uint32_t s = 0; s < steps; ++s) p = idx[p];
    if (p == 0xFFFFFFFFu) out[0] = p;
}

// Streams `iters` passes over `buf` with a grid stride.  Reads dominate, which
// is what the DG kernels' DRAM traffic looks like; the store keeps it live.
__global__ void loader(const double* __restrict__ buf, size_t n, int iters,
                       double* sink, int delay)
{
    double acc = 0.0;
    for (int it = 0; it < iters; ++it)
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
             i += (size_t)gridDim.x * blockDim.x) {
            acc += buf[i];
            // Throttle: FP64 work between loads lowers the request rate at
            // FIXED occupancy.  Sweeping the loader's WIDTH instead would move
            // the queue depth and the utilisation together, which is what made
            // the first sweep non-monotone (three widths at 4.6-4.9 TB/s gave
            // Q = 5.55, 3.36 and 2.22).
            for (int d = 0; d < delay; ++d) acc = acc * 1.0000001 + 1e-12;
        }
    if (acc == 1.0e308) sink[0] = acc;
}

int main()
{
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    int khz = 0; CK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    const uint32_t n = 128u * 1024u * 1024u;          // 512 MiB of indices
    const uint32_t stride = 4096u / 4u;
    std::vector<uint32_t> h(n);
    uint32_t cur = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t nxt = (uint32_t)(((uint64_t)cur + stride * 7 + 4099) % n);
        h[cur] = nxt; cur = nxt;
    }
    uint32_t *d_idx, *d_out; double *d_buf, *d_sink, *d_small;
    const size_t nb = 256u * 1024u * 1024u;           // 2 GiB of loader data
    // Control: the same loader, same occupancy, same instruction mix, on a
    // buffer that fits in L2 on all three machines (40 MB is the smallest).
    // Any inflation it causes is SM contention, not DRAM queueing -- without
    // it the sweep reads Q = 5.75 at 33 GB/s, which is obviously not a memory
    // queue.  Q_dram = Q_measured / Q_control.
    const size_t ns_small = 1u * 1024u * 1024u;       // 8 MiB
    CK(cudaMalloc(&d_idx, (size_t)n * 4)); CK(cudaMalloc(&d_out, 4));
    CK(cudaMalloc(&d_buf, nb * sizeof(double))); CK(cudaMalloc(&d_sink, 8));
    CK(cudaMalloc(&d_small, ns_small * sizeof(double)));
    CK(cudaMemset(d_small, 0, ns_small * sizeof(double)));
    CK(cudaMemcpy(d_idx, h.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_buf, 0, nb * sizeof(double)));

    nvmlDevice_t nvd; unsigned int sm_clk = 0, mem_clk = 0;
    bool nvml_ok = (nvmlInit_v2() == NVML_SUCCESS) &&
                   (nvmlDeviceGetHandleByIndex_v2(0, &nvd) == NVML_SUCCESS);

    int lo = 0, hi = 0; cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStream_t sChase, sLoad;
    CK(cudaStreamCreateWithPriority(&sChase, cudaStreamNonBlocking, hi));   // high
    CK(cudaStreamCreateWithPriority(&sLoad, cudaStreamNonBlocking, lo));    // low
    const uint32_t steps = 2048;
    // Bring the GPU to a steady clock state first.  Measured on GB200: without
    // this the unloaded point reads 446 ns and the lightly-loaded one 255 ns,
    // which is not physical -- the idle GPU clocks its memory down, so the
    // "unloaded" latency was being taken on a slower machine than the loaded
    // one.  Every point below is taken with the loader having been running.
    for (int w = 0; w < 3; ++w) {
        loader<<<pr.multiProcessorCount * 4, 256, 0, sLoad>>>(d_buf, nb, 2, d_sink, 0);
        CK(cudaDeviceSynchronize());
    }

    printf("device %s  SM %d  clock %.3f GHz\n", pr.name, pr.multiProcessorCount,
           khz / 1e6);
    printf("%10s %12s %12s %10s %10s %9s %7s %8s\n", "delay", "load GB/s",
           "chase ns", "cycles", "Q=L/L0", "cover", "smMHz", "memMHz");
    double L0 = 0.0;
    // 0 blocks = the unloaded point; then a sweep of loader widths.
    // Fixed width (32 blocks per SM), swept throttle.  delay = -1 is the
    // unloaded point.
    const int delays[] = {-1, 4096, 1024, 256, 64, 32, 16, 8, 4, 2, 1, 0};
    for (int pass = 0; pass < 2; ++pass) {
      const double* lbuf = pass ? d_small : d_buf;
      size_t ln = pass ? ns_small : nb;
      printf("-- %s loader\n", pass ? "L2-resident (contention control)" : "DRAM");
      L0 = 0.0;
      for (int wi = 0; wi < (int)(sizeof(delays) / sizeof(int)); ++wi) {
        int delay = delays[wi];
        int blocks = delay < 0 ? 0 : pr.multiProcessorCount * 32;
        cudaEvent_t ca, cb, la, lb;
        cudaEventCreate(&ca); cudaEventCreate(&cb);
        cudaEventCreate(&la); cudaEventCreate(&lb);
        chase<<<1, 32, 0, sChase>>>(d_idx, n, 64, d_out);       // warm
        CK(cudaDeviceSynchronize());
        // The loader must cover the whole chase, so scale its work with how
        // slow it will be: a narrow loader is bandwidth-poor but long-lived.
        int iters = 1;
        if (blocks > 0) {                    // calibrate: how long is one pass?
            cudaEvent_t pa, pb; cudaEventCreate(&pa); cudaEventCreate(&pb);
            cudaEventRecord(pa, sLoad);
            loader<<<blocks, 256, 0, sLoad>>>(lbuf, ln, 1, d_sink, delay);
            cudaEventRecord(pb, sLoad);
            CK(cudaDeviceSynchronize());
            float pms = 0; cudaEventElapsedTime(&pms, pa, pb);
            // chase takes steps * ~0.5 us; want the loader ~4x longer
            double want_ms = 4.0 * steps * 0.5e-3;
            iters = (int)(want_ms / pms) + 1;
            if (iters > 4096) iters = 4096;
            cudaEventDestroy(pa); cudaEventDestroy(pb);
        }
        if (blocks > 0) {
            cudaEventRecord(la, sLoad);
            loader<<<blocks, 256, 0, sLoad>>>(lbuf, ln, iters, d_sink, delay);
            cudaEventRecord(lb, sLoad);
        }
        cudaEventRecord(ca, sChase);
        chase<<<1, 32, 0, sChase>>>(d_idx, n, steps, d_out);
        cudaEventRecord(cb, sChase);
        CK(cudaDeviceSynchronize());
        float cms = 0, lms = 0;
        cudaEventElapsedTime(&cms, ca, cb);
        double gbs = 0.0;
        if (blocks > 0) {
            cudaEventElapsedTime(&lms, la, lb);
            gbs = (double)ln * sizeof(double) * iters / (lms * 1e-3) / 1e9;
        }
        if (nvml_ok) {                       // sampled while the loader still runs
            nvmlDeviceGetClockInfo(nvd, NVML_CLOCK_SM, &sm_clk);
            nvmlDeviceGetClockInfo(nvd, NVML_CLOCK_MEM, &mem_clk);
        }
        double ns = cms * 1e6 / steps;
        // Was the chase actually covered?  If the loader finished first the
        // point is contaminated, so report the overlap rather than hide it.
        double cover = blocks > 0 ? lms / cms : 1.0;
        if (wi == 0) L0 = ns;
        printf("%10d %12.1f %12.1f %10.1f %10.3f %9.2f %7u %8u\n", delay, gbs,
               ns, ns * sm_clk / 1e3, ns / L0, cover, sm_clk, mem_clk);
        cudaEventDestroy(ca); cudaEventDestroy(cb);
        cudaEventDestroy(la); cudaEventDestroy(lb);
      }
    }
    return 0;
}
