// Shared-memory latency probe -- OUT OF ROLE, NOT PART OF ANY BENCHMARK PATH.
//
// Why this exists
// ---------------
// measurement_inventory.md 23.6: on A100, twelve sequential processes running
// the SAME p=511 FUSED_TC binary on the SAME GPU differ by 2.3-2.9% in kernel
// time, and every quantity ncu can count is identical -- instruction count to
// the digit, DRAM bytes to 0.000%, MIO instructions issued bit-identical, LSU
// writeback-active cycles to 0.06%, occupancy, CTA distribution, carveout,
// register count.  The extra time appears only as short_scoreboard and
// mio_throttle waiting, i.e. as LATENCY on the shared/MIO path with no extra
// work behind it.  Sections 22 and 23 excluded all three candidates 21.19
// named (CTA-to-SM assignment, L1/shared carveout, ECC).
//
// That leaves one question a counter cannot answer: is the varying shared
// latency a property of THIS KERNEL, or of the machine?  This probe answers it
// by removing the kernel.  It contains no DG arithmetic, no mma, no global
// traffic in the shared test, and no dependence on the extracted interface: it
// is a dependent chain of shared-memory loads and nothing else.
//
//   Same continuum across processes here  -> the effect is the machine's, and
//       nothing in the DG implementation causes or can fix it.
//   Flat here                             -> the effect needs this kernel's
//       shared access pattern, and the search goes back into the kernel.
//
// Launch shape deliberately matches the real kernel as measured by ncu in
// section 23.1, so that CTA scheduling and the carveout are not free
// variables: 32 768 blocks, 128 threads, 32 768 B of static shared per block,
// which is what gives launch__occupancy_limit_shared_mem = 3 on A100.
//
// AGENTS.md: this is an out-of-role ablation.  It is not dispatched from any
// DqdtKernel_Type, its timing is never reported as a path result, and it must
// not be built into scale-dg_extraction.
//
// Build (per machine):
//   nvcc -O3 -arch=sm_80  -o shmem_probe jobs/shmem_latency_probe.cu   # A100
//   nvcc -O3 -arch=sm_90  -o shmem_probe jobs/shmem_latency_probe.cu   # H100
//   nvcc -O3 -arch=sm_100 -o shmem_probe jobs/shmem_latency_probe.cu   # GB200
// Run:  ./shmem_probe [reps]        (default 3)

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call,        \
                   cudaGetErrorString(err_));                                  \
      return 2;                                                                \
    }                                                                          \
  } while (0)

static constexpr int kBlocks = 32768;   // = the real kernel's grid
static constexpr int kThreads = 128;    // = the real kernel's block
static constexpr int kInts = 8192;      // 32 768 B of static shared
static constexpr int kStride = 128;     // 128 ints = 512 B: conflict-free
static constexpr int kIters = 4096;

// Dependent chain of shared loads.  Each step is 512 B further on, so the 32
// lanes of a warp read 32 consecutive ints -- one wavefront, no bank conflict
// -- and every load waits for the previous one.  The measured cycles are
// therefore shared-memory LATENCY, not bandwidth.
__global__ __launch_bounds__(kThreads) void shmem_chase(
    unsigned long long *cycles, int *sink) {
  __shared__ int buf[kInts];
  const int tid = threadIdx.x;
  for (int i = tid; i < kInts; i += kThreads) buf[i] = (i + kStride) % kInts;
  __syncthreads();

  int idx = tid;
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < kIters; ++i) idx = buf[idx];
  const unsigned long long t1 = clock64();

  if (tid == 0) cycles[blockIdx.x] = t1 - t0;
  if (idx < 0) sink[0] = idx;  // keep the chain live
}

// Control on the other path: the same dependent chain through global memory,
// so that a process which is slow on shared can be told apart from one that is
// slow on everything.  Section 23 measured long_scoreboard FALLING while
// short_scoreboard rose, so the prediction is that this one does not move.
__global__ __launch_bounds__(kThreads) void global_chase(
    const int *chain, unsigned long long *cycles, int *sink) {
  const int tid = threadIdx.x;
  int idx = (blockIdx.x * kThreads + tid) % (kBlocks * kThreads);
  const unsigned long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < 512; ++i) idx = chain[idx];
  const unsigned long long t1 = clock64();
  if (tid == 0) cycles[blockIdx.x] = t1 - t0;
  if (idx < 0) sink[0] = idx;
}

static void report(const char *tag, std::vector<unsigned long long> &v,
                   int iters, float ms) {
  std::sort(v.begin(), v.end());
  const double med = static_cast<double>(v[v.size() / 2]);
  const double p10 = static_cast<double>(v[v.size() / 10]);
  const double p90 = static_cast<double>(v[(v.size() * 9) / 10]);
  double sum = 0.0;
  for (unsigned long long x : v) sum += static_cast<double>(x);
  const double mean = sum / static_cast<double>(v.size());
  std::printf(
      "%-8s cycles/access mean=%.4f median=%.4f p10=%.4f p90=%.4f "
      "min=%.4f max=%.4f  kernel_ms=%.4f\n",
      tag, mean / iters, med / iters, p10 / iters, p90 / iters,
      static_cast<double>(v.front()) / iters,
      static_cast<double>(v.back()) / iters, ms);
}

int main(int argc, char **argv) {
  const int reps = (argc > 1) ? std::atoi(argv[1]) : 3;

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0;
  // cudaDeviceProp::clockRate was removed in CUDA 13; the attribute query
  // works on every version this repository builds with.
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  std::printf("device=%s sms=%d shared_per_block_optin=%zu clock_khz=%d\n",
              prop.name, prop.multiProcessorCount,
              static_cast<size_t>(prop.sharedMemPerBlockOptin), clock_khz);

  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  int *d_chain = nullptr;
  const size_t chain_n = static_cast<size_t>(kBlocks) * kThreads;
  CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long) * kBlocks));
  CHECK(cudaMalloc(&d_sink, sizeof(int)));
  CHECK(cudaMalloc(&d_chain, sizeof(int) * chain_n));

  // A stride large enough to miss L1 and L2 on every step.
  std::vector<int> chain(chain_n);
  const size_t stride = 1u << 20;  // 4 MiB apart
  for (size_t i = 0; i < chain_n; ++i)
    chain[i] = static_cast<int>((i + stride) % chain_n);
  CHECK(cudaMemcpy(d_chain, chain.data(), sizeof(int) * chain_n,
                   cudaMemcpyHostToDevice));

  cudaEvent_t beg, end;
  CHECK(cudaEventCreate(&beg));
  CHECK(cudaEventCreate(&end));
  std::vector<unsigned long long> host(kBlocks);

  for (int r = 0; r < reps; ++r) {
    CHECK(cudaEventRecord(beg));
    shmem_chase<<<kBlocks, kThreads>>>(d_cycles, d_sink);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaGetLastError());
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, beg, end));
    CHECK(cudaMemcpy(host.data(), d_cycles,
                     sizeof(unsigned long long) * kBlocks,
                     cudaMemcpyDeviceToHost));
    std::printf("rep %d ", r + 1);
    report("shared", host, kIters, ms);

    CHECK(cudaEventRecord(beg));
    global_chase<<<kBlocks, kThreads>>>(d_chain, d_cycles, d_sink);
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));
    CHECK(cudaGetLastError());
    CHECK(cudaEventElapsedTime(&ms, beg, end));
    CHECK(cudaMemcpy(host.data(), d_cycles,
                     sizeof(unsigned long long) * kBlocks,
                     cudaMemcpyDeviceToHost));
    std::printf("rep %d ", r + 1);
    report("global", host, 512, ms);
  }

  CHECK(cudaFree(d_cycles));
  CHECK(cudaFree(d_sink));
  CHECK(cudaFree(d_chain));
  return 0;
}
