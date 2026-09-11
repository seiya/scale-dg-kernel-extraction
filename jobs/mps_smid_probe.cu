// Which SMs does a client actually get?
//
// reports/subpipe_transfer_model.md 5Z reads CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
// as "the client runs on that fraction of the SMs", and infers from the timing
// that it worked.  That is an inference, not a measurement: a percentage that
// were silently ignored, rounded to a coarse granularity, or applied as a
// thread-slot limit rather than an SM partition would all leave timings that
// still look plausible.
//
// This reads %smid from every block and counts the distinct SMs that ran any
// block, which is what the claim actually needs.  Device-reported SM count is
// printed too: MPS does not change it, so it is the denominator.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void stamp(unsigned char* hit)
{
    unsigned s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    hit[s] = 1;
}
int main()
{
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    const int slots = 4096;                       // more than any smid
    unsigned char *d, h[slots];
    cudaMalloc(&d, slots);
    cudaMemset(d, 0, slots);
    // Many small blocks so every SM the client may use receives at least one.
    stamp<<<nsm * 200, 32>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e) { printf("CUDA %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, slots, cudaMemcpyDeviceToHost);
    int used = 0, maxid = -1;
    for (int i = 0; i < slots; ++i) if (h[i]) { ++used; maxid = i; }
    printf("device_sm=%d used_sm=%d max_smid=%d fraction=%.4f\n",
           nsm, used, maxid, (double)used / nsm);
    cudaFree(d);
    return 0;
}
