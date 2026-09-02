#include <cuda_runtime.h>
#include <cstdio>

#include "fused_kernel_geom.h"

// CUDAFORTRAN_FUSED: CUDA-core fused tendency.  Natural-order shared panels
// and length-Nq inner products; not the MMA fragment schedule.
// Restored from the Fortran kernel at 2dadc41^ (launch_bounds(256,8)).

extern cudaStream_t dg_cuda_stream;

static void check_cuda_cc(const char *what)
{
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
  }
}

// p=7 CC output-tile knobs (p7_gap_study.md sec 5).  P7_CC_THREADS threads
// cover the 512 nodes, so each thread owns 512 / P7_CC_THREADS outputs that
// share mat_x, mat_y and sFluxZ.  P7_THREADS itself is shared with the TC
// kernel (fused_kernel_geom.h) and must not be touched here.
#ifndef P7_CC_THREADS
#define P7_CC_THREADS 128
#endif
#ifndef P7_CC_BPSM
#define P7_CC_BPSM 5
#endif
#define P7_CC_NOUT (512 / P7_CC_THREADS)
#define P7_CC_KSTEP (P7_CC_THREADS / 64)

#define P7_XFACE_PLANE 72
__device__ __forceinline__ void p7_stage_mfaces(double *sMx, double *sMyz,
                                               int node, double qn, double u,
                                               double v, double w)
{
  const int i = node & 7;
  const int j = (node >> 3) & 7;
  const int k = node >> 6;
  if (i == 7 || i == 0) {
    double *const m = sMx + ((i == 7) ? 0 : P7_XFACE_PLANE) +
                      ((node >> 3) & 7) + (k << 3);
    m[0] = qn;
    m[144] = u;
    m[288] = v;
    m[432] = w;
  }
  if (j == 0 || j == 7) {
    const int pt = i + (k << 3);
    double *const m = sMyz + ((j == 0) ? 0 : 72) + pt;
    m[0] = qn;
    m[288] = u;
    m[576] = v;
    m[864] = w;
  }
  if (k == 0 || k == 7) {
    const int pt = i + (j << 3);
    double *const m = sMyz + ((k == 0) ? 144 : 216) + pt;
    m[0] = qn;
    m[288] = u;
    m[576] = v;
    m[864] = w;
  }
}

// 6 blocks/SM: 40 registers, enough to absorb sMface without spill.
// Occupancy 75%.  8 blocks spill; 4/5/7 lose on occupied GPU A/B.
__global__ __launch_bounds__(P7_CC_THREADS, P7_CC_BPSM) void tendency_fused_p7_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  constexpr int NOUT = P7_CC_NOUT;
  __shared__ double sD1D[64];
  __shared__ double sLift[48];
  __shared__ double sflux_bnd[384];
  // M-side q,u,v,w of the two x-normal faces.  Face points 64-127 / 192-255
  // gather with stride 8; the same element's volume loads already hold the
  // values.  Layout matches cuda_dg_kernels_tc.cu (pad 72, field stride 144).
  __shared__ double sMface[4 * 144];
  // M-side of faces 1,3 (offset 0/72) and 5,6 (offset 144/216), field stride 288.
  __shared__ double sMyz[4 * 288];
  __shared__ double sFluxX[512], sFluxY[512], sFluxZ[512];

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int elem_offset = elem * 512;
  const int face_offset = elem * 384;
  const int npoint = 512 * Ne;
  const int nface = 384 * Ne;

  if (tid < 64) {
    sD1D[tid] = D1D[tid];
  } else if (tid < 112) {
    sLift[tid - 64] = Lift1D[tid - 64];
  }

#pragma unroll
  for (int m = 0; m < NOUT; ++m) {
    const int node = tid + P7_CC_THREADS * m;
    const int idx = elem_offset + node;
    const double qn = q[idx], un = u[idx], vn = v[idx], wn = w[idx];
    sFluxX[node] = qn * un;
    sFluxY[node] = qn * vn;
    sFluxZ[node] = qn * wn;
    p7_stage_mfaces(sMface, sMyz, node, qn, un, vn, wn);
  }
  __syncthreads();

  // Faces 1-4 (fp < 256): the x-normal pair lives in sMface, the rest in
  // sMyz.  Faces 5,6 (fp >= 256) are always sMyz.  Splitting the two ranges
  // keeps the selector free of a runtime `fp < 256` test, which is what the
  // 256-thread form compiled to before P7_CC_THREADS became a knob.
#pragma unroll
  for (int pl = 0; pl < 256 / P7_CC_THREADS; ++pl) {
    const int fp = tid + P7_CC_THREADS * pl;
    const int fidx = face_offset + fp;
    const int iP = VMapP[fidx] - 1;
    const double fn1 = normal_fn[fidx];
    const double fn2 = normal_fn[fidx + nface];
    const double fn3 = normal_fn[fidx + 2 * nface];
    double qM, VelM;
    if ((fp & 64) != 0) {
      const double *const mp =
          sMface + (((fp & 128) != 0) ? P7_XFACE_PLANE : 0) + (fp & 63);
      qM = mp[0];
      VelM = mp[144] * fn1 + mp[288] * fn2 + mp[432] * fn3;
    } else {
      const double *const mp =
          sMyz + (((fp & 128) != 0) ? 72 : 0) + (fp & 63);
      qM = mp[0];
      VelM = mp[288] * fn1 + mp[576] * fn2 + mp[864] * fn3;
    }
    const double qP = __ldg(q + iP);
    const double VelP =
        __ldg(u + iP) * fn1 + __ldg(v + iP) * fn2 + __ldg(w + iP) * fn3;
    const double alpha = 0.5 * fabs(VelP + VelM);
    sflux_bnd[fp] =
        0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  }
#pragma unroll
  for (int ph = 0; ph < (128 + P7_CC_THREADS - 1) / P7_CC_THREADS; ++ph) {
    const int fp = 256 + tid + P7_CC_THREADS * ph;
    if (fp < 384) {
      const int fidx = face_offset + fp;
      const int iP = VMapP[fidx] - 1;
      const double fn1 = normal_fn[fidx];
      const double fn2 = normal_fn[fidx + nface];
      const double fn3 = normal_fn[fidx + 2 * nface];
      const double *const mp = sMyz + ((fp >= 320) ? 216 : 144) + (fp & 63);
      const double qM = mp[0];
      const double VelM = mp[288] * fn1 + mp[576] * fn2 + mp[864] * fn3;
      const double qP = __ldg(q + iP);
      const double VelP =
          __ldg(u + iP) * fn1 + __ldg(v + iP) * fn2 + __ldg(w + iP) * fn3;
      const double alpha = 0.5 * fabs(VelP + VelM);
      sflux_bnd[fp] =
          0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
    }
  }
  __syncthreads();

  const int i = tid & 7;
  const int j = (tid >> 3) & 7;
  const int k0 = tid >> 6;
  int kk[NOUT];
  double sum_x[NOUT], sum_y[NOUT], sum_z[NOUT];
#pragma unroll
  for (int m = 0; m < NOUT; ++m) {
    kk[m] = k0 + P7_CC_KSTEP * m;
    sum_x[m] = 0.0;
    sum_y[m] = 0.0;
    sum_z[m] = 0.0;
  }
#pragma unroll
  for (int l = 0; l < 8; ++l) {
    const int ix = l + j * 8;
    const int iy = i + l * 8;
    const int iz = i + j * 8 + l * 64;
    const double mat_x = sD1D[i + l * 8];
    const double mat_y = sD1D[j + l * 8];
    const double fz = sFluxZ[iz];
#pragma unroll
    for (int m = 0; m < NOUT; ++m) {
      sum_x[m] = fma(mat_x, sFluxX[ix + kk[m] * 64], sum_x[m]);
      sum_y[m] = fma(mat_y, sFluxY[iy + kk[m] * 64], sum_y[m]);
      sum_z[m] = fma(sD1D[kk[m] + l * 8], fz, sum_z[m]);
    }
  }

  const int face5 = 256 + i + j * 8;
  const int face6 = 320 + i + j * 8;
  const double fb5 = sflux_bnd[face5];
  const double fb6 = sflux_bnd[face6];
  const double lfj = sLift[j];
  const double lfi = sLift[i + 8];
  const double lfj3 = sLift[j + 16];
  const double lfi4 = sLift[i + 24];
#pragma unroll
  for (int m = 0; m < NOUT; ++m) {
    const int k = kk[m];
    const int idx = elem_offset + tid + P7_CC_THREADS * m;
    const double lift = lfj * sflux_bnd[i + k * 8] +
                        lfi * sflux_bnd[64 + j + k * 8] +
                        lfj3 * sflux_bnd[128 + i + k * 8] +
                        lfi4 * sflux_bnd[192 + j + k * 8] +
                        sLift[k + 32] * fb5 + sLift[k + 40] * fb6;
    dqdt[idx] = -(Escale[idx] * sum_x[m] + Escale[idx + npoint] * sum_y[m] +
                  Escale[idx + 2 * npoint] * sum_z[m] + lift);
  }
}

extern "C" void launch_tendency_fused_p7(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  tendency_fused_p7_cc_kernel<<<Ne, P7_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda_cc("tendency_fused_p7_cc_kernel");
}
