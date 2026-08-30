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
__global__ __launch_bounds__(P7_THREADS, 6) void tendency_fused_p7_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
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
  const int node1 = tid;
  const int node2 = tid + 256;
  const int elem_offset = elem * 512;
  const int face_offset = elem * 384;
  const int npoint = 512 * Ne;
  const int nface = 384 * Ne;

  if (tid < 64) {
    sD1D[tid] = D1D[tid];
  } else if (tid < 112) {
    sLift[tid - 64] = Lift1D[tid - 64];
  }

  const int idx1 = elem_offset + node1;
  const int idx2 = elem_offset + node2;
  {
    const double q1 = q[idx1], u1 = u[idx1], v1 = v[idx1], w1 = w[idx1];
    sFluxX[node1] = q1 * u1;
    sFluxY[node1] = q1 * v1;
    sFluxZ[node1] = q1 * w1;
    p7_stage_mfaces(sMface, sMyz, node1, q1, u1, v1, w1);
  }
  {
    const double q2 = q[idx2], u2 = u[idx2], v2 = v[idx2], w2 = w[idx2];
    sFluxX[node2] = q2 * u2;
    sFluxY[node2] = q2 * v2;
    sFluxZ[node2] = q2 * w2;
    p7_stage_mfaces(sMface, sMyz, node2, q2, u2, v2, w2);
  }
  __syncthreads();

  int fp = tid;
  int fidx = face_offset + fp;
  int iP = VMapP[fidx] - 1;
  const double fn1 = normal_fn[fidx];
  const double fn2 = normal_fn[fidx + nface];
  const double fn3 = normal_fn[fidx + 2 * nface];
  double qM, VelM;
  if ((fp & 64) != 0) {
    const double *const m =
        sMface + (((fp & 128) != 0) ? P7_XFACE_PLANE : 0) + (fp & 63);
    qM = m[0];
    VelM = m[144] * fn1 + m[288] * fn2 + m[432] * fn3;
  } else {
    const double *const m = sMyz + (((fp & 128) != 0) ? 72 : 0) + (fp & 63);
    qM = m[0];
    VelM = m[288] * fn1 + m[576] * fn2 + m[864] * fn3;
  }
  double qP = __ldg(q + iP);
  double VelP = __ldg(u + iP) * fn1 + __ldg(v + iP) * fn2 + __ldg(w + iP) * fn3;
  double alpha = 0.5 * fabs(VelP + VelM);
  sflux_bnd[fp] =
      0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  if (tid < 128) {
    fp = tid + 256;
    fidx = face_offset + fp;
    iP = VMapP[fidx] - 1;
    const double *const m = sMyz + ((fp >= 320) ? 216 : 144) + (fp & 63);
    qM = m[0];
    VelM = m[288] * normal_fn[fidx] + m[576] * normal_fn[fidx + nface] +
           m[864] * normal_fn[fidx + 2 * nface];
    qP = __ldg(q + iP);
    VelP = __ldg(u + iP) * normal_fn[fidx] + __ldg(v + iP) * normal_fn[fidx + nface] +
           __ldg(w + iP) * normal_fn[fidx + 2 * nface];
    alpha = 0.5 * fabs(VelP + VelM);
    sflux_bnd[fp] =
        0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  }
  __syncthreads();

  const int i = node1 & 7;
  const int j = (node1 >> 3) & 7;
  const int k1 = node1 >> 6;
  const int k2 = k1 + 4;
  double sum_x1 = 0.0, sum_y1 = 0.0, sum_z1 = 0.0;
  double sum_x2 = 0.0, sum_y2 = 0.0, sum_z2 = 0.0;
#pragma unroll
  for (int l = 0; l < 8; ++l) {
    const int ix1 = l + j * 8 + k1 * 64;
    const int iy1 = i + l * 8 + k1 * 64;
    const int iz = i + j * 8 + l * 64;
    const double mat_x = sD1D[i + l * 8];
    const double mat_y = sD1D[j + l * 8];
    const double mat_z1 = sD1D[k1 + l * 8];
    const double mat_z2 = sD1D[k2 + l * 8];
    sum_x1 = fma(mat_x, sFluxX[ix1], sum_x1);
    sum_y1 = fma(mat_y, sFluxY[iy1], sum_y1);
    sum_z1 = fma(mat_z1, sFluxZ[iz], sum_z1);
    sum_x2 = fma(mat_x, sFluxX[ix1 + 256], sum_x2);
    sum_y2 = fma(mat_y, sFluxY[iy1 + 256], sum_y2);
    sum_z2 = fma(mat_z2, sFluxZ[iz], sum_z2);
  }

  const int face11 = i + k1 * 8;
  const int face12 = 64 + j + k1 * 8;
  const int face13 = 128 + i + k1 * 8;
  const int face14 = 192 + j + k1 * 8;
  const int face15 = 256 + i + j * 8;
  const int face16 = 320 + i + j * 8;
  const int face21 = i + k2 * 8;
  const int face22 = 64 + j + k2 * 8;
  const int face23 = 128 + i + k2 * 8;
  const int face24 = 192 + j + k2 * 8;
  const double lift1 = sLift[j] * sflux_bnd[face11] +
                       sLift[i + 8] * sflux_bnd[face12] +
                       sLift[j + 16] * sflux_bnd[face13] +
                       sLift[i + 24] * sflux_bnd[face14] +
                       sLift[k1 + 32] * sflux_bnd[face15] +
                       sLift[k1 + 40] * sflux_bnd[face16];
  const double lift2 = sLift[j] * sflux_bnd[face21] +
                       sLift[i + 8] * sflux_bnd[face22] +
                       sLift[j + 16] * sflux_bnd[face23] +
                       sLift[i + 24] * sflux_bnd[face24] +
                       sLift[k2 + 32] * sflux_bnd[face15] +
                       sLift[k2 + 40] * sflux_bnd[face16];

  dqdt[idx1] = -(Escale[idx1] * sum_x1 + Escale[idx1 + npoint] * sum_y1 +
                 Escale[idx1 + 2 * npoint] * sum_z1 + lift1);
  dqdt[idx2] = -(Escale[idx2] * sum_x2 + Escale[idx2 + npoint] * sum_y2 +
                 Escale[idx2 + 2 * npoint] * sum_z2 + lift2);
}

extern "C" void launch_tendency_fused_p7(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  tendency_fused_p7_cc_kernel<<<Ne, P7_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda_cc("tendency_fused_p7_cc_kernel");
}
