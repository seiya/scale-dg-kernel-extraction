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

__global__ __launch_bounds__(P7_THREADS, P7_BPSM) void tendency_fused_p7_cc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ double sD1D[64];
  __shared__ double sLift[48];
  __shared__ double sflux_bnd[384];
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
    const double q1 = q[idx1];
    sFluxX[node1] = q1 * u[idx1];
    sFluxY[node1] = q1 * v[idx1];
    sFluxZ[node1] = q1 * w[idx1];
  }
  {
    const double q2 = q[idx2];
    sFluxX[node2] = q2 * u[idx2];
    sFluxY[node2] = q2 * v[idx2];
    sFluxZ[node2] = q2 * w[idx2];
  }

  int fp = tid;
  int fidx = face_offset + fp;
  int iM = VMapM[fidx] - 1;
  int iP = VMapP[fidx] - 1;
  double qM = q[iM];
  double qP = q[iP];
  double VelM = u[iM] * normal_fn[fidx] + v[iM] * normal_fn[fidx + nface] +
                w[iM] * normal_fn[fidx + 2 * nface];
  double VelP = u[iP] * normal_fn[fidx] + v[iP] * normal_fn[fidx + nface] +
                w[iP] * normal_fn[fidx + 2 * nface];
  double alpha = 0.5 * fabs(VelP + VelM);
  sflux_bnd[fp] =
      0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  if (tid < 128) {
    fp = tid + 256;
    fidx = face_offset + fp;
    iM = VMapM[fidx] - 1;
    iP = VMapP[fidx] - 1;
    qM = q[iM];
    qP = q[iP];
    VelM = u[iM] * normal_fn[fidx] + v[iM] * normal_fn[fidx + nface] +
           w[iM] * normal_fn[fidx + 2 * nface];
    VelP = u[iP] * normal_fn[fidx] + v[iP] * normal_fn[fidx + nface] +
           w[iP] * normal_fn[fidx + 2 * nface];
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
