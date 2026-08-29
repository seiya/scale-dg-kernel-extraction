#include <cuda_runtime.h>
#include <cstdio>

#include "fused_kernel_geom.h"

// CUDAFORTRAN_FUSED CUDA-core schedule for p=15..255.
// Geometry matches 2dadc41^ Fortran, not fused_kernel_geom.h TC sizes
// (p=31 is 1024 threads, not P31_THREADS=512).

extern cudaStream_t dg_cuda_stream;

#define P31_CC_THREADS 1024
#define P63_CC_THREADS 1024
#define P63_CC_BPE 64
#define P127_CC_THREADS 1024
#define P127_CC_BPE 512
#define P255_CC_BPE 65536

static void check_cuda_cc_hp(const char *what)
{
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
  }
}

__global__ __launch_bounds__(P15_THREADS, 1) void tendency_fused_p15_cc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ double sD1D[256];
  __shared__ double sLift[96];
  __shared__ double sbuf[4096];

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int nd1 = tid;
  const int nd2 = tid + 1024;
  const int nd3 = tid + 2048;
  const int nd4 = tid + 3072;
  const int elem_offset = elem * 4096;
  const int face_offset = elem * 1536;
  const int npoint = 4096 * Ne;
  const int nface = 1536 * Ne;

  if (tid < 256) {
    sD1D[tid] = D1D[tid];
  } else if (tid < 352) {
    sLift[tid - 256] = Lift1D[tid - 256];
  }

  const int idx1 = elem_offset + nd1;
  const int idx2 = elem_offset + nd2;
  const int idx3 = elem_offset + nd3;
  const int idx4 = elem_offset + nd4;
  const double qv1 = q[idx1];
  const double qv2 = q[idx2];
  const double qv3 = q[idx3];
  const double qv4 = q[idx4];

  const int i = tid % 16;
  const int j = (tid / 16) % 16;
  const int k1 = tid / 256;
  const int k2 = k1 + 4;
  const int k3 = k1 + 8;
  const int k4 = k1 + 12;

  sbuf[nd1] = qv1 * u[idx1];
  sbuf[nd2] = qv2 * u[idx2];
  sbuf[nd3] = qv3 * u[idx3];
  sbuf[nd4] = qv4 * u[idx4];
  __syncthreads();

  double s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
  for (int l = 0; l < 16; ++l) {
    const int ib = l + j * 16 + k1 * 256;
    const double dm = sD1D[i + l * 16];
    s1 += dm * sbuf[ib];
    s2 += dm * sbuf[ib + 1024];
    s3 += dm * sbuf[ib + 2048];
    s4 += dm * sbuf[ib + 3072];
  }
  double acc1 = Escale[idx1] * s1;
  double acc2 = Escale[idx2] * s2;
  double acc3 = Escale[idx3] * s3;
  double acc4 = Escale[idx4] * s4;
  __syncthreads();

  sbuf[nd1] = qv1 * v[idx1];
  sbuf[nd2] = qv2 * v[idx2];
  sbuf[nd3] = qv3 * v[idx3];
  sbuf[nd4] = qv4 * v[idx4];
  __syncthreads();

  s1 = s2 = s3 = s4 = 0.0;
  for (int l = 0; l < 16; ++l) {
    const int ib = i + l * 16 + k1 * 256;
    const double dm = sD1D[j + l * 16];
    s1 += dm * sbuf[ib];
    s2 += dm * sbuf[ib + 1024];
    s3 += dm * sbuf[ib + 2048];
    s4 += dm * sbuf[ib + 3072];
  }
  acc1 += Escale[idx1 + npoint] * s1;
  acc2 += Escale[idx2 + npoint] * s2;
  acc3 += Escale[idx3 + npoint] * s3;
  acc4 += Escale[idx4 + npoint] * s4;
  __syncthreads();

  sbuf[nd1] = qv1 * w[idx1];
  sbuf[nd2] = qv2 * w[idx2];
  sbuf[nd3] = qv3 * w[idx3];
  sbuf[nd4] = qv4 * w[idx4];
  __syncthreads();

  s1 = s2 = s3 = s4 = 0.0;
  for (int l = 0; l < 16; ++l) {
    const int ib = i + j * 16 + l * 256;
    const double bv = sbuf[ib];
    s1 += sD1D[k1 + l * 16] * bv;
    s2 += sD1D[k2 + l * 16] * bv;
    s3 += sD1D[k3 + l * 16] * bv;
    s4 += sD1D[k4 + l * 16] * bv;
  }
  acc1 += Escale[idx1 + 2 * npoint] * s1;
  acc2 += Escale[idx2 + 2 * npoint] * s2;
  acc3 += Escale[idx3 + 2 * npoint] * s3;
  acc4 += Escale[idx4 + 2 * npoint] * s4;
  __syncthreads();

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
  sbuf[fp] = 0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  if (tid < 512) {
    fp = tid + 1024;
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
    sbuf[fp] = 0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  }
  __syncthreads();

  const int fa1 = i + k1 * 16;
  const int fa2 = 256 + j + k1 * 16;
  const int fa3 = 512 + i + k1 * 16;
  const int fa4 = 768 + j + k1 * 16;
  const int fa5 = 1024 + i + j * 16;
  const int fa6 = 1280 + i + j * 16;

  dqdt[idx1] = -(acc1 + sLift[j] * sbuf[fa1] + sLift[i + 16] * sbuf[fa2] +
                 sLift[j + 32] * sbuf[fa3] + sLift[i + 48] * sbuf[fa4] +
                 sLift[k1 + 64] * sbuf[fa5] + sLift[k1 + 80] * sbuf[fa6]);
  dqdt[idx2] = -(acc2 + sLift[j] * sbuf[fa1 + 64] + sLift[i + 16] * sbuf[fa2 + 64] +
                 sLift[j + 32] * sbuf[fa3 + 64] + sLift[i + 48] * sbuf[fa4 + 64] +
                 sLift[k2 + 64] * sbuf[fa5] + sLift[k2 + 80] * sbuf[fa6]);
  dqdt[idx3] = -(acc3 + sLift[j] * sbuf[fa1 + 128] + sLift[i + 16] * sbuf[fa2 + 128] +
                 sLift[j + 32] * sbuf[fa3 + 128] + sLift[i + 48] * sbuf[fa4 + 128] +
                 sLift[k3 + 64] * sbuf[fa5] + sLift[k3 + 80] * sbuf[fa6]);
  dqdt[idx4] = -(acc4 + sLift[j] * sbuf[fa1 + 192] + sLift[i + 16] * sbuf[fa2 + 192] +
                 sLift[j + 32] * sbuf[fa3 + 192] + sLift[i + 48] * sbuf[fa4 + 192] +
                 sLift[k4 + 64] * sbuf[fa5] + sLift[k4 + 80] * sbuf[fa6]);
}

__device__ double p31_face_flux_cc(int fidx, const double *q, const double *u,
                                   const double *v, const double *w,
                                   const int *VMapM, const int *VMapP,
                                   const double *normal_fn, const double *Fscale,
                                   int nface)
{
  const int iM = VMapM[fidx] - 1;
  const int iP = VMapP[fidx] - 1;
  const double qM = q[iM];
  const double qP = q[iP];
  const double VelM = u[iM] * normal_fn[fidx] + v[iM] * normal_fn[fidx + nface] +
                      w[iM] * normal_fn[fidx + 2 * nface];
  const double VelP = u[iP] * normal_fn[fidx] + v[iP] * normal_fn[fidx + nface] +
                      w[iP] * normal_fn[fidx + 2 * nface];
  const double alpha = 0.5 * fabs(VelP + VelM);
  return 0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
}

__global__ __launch_bounds__(P31_CC_THREADS, 1) void tendency_fused_p31_xz_cc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ double sD1D[1024];
  __shared__ double sLift[192];
  __shared__ double sQU[1024], sQW[1024];
  __shared__ double sfz5[1024], sfz6[1024];

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i = tid % 32;
  const int k = tid / 32;
  const int elem_offset = elem * 32768;
  const int face_offset = elem * 6144;
  const int npoint = 32768 * Ne;
  const int nface = 6144 * Ne;

  sD1D[tid] = D1D[tid];
  if (tid < 192) {
    sLift[tid] = Lift1D[tid];
  }

  const double fy1 = p31_face_flux_cc(face_offset + i + k * 32, q, u, v, w,
                                      VMapM, VMapP, normal_fn, Fscale, nface);
  const double fy3 = p31_face_flux_cc(face_offset + 2048 + i + k * 32, q, u, v, w,
                                      VMapM, VMapP, normal_fn, Fscale, nface);
  const double fx2 = p31_face_flux_cc(face_offset + 1024 + i + k * 32, q, u, v, w,
                                      VMapM, VMapP, normal_fn, Fscale, nface);
  const double fx4 = p31_face_flux_cc(face_offset + 3072 + i + k * 32, q, u, v, w,
                                      VMapM, VMapP, normal_fn, Fscale, nface);
  sfz5[tid] = p31_face_flux_cc(face_offset + 4096 + i + k * 32, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
  sfz6[tid] = p31_face_flux_cc(face_offset + 5120 + i + k * 32, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
  __syncthreads();

  for (int j = 0; j < 32; ++j) {
    const int idx = elem_offset + i + j * 32 + k * 1024;
    const double qv = q[idx];
    sQU[tid] = qv * u[idx];
    sQW[tid] = qv * w[idx];
    __syncthreads();

    double sx = 0.0;
    double sz = 0.0;
    for (int l = 0; l < 32; ++l) {
      sx += sD1D[i + l * 32] * sQU[l + k * 32];
      sz += sD1D[k + l * 32] * sQW[i + l * 32];
    }

    dqdt[idx] = -(Escale[idx] * sx + Escale[idx + 2 * npoint] * sz +
                  sLift[j] * fy1 + sLift[i + 32] * __shfl_sync(0xffffffff, fx2, j) +
                  sLift[j + 64] * fy3 +
                  sLift[i + 96] * __shfl_sync(0xffffffff, fx4, j) +
                  sLift[k + 128] * sfz5[i + j * 32] +
                  sLift[k + 160] * sfz6[i + j * 32]);
    __syncthreads();
  }
}

__global__ __launch_bounds__(P31_CC_THREADS, 1) void tendency_fused_p31_y_cc_kernel(
    double *dqdt, const double *D1D, const double *q, const double *v,
    const double *Escale, int Ne)
{
  __shared__ double sD1D[1024];
  __shared__ double sQV[1024];

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i = tid % 32;
  const int j = tid / 32;
  const int elem_offset = elem * 32768;
  const int npoint = 32768 * Ne;

  sD1D[tid] = D1D[tid];
  __syncthreads();

  for (int k = 0; k < 32; ++k) {
    const int idx = elem_offset + i + j * 32 + k * 1024;
    sQV[tid] = q[idx] * v[idx];
    __syncthreads();

    double sy = 0.0;
    for (int l = 0; l < 32; ++l) {
      sy += sD1D[j + l * 32] * sQV[i + l * 32];
    }
    dqdt[idx] = dqdt[idx] - Escale[idx + npoint] * sy;
    __syncthreads();
  }
}

__global__ __launch_bounds__(P63_CC_THREADS, 1) void tendency_fused_p63_xz_cc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  __shared__ double sD[64 * 16];
  __shared__ double sFU[16 * 64];
  __shared__ double sFW[64 * 16];

  const int elem = (int)blockIdx.x / P63_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  const int j = (int)blockIdx.x % P63_CC_BPE;
  const int tid = (int)threadIdx.x;
  const int i = tid % 64;
  const int kb = tid / 64;
  const int elem_offset = elem * 262144;
  const int face_offset = elem * 24576;
  const int npoint = 262144 * Ne;

  int kk[4];
  double sx[4], sz[4];
  for (int m = 0; m < 4; ++m) {
    kk[m] = kb + 16 * m;
    sx[m] = 0.0;
    sz[m] = 0.0;
  }

  const int sl = tid % 16;
  const int sk = tid / 16;

  for (int l0 = 0; l0 <= 48; l0 += 16) {
    sD[kb * 64 + i] = D1D[i + (l0 + kb) * 64];
    int gidx = elem_offset + (l0 + sl) + j * 64 + sk * 4096;
    sFU[sk * 16 + sl] = q[gidx] * u[gidx];
    gidx = elem_offset + i + j * 64 + (l0 + kb) * 4096;
    sFW[kb * 64 + i] = q[gidx] * w[gidx];
    __syncthreads();

    for (int lc = 0; lc < 16; ++lc) {
      const double dxi = sD[lc * 64 + i];
      const double fwi = sFW[lc * 64 + i];
      for (int m = 0; m < 4; ++m) {
        sx[m] += dxi * sFU[kk[m] * 16 + lc];
        sz[m] += fwi * sD[lc * 64 + kk[m]];
      }
    }
    __syncthreads();
  }

  const double lf1 = Lift1D[j];
  const double lf2 = Lift1D[i + 64];
  const double lf3 = Lift1D[j + 128];
  const double lf4 = Lift1D[i + 192];
  const double fb5 = flux_bnd[face_offset + 16384 + i + j * 64];
  const double fb6 = flux_bnd[face_offset + 20480 + i + j * 64];

  for (int m = 0; m < 4; ++m) {
    const int k = kk[m];
    const int idx = elem_offset + i + j * 64 + k * 4096;
    dqdt[idx] = -(Escale[idx] * sx[m] + Escale[idx + 2 * npoint] * sz[m] +
                  lf1 * flux_bnd[face_offset + i + k * 64] +
                  lf2 * flux_bnd[face_offset + 4096 + j + k * 64] +
                  lf3 * flux_bnd[face_offset + 8192 + i + k * 64] +
                  lf4 * flux_bnd[face_offset + 12288 + j + k * 64] +
                  Lift1D[k + 256] * fb5 + Lift1D[k + 320] * fb6);
  }
}

__global__ __launch_bounds__(P63_CC_THREADS, 1) void tendency_fused_p63_y_cc_kernel(
    double *dqdt, const double *D1D, const double *q, const double *v,
    const double *Escale, int Ne)
{
  __shared__ double sD[64 * 16];
  __shared__ double sFV[64 * 16];

  const int elem = (int)blockIdx.x / P63_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  const int k = (int)blockIdx.x % P63_CC_BPE;
  const int tid = (int)threadIdx.x;
  const int i = tid % 64;
  const int jb = tid / 64;
  const int elem_offset = elem * 262144;
  const int npoint = 262144 * Ne;

  int jj[4];
  double sy[4];
  for (int m = 0; m < 4; ++m) {
    jj[m] = jb + 16 * m;
    sy[m] = 0.0;
  }

  for (int l0 = 0; l0 <= 48; l0 += 16) {
    sD[jb * 64 + i] = D1D[i + (l0 + jb) * 64];
    const int gidx = elem_offset + i + (l0 + jb) * 64 + k * 4096;
    sFV[jb * 64 + i] = q[gidx] * v[gidx];
    __syncthreads();

    for (int lc = 0; lc < 16; ++lc) {
      const double fvi = sFV[lc * 64 + i];
      for (int m = 0; m < 4; ++m) {
        sy[m] += fvi * sD[lc * 64 + jj[m]];
      }
    }
    __syncthreads();
  }

  for (int m = 0; m < 4; ++m) {
    const int j = jj[m];
    const int idx = elem_offset + i + j * 64 + k * 4096;
    dqdt[idx] = dqdt[idx] - Escale[idx + npoint] * sy[m];
  }
}

__global__ __launch_bounds__(P127_CC_THREADS, 1) void tendency_fused_p127_xz_cc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  __shared__ double sDn[64 * 16];
  __shared__ double sDm[64 * 16];
  __shared__ double sFU[16 * 64];
  __shared__ double sFW[64 * 16];

  const int bidx = (int)blockIdx.x;
  const int ibase = (bidx % 2) * 64;
  const int kbase = ((bidx / 2) % 2) * 64;
  const int j = (bidx / 4) % 128;
  const int elem = bidx / P127_CC_BPE;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i = tid % 64;
  const int kb = tid / 64;
  const int ig = ibase + i;
  const int elem_offset = elem * 2097152;
  const int face_offset = elem * 98304;
  const int npoint = 2097152 * Ne;

  int kk[4];
  double sx[4], sz[4];
  for (int m = 0; m < 4; ++m) {
    kk[m] = kb + 16 * m;
    sx[m] = 0.0;
    sz[m] = 0.0;
  }

  const int sl = tid % 16;
  const int sk = tid / 16;

  for (int l0 = 0; l0 <= 112; l0 += 16) {
    sDn[kb * 64 + i] = D1D[ig + (l0 + kb) * 128];
    sDm[kb * 64 + i] = D1D[kbase + i + (l0 + kb) * 128];
    int gidx = elem_offset + (l0 + sl) + j * 128 + (kbase + sk) * 16384;
    sFU[sk * 16 + sl] = q[gidx] * u[gidx];
    gidx = elem_offset + ig + j * 128 + (l0 + kb) * 16384;
    sFW[kb * 64 + i] = q[gidx] * w[gidx];
    __syncthreads();

    for (int lc = 0; lc < 16; ++lc) {
      const double dxi = sDn[lc * 64 + i];
      const double fwi = sFW[lc * 64 + i];
      for (int m = 0; m < 4; ++m) {
        sx[m] += dxi * sFU[kk[m] * 16 + lc];
        sz[m] += fwi * sDm[lc * 64 + kk[m]];
      }
    }
    __syncthreads();
  }

  const double lf1 = Lift1D[j];
  const double lf2 = Lift1D[ig + 128];
  const double lf3 = Lift1D[j + 256];
  const double lf4 = Lift1D[ig + 384];
  const double fb5 = flux_bnd[face_offset + 65536 + ig + j * 128];
  const double fb6 = flux_bnd[face_offset + 81920 + ig + j * 128];

  for (int m = 0; m < 4; ++m) {
    const int k = kbase + kk[m];
    const int idx = elem_offset + ig + j * 128 + k * 16384;
    dqdt[idx] = -(Escale[idx] * sx[m] + Escale[idx + 2 * npoint] * sz[m] +
                  lf1 * flux_bnd[face_offset + ig + k * 128] +
                  lf2 * flux_bnd[face_offset + 16384 + j + k * 128] +
                  lf3 * flux_bnd[face_offset + 32768 + ig + k * 128] +
                  lf4 * flux_bnd[face_offset + 49152 + j + k * 128] +
                  Lift1D[k + 512] * fb5 + Lift1D[k + 640] * fb6);
  }
}

__global__ __launch_bounds__(P127_CC_THREADS, 1) void tendency_fused_p127_y_cc_kernel(
    double *dqdt, const double *D1D, const double *q, const double *v,
    const double *Escale, int Ne)
{
  __shared__ double sDm[64 * 16];
  __shared__ double sFV[64 * 16];

  const int bidx = (int)blockIdx.x;
  const int ibase = (bidx % 2) * 64;
  const int jbase = ((bidx / 2) % 2) * 64;
  const int k = (bidx / 4) % 128;
  const int elem = bidx / P127_CC_BPE;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i = tid % 64;
  const int jb = tid / 64;
  const int ig = ibase + i;
  const int elem_offset = elem * 2097152;
  const int npoint = 2097152 * Ne;

  int jj[4];
  double sy[4];
  for (int m = 0; m < 4; ++m) {
    jj[m] = jb + 16 * m;
    sy[m] = 0.0;
  }

  for (int l0 = 0; l0 <= 112; l0 += 16) {
    sDm[jb * 64 + i] = D1D[jbase + i + (l0 + jb) * 128];
    const int gidx = elem_offset + ig + (l0 + jb) * 128 + k * 16384;
    sFV[jb * 64 + i] = q[gidx] * v[gidx];
    __syncthreads();

    for (int lc = 0; lc < 16; ++lc) {
      const double fvi = sFV[lc * 64 + i];
      for (int m = 0; m < 4; ++m) {
        sy[m] += fvi * sDm[lc * 64 + jj[m]];
      }
    }
    __syncthreads();
  }

  for (int m = 0; m < 4; ++m) {
    const int j = jbase + jj[m];
    const int idx = elem_offset + ig + j * 128 + k * 16384;
    dqdt[idx] = dqdt[idx] - Escale[idx + npoint] * sy[m];
  }
}

__global__ __launch_bounds__(256, 1) void tendency_x_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ double sD[16 * 16];
  __shared__ double sQ[16 * 16];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int elem = block0 / P255_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  int local_block = block0 % P255_CC_BPE;
  const int k = local_block / 256;
  local_block %= 256;
  const int tile_j = local_block / 16;
  const int tile_i = local_block % 16;
  const int i = tile_i * 16 + tx;
  const int j = tile_j * 16 + ty;
  const int elem_offset = elem * 16777216;

  double sum = 0.0;
  for (int ltile = 0; ltile < 16; ++ltile) {
    int l = ltile * 16 + ty;
    sD[ty * 16 + tx] = D1D[i + l * 256];
    l = ltile * 16 + tx;
    const int idx = elem_offset + l + j * 256 + k * 65536;
    sQ[ty * 16 + tx] = q[idx] * velocity[idx];
    __syncthreads();
    for (int t = 0; t < 16; ++t) {
      sum += sD[t * 16 + tx] * sQ[ty * 16 + t];
    }
    __syncthreads();
  }

  const int idx = elem_offset + i + j * 256 + k * 65536;
  const int fp = j + k * 256;
  const int elem_face_offset = elem * 6 * 65536;
  const double lift_value =
      Lift1D[i + 256] * flux_bnd[elem_face_offset + 65536 + fp] +
      Lift1D[i + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp];
  dqdt[idx] = -(Escale[idx] * sum + lift_value);
}

__global__ __launch_bounds__(256, 1) void tendency_y_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ double sD[16 * 16];
  __shared__ double sQ[16 * 16];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int elem = block0 / P255_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  int local_block = block0 % P255_CC_BPE;
  const int k = local_block / 256;
  local_block %= 256;
  const int tile_j = local_block / 16;
  const int tile_i = local_block % 16;
  const int i = tile_i * 16 + tx;
  const int j = tile_j * 16 + ty;
  const int elem_offset = elem * 16777216;

  double sum = 0.0;
  for (int ltile = 0; ltile < 16; ++ltile) {
    const int l = ltile * 16 + ty;
    const int idx = elem_offset + i + l * 256 + k * 65536;
    sQ[ty * 16 + tx] = q[idx] * velocity[idx];
    sD[ty * 16 + tx] = D1D[tile_j * 16 + tx + l * 256];
    __syncthreads();
    for (int t = 0; t < 16; ++t) {
      sum += sQ[t * 16 + tx] * sD[t * 16 + ty];
    }
    __syncthreads();
  }

  const int idx = elem_offset + i + j * 256 + k * 65536;
  const int fp = i + k * 256;
  const int elem_face_offset = elem * 6 * 65536;
  const double lift_value = Lift1D[j] * flux_bnd[elem_face_offset + fp] +
                            Lift1D[j + 2 * 256] *
                                flux_bnd[elem_face_offset + 2 * 65536 + fp];
  const int npoint = 16777216 * Ne;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * sum + lift_value);
}

__global__ __launch_bounds__(256, 1) void tendency_z_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ double sD[16 * 16];
  __shared__ double sQ[16 * 16];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int elem = block0 / P255_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  const int local_block = block0 % P255_CC_BPE;
  const int tile_k = local_block / 4096;
  const int tile_line = local_block % 4096;
  const int line = tile_line * 16 + tx;
  const int k = tile_k * 16 + ty;
  const int elem_offset = elem * 16777216;

  double sum = 0.0;
  for (int ltile = 0; ltile < 16; ++ltile) {
    const int l = ltile * 16 + ty;
    const int idx = elem_offset + line + l * 65536;
    sQ[ty * 16 + tx] = q[idx] * velocity[idx];
    sD[ty * 16 + tx] = D1D[tile_k * 16 + tx + l * 256];
    __syncthreads();
    for (int t = 0; t < 16; ++t) {
      sum += sQ[t * 16 + tx] * sD[t * 16 + ty];
    }
    __syncthreads();
  }

  const int idx = elem_offset + line + k * 65536;
  const int fp = line;
  const int elem_face_offset = elem * 6 * 65536;
  const double lift_value =
      Lift1D[k + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + fp] +
      Lift1D[k + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + fp];
  const int npoint = 16777216 * Ne;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * sum + lift_value);
}

extern "C" void launch_tendency_fused_p15(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  tendency_fused_p15_cc_kernel<<<Ne, P15_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda_cc_hp("tendency_fused_p15_cc_kernel");
}

extern "C" void launch_tendency_fused_p31(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  tendency_fused_p31_xz_cc_kernel<<<Ne, P31_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  tendency_fused_p31_y_cc_kernel<<<Ne, P31_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p31_cc_kernels");
}

extern "C" void launch_tendency_fused_p63(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = P63_CC_BPE * Ne;
  tendency_fused_p63_xz_cc_kernel<<<nblock, P63_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p63_y_cc_kernel<<<nblock, P63_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p63_cc_kernels");
}

extern "C" void launch_tendency_fused_p127(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = P127_CC_BPE * Ne;
  tendency_fused_p127_xz_cc_kernel<<<nblock, P127_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p127_y_cc_kernel<<<nblock, P127_CC_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p127_cc_kernels");
}

extern "C" void launch_tendency_xyz_p255(
    double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, int Ne)
{
  const dim3 threads(16, 16);
  const int nblock = P255_CC_BPE * Ne;
  tendency_x_p255_cc_kernel<<<nblock, threads, 0, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_y_p255_cc_kernel<<<nblock, threads, 0, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_z_p255_cc_kernel<<<nblock, threads, 0, dg_cuda_stream>>>(
      dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, Ne);
  check_cuda_cc_hp("tendency_xyz_p255_cc_kernels");
}
