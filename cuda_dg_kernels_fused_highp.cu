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
#ifndef P255_CC_ABLATE
#define P255_CC_ABLATE 0
#endif
// 1=INNER1, 2=all global 1.0, 3=no epilogue, 4=no barrier,
// 5=D operand 1.0 (Q real), 6=Q/vel 1.0 (D real).

static void check_cuda_cc_hp(const char *what)
{
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
  }
}

// Fortran 2dadc41^ had no launch_bounds; nvcc needs a cap or a 1024-thread
// block will not launch.  minBlocks=1 matches the previous C++ port.
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
#pragma unroll
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
#pragma unroll
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
#pragma unroll
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

// One thread writes two outputs so the K-reduction reuses one D/Q operand
// (ncu 66860: L1/TEX 98%; inner 16→1 is 5058→2188 µs).  128 threads/block.
__global__ __launch_bounds__(128, 8) void tendency_x_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ double sQ0[33 * 16];
  __shared__ double sQ1[33 * 16];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int nblock_pe = P255_CC_BPE / 4;
  const int elem = block0 / nblock_pe;
  if (elem >= Ne) {
    return;
  }
  int local_block = block0 % nblock_pe;
  const int k = local_block / 64;
  local_block %= 64;
  const int quad_j = local_block / 8;
  const int pair_i = local_block % 8;
  const int i0 = pair_i * 32 + tx;
  const int i1 = i0 + 16;
  const int j0 = quad_j * 32 + ty;
  const int j1 = j0 + 8;
  const int j2 = j0 + 16;
  const int j3 = j0 + 24;
  const int elem_offset = elem * 16777216;

  double s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
  for (int ltile = 0; ltile < 8; ++ltile) {
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 6
    sQ0[ty * 33 + tx] = 1.0;
    sQ0[ty * 33 + tx + 16] = 1.0;
    sQ0[(ty + 8) * 33 + tx] = 1.0;
    sQ0[(ty + 8) * 33 + tx + 16] = 1.0;
    sQ1[ty * 33 + tx] = 1.0;
    sQ1[ty * 33 + tx + 16] = 1.0;
    sQ1[(ty + 8) * 33 + tx] = 1.0;
    sQ1[(ty + 8) * 33 + tx + 16] = 1.0;
#else
    const int l = ltile * 32 + tx;
    int gidx = elem_offset + l + j0 * 256 + k * 65536;
    sQ0[ty * 33 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + (l + 16) + j0 * 256 + k * 65536;
    sQ0[ty * 33 + tx + 16] = q[gidx] * velocity[gidx];
    gidx = elem_offset + l + j1 * 256 + k * 65536;
    sQ0[(ty + 8) * 33 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + (l + 16) + j1 * 256 + k * 65536;
    sQ0[(ty + 8) * 33 + tx + 16] = q[gidx] * velocity[gidx];
    gidx = elem_offset + l + j2 * 256 + k * 65536;
    sQ1[ty * 33 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + (l + 16) + j2 * 256 + k * 65536;
    sQ1[ty * 33 + tx + 16] = q[gidx] * velocity[gidx];
    gidx = elem_offset + l + j3 * 256 + k * 65536;
    sQ1[(ty + 8) * 33 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + (l + 16) + j3 * 256 + k * 65536;
    sQ1[(ty + 8) * 33 + tx + 16] = q[gidx] * velocity[gidx];
#endif
#if P255_CC_ABLATE != 4
    __syncthreads();
#endif
#pragma unroll
#if P255_CC_ABLATE == 1
    for (int t = 0; t < 1; ++t)
#else
    for (int t = 0; t < 32; ++t)
#endif
    {
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
      const double d0 = 1.0;
      const double d1 = 1.0;
#else
      const double d0 = __ldg(D1D + i0 + (ltile * 32 + t) * 256);
      const double d1 = __ldg(D1D + i1 + (ltile * 32 + t) * 256);
#endif
      const double f0 = sQ0[ty * 33 + t];
      const double f1 = sQ0[(ty + 8) * 33 + t];
      const double f2 = sQ1[ty * 33 + t];
      const double f3 = sQ1[(ty + 8) * 33 + t];
      s00 += d0 * f0;
      s01 += d0 * f1;
      s02 += d0 * f2;
      s03 += d0 * f3;
      s10 += d1 * f0;
      s11 += d1 * f1;
      s12 += d1 * f2;
      s13 += d1 * f3;
    }
#if P255_CC_ABLATE != 4
    if (ltile + 1 < 8) {
      __syncthreads();
    }
#endif
  }

  const int fp0 = j0 + k * 256;
  const int fp1 = j1 + k * 256;
  const int fp2 = j2 + k * 256;
  const int fp3 = j3 + k * 256;
  const int elem_face_offset = elem * 6 * 65536;
#if P255_CC_ABLATE == 3
  int idx = elem_offset + i0 + j0 * 256 + k * 65536;
  dqdt[idx] = -s00;
  idx = elem_offset + i0 + j1 * 256 + k * 65536;
  dqdt[idx] = -s01;
  idx = elem_offset + i0 + j2 * 256 + k * 65536;
  dqdt[idx] = -s02;
  idx = elem_offset + i0 + j3 * 256 + k * 65536;
  dqdt[idx] = -s03;
  idx = elem_offset + i1 + j0 * 256 + k * 65536;
  dqdt[idx] = -s10;
  idx = elem_offset + i1 + j1 * 256 + k * 65536;
  dqdt[idx] = -s11;
  idx = elem_offset + i1 + j2 * 256 + k * 65536;
  dqdt[idx] = -s12;
  idx = elem_offset + i1 + j3 * 256 + k * 65536;
  dqdt[idx] = -s13;
#else
  const double lift00 =
      Lift1D[i0 + 256] * flux_bnd[elem_face_offset + 65536 + fp0] +
      Lift1D[i0 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp0];
  const double lift01 =
      Lift1D[i0 + 256] * flux_bnd[elem_face_offset + 65536 + fp1] +
      Lift1D[i0 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp1];
  const double lift02 =
      Lift1D[i0 + 256] * flux_bnd[elem_face_offset + 65536 + fp2] +
      Lift1D[i0 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp2];
  const double lift03 =
      Lift1D[i0 + 256] * flux_bnd[elem_face_offset + 65536 + fp3] +
      Lift1D[i0 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp3];
  const double lift10 =
      Lift1D[i1 + 256] * flux_bnd[elem_face_offset + 65536 + fp0] +
      Lift1D[i1 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp0];
  const double lift11 =
      Lift1D[i1 + 256] * flux_bnd[elem_face_offset + 65536 + fp1] +
      Lift1D[i1 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp1];
  const double lift12 =
      Lift1D[i1 + 256] * flux_bnd[elem_face_offset + 65536 + fp2] +
      Lift1D[i1 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp2];
  const double lift13 =
      Lift1D[i1 + 256] * flux_bnd[elem_face_offset + 65536 + fp3] +
      Lift1D[i1 + 3 * 256] * flux_bnd[elem_face_offset + 3 * 65536 + fp3];
  int idx = elem_offset + i0 + j0 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s00 + lift00);
  idx = elem_offset + i0 + j1 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s01 + lift01);
  idx = elem_offset + i0 + j2 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s02 + lift02);
  idx = elem_offset + i0 + j3 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s03 + lift03);
  idx = elem_offset + i1 + j0 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s10 + lift10);
  idx = elem_offset + i1 + j1 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s11 + lift11);
  idx = elem_offset + i1 + j2 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s12 + lift12);
  idx = elem_offset + i1 + j3 * 256 + k * 65536;
  dqdt[idx] = -(__ldg(Escale + idx) * s13 + lift13);
#endif
}

__global__ __launch_bounds__(128, 8) void tendency_y_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ double sD[2][16 * 32];
  __shared__ double sQ0[2][16 * 16];
  __shared__ double sQ1[2][16 * 16];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int nblock_pe = P255_CC_BPE / 4;
  const int elem = block0 / nblock_pe;
  if (elem >= Ne) {
    return;
  }
  int local_block = block0 % nblock_pe;
  const int k = local_block / 64;
  local_block %= 64;
  const int quad_j = local_block / 8;
  const int pair_i = local_block % 8;
  const int i0 = pair_i * 32 + tx;
  const int i1 = i0 + 16;
  const int j0 = quad_j * 32 + ty;
  const int j1 = j0 + 8;
  const int j2 = j0 + 16;
  const int j3 = j0 + 24;
  const int jbase = quad_j * 32;
  const int elem_offset = elem * 16777216;

  double s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
  int p = 0;
  int ft = 0;
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
    sD[p][ty * 32 + tx] = 1.0;
    sD[p][ty * 32 + tx + 16] = 1.0;
    sD[p][(ty + 8) * 32 + tx] = 1.0;
    sD[p][(ty + 8) * 32 + tx + 16] = 1.0;
#else
    sD[p][ty * 32 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + tx + (ft * 16 + ty) * 256];
    sD[p][ty * 32 + 16 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + 16 + tx + (ft * 16 + ty) * 256];
    sD[p][(ty + 8) * 32 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + tx + (ft * 16 + ty + 8) * 256];
    sD[p][(ty + 8) * 32 + 16 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + 16 + tx + (ft * 16 + ty + 8) * 256];
#endif
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 6
    sQ0[p][ty * 16 + tx] = 1.0;
    sQ0[p][(ty + 8) * 16 + tx] = 1.0;
    sQ1[p][ty * 16 + tx] = 1.0;
    sQ1[p][(ty + 8) * 16 + tx] = 1.0;
#else
    const int l0 = ft * 16 + ty;
    int gidx = elem_offset + i0 + l0 * 256 + k * 65536;
    sQ0[p][ty * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + i0 + (l0 + 8) * 256 + k * 65536;
    sQ0[p][(ty + 8) * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + i1 + l0 * 256 + k * 65536;
    sQ1[p][ty * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + i1 + (l0 + 8) * 256 + k * 65536;
    sQ1[p][(ty + 8) * 16 + tx] = q[gidx] * velocity[gidx];
#endif
#if P255_CC_ABLATE != 4
    __syncthreads();
#endif
  for (int ltile = 0; ltile < 16; ++ltile) {
#pragma unroll
#if P255_CC_ABLATE == 1
    for (int t = 0; t < 1; ++t)
#else
    for (int t = 0; t < 16; ++t)
#endif
    {
      const double2 dj0 =
          *reinterpret_cast<const double2 *>(&sD[p][t * 32 + 2 * ty]);
      const double2 dj1 =
          *reinterpret_cast<const double2 *>(&sD[p][t * 32 + 16 + 2 * ty]);
      const double f0 = sQ0[p][t * 16 + tx];
      const double f1 = sQ1[p][t * 16 + tx];
      s00 += f0 * dj0.x;
      s01 += f0 * dj0.y;
      s02 += f0 * dj1.x;
      s03 += f0 * dj1.y;
      s10 += f1 * dj0.x;
      s11 += f1 * dj0.y;
      s12 += f1 * dj1.x;
      s13 += f1 * dj1.y;
    }
    if (ltile + 1 < 16) {
      p ^= 1;
      ft = ltile + 1;
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
    sD[p][ty * 32 + tx] = 1.0;
    sD[p][ty * 32 + tx + 16] = 1.0;
    sD[p][(ty + 8) * 32 + tx] = 1.0;
    sD[p][(ty + 8) * 32 + tx + 16] = 1.0;
#else
    sD[p][ty * 32 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + tx + (ft * 16 + ty) * 256];
    sD[p][ty * 32 + 16 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + 16 + tx + (ft * 16 + ty) * 256];
    sD[p][(ty + 8) * 32 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + tx + (ft * 16 + ty + 8) * 256];
    sD[p][(ty + 8) * 32 + 16 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[jbase + 16 + tx + (ft * 16 + ty + 8) * 256];
#endif
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 6
    sQ0[p][ty * 16 + tx] = 1.0;
    sQ0[p][(ty + 8) * 16 + tx] = 1.0;
    sQ1[p][ty * 16 + tx] = 1.0;
    sQ1[p][(ty + 8) * 16 + tx] = 1.0;
#else
    const int l0 = ft * 16 + ty;
    int gidx = elem_offset + i0 + l0 * 256 + k * 65536;
    sQ0[p][ty * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + i0 + (l0 + 8) * 256 + k * 65536;
    sQ0[p][(ty + 8) * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + i1 + l0 * 256 + k * 65536;
    sQ1[p][ty * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + i1 + (l0 + 8) * 256 + k * 65536;
    sQ1[p][(ty + 8) * 16 + tx] = q[gidx] * velocity[gidx];
#endif
#if P255_CC_ABLATE != 4
      __syncthreads();
#endif
    }
  }

  const int npoint = 16777216 * Ne;
  const int fp0 = i0 + k * 256;
  const int fp1 = i1 + k * 256;
  const int elem_face_offset = elem * 6 * 65536;
#if P255_CC_ABLATE == 3
  int idx = elem_offset + i0 + j0 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s00;
  idx = elem_offset + i0 + j1 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s01;
  idx = elem_offset + i0 + j2 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s02;
  idx = elem_offset + i0 + j3 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s03;
  idx = elem_offset + i1 + j0 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s10;
  idx = elem_offset + i1 + j1 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s11;
  idx = elem_offset + i1 + j2 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s12;
  idx = elem_offset + i1 + j3 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - s13;
#else
  const double lift00 = Lift1D[j0] * flux_bnd[elem_face_offset + fp0] +
                        Lift1D[j0 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp0];
  const double lift01 = Lift1D[j1] * flux_bnd[elem_face_offset + fp0] +
                        Lift1D[j1 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp0];
  const double lift02 = Lift1D[j2] * flux_bnd[elem_face_offset + fp0] +
                        Lift1D[j2 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp0];
  const double lift03 = Lift1D[j3] * flux_bnd[elem_face_offset + fp0] +
                        Lift1D[j3 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp0];
  const double lift10 = Lift1D[j0] * flux_bnd[elem_face_offset + fp1] +
                        Lift1D[j0 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp1];
  const double lift11 = Lift1D[j1] * flux_bnd[elem_face_offset + fp1] +
                        Lift1D[j1 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp1];
  const double lift12 = Lift1D[j2] * flux_bnd[elem_face_offset + fp1] +
                        Lift1D[j2 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp1];
  const double lift13 = Lift1D[j3] * flux_bnd[elem_face_offset + fp1] +
                        Lift1D[j3 + 2 * 256] *
                            flux_bnd[elem_face_offset + 2 * 65536 + fp1];
  int idx = elem_offset + i0 + j0 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s00 + lift00);
  idx = elem_offset + i0 + j1 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s01 + lift01);
  idx = elem_offset + i0 + j2 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s02 + lift02);
  idx = elem_offset + i0 + j3 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s03 + lift03);
  idx = elem_offset + i1 + j0 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s10 + lift10);
  idx = elem_offset + i1 + j1 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s11 + lift11);
  idx = elem_offset + i1 + j2 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s12 + lift12);
  idx = elem_offset + i1 + j3 * 256 + k * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * s13 + lift13);
#endif
}

__global__ __launch_bounds__(128, 8) void tendency_z_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ double sD[16 * 32];
  __shared__ double sQ0[16 * 16];
  __shared__ double sQ1[16 * 16];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int nblock_pe = P255_CC_BPE / 4;
  const int elem = block0 / nblock_pe;
  if (elem >= Ne) {
    return;
  }
  const int local_block = block0 % nblock_pe;
  const int quad_k = local_block / 2048;
  const int pair = local_block % 2048;
  const int line0 = pair * 32 + tx;
  const int line1 = line0 + 16;
  const int k0 = quad_k * 32 + ty;
  const int k1 = k0 + 8;
  const int k2 = k0 + 16;
  const int k3 = k0 + 24;
  const int kbase = quad_k * 32;
  const int elem_offset = elem * 16777216;

  double s00 = 0.0, s01 = 0.0, s02 = 0.0, s03 = 0.0;
  double s10 = 0.0, s11 = 0.0, s12 = 0.0, s13 = 0.0;
  for (int ltile = 0; ltile < 16; ++ltile) {
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
    sD[ty * 32 + tx] = 1.0;
    sD[ty * 32 + tx + 16] = 1.0;
    sD[(ty + 8) * 32 + tx] = 1.0;
    sD[(ty + 8) * 32 + tx + 16] = 1.0;
#else
    sD[ty * 32 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[kbase + tx + (ltile * 16 + ty) * 256];
    sD[ty * 32 + 16 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[kbase + 16 + tx + (ltile * 16 + ty) * 256];
    sD[(ty + 8) * 32 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[kbase + tx + (ltile * 16 + ty + 8) * 256];
    sD[(ty + 8) * 32 + 16 + 2 * (tx & 7) + (tx >> 3)] =
        D1D[kbase + 16 + tx + (ltile * 16 + ty + 8) * 256];
#endif
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 6
    sQ0[ty * 16 + tx] = 1.0;
    sQ0[(ty + 8) * 16 + tx] = 1.0;
    sQ1[ty * 16 + tx] = 1.0;
    sQ1[(ty + 8) * 16 + tx] = 1.0;
#else
    const int l0 = ltile * 16 + ty;
    int gidx = elem_offset + line0 + l0 * 65536;
    sQ0[ty * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + line0 + (l0 + 8) * 65536;
    sQ0[(ty + 8) * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + line1 + l0 * 65536;
    sQ1[ty * 16 + tx] = q[gidx] * velocity[gidx];
    gidx = elem_offset + line1 + (l0 + 8) * 65536;
    sQ1[(ty + 8) * 16 + tx] = q[gidx] * velocity[gidx];
#endif
#if P255_CC_ABLATE != 4
    __syncthreads();
#endif
#pragma unroll
#if P255_CC_ABLATE == 1
    for (int t = 0; t < 1; ++t)
#else
    for (int t = 0; t < 16; ++t)
#endif
    {
      const double2 dk0 =
          *reinterpret_cast<const double2 *>(&sD[t * 32 + 2 * ty]);
      const double2 dk1 =
          *reinterpret_cast<const double2 *>(&sD[t * 32 + 16 + 2 * ty]);
      const double f0 = sQ0[t * 16 + tx];
      const double f1 = sQ1[t * 16 + tx];
      s00 += f0 * dk0.x;
      s01 += f0 * dk0.y;
      s02 += f0 * dk1.x;
      s03 += f0 * dk1.y;
      s10 += f1 * dk0.x;
      s11 += f1 * dk0.y;
      s12 += f1 * dk1.x;
      s13 += f1 * dk1.y;
    }
#if P255_CC_ABLATE != 4
    if (ltile + 1 < 16) {
      __syncthreads();
    }
#endif
  }

  const int npoint = 16777216 * Ne;
  const int elem_face_offset = elem * 6 * 65536;
#if P255_CC_ABLATE == 3
  int idx = elem_offset + line0 + k0 * 65536;
  dqdt[idx] = dqdt[idx] - s00;
  idx = elem_offset + line0 + k1 * 65536;
  dqdt[idx] = dqdt[idx] - s01;
  idx = elem_offset + line0 + k2 * 65536;
  dqdt[idx] = dqdt[idx] - s02;
  idx = elem_offset + line0 + k3 * 65536;
  dqdt[idx] = dqdt[idx] - s03;
  idx = elem_offset + line1 + k0 * 65536;
  dqdt[idx] = dqdt[idx] - s10;
  idx = elem_offset + line1 + k1 * 65536;
  dqdt[idx] = dqdt[idx] - s11;
  idx = elem_offset + line1 + k2 * 65536;
  dqdt[idx] = dqdt[idx] - s12;
  idx = elem_offset + line1 + k3 * 65536;
  dqdt[idx] = dqdt[idx] - s13;
#else
  const double lift00 =
      Lift1D[k0 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line0] +
      Lift1D[k0 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line0];
  const double lift01 =
      Lift1D[k1 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line0] +
      Lift1D[k1 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line0];
  const double lift02 =
      Lift1D[k2 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line0] +
      Lift1D[k2 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line0];
  const double lift03 =
      Lift1D[k3 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line0] +
      Lift1D[k3 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line0];
  const double lift10 =
      Lift1D[k0 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line1] +
      Lift1D[k0 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line1];
  const double lift11 =
      Lift1D[k1 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line1] +
      Lift1D[k1 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line1];
  const double lift12 =
      Lift1D[k2 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line1] +
      Lift1D[k2 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line1];
  const double lift13 =
      Lift1D[k3 + 4 * 256] * flux_bnd[elem_face_offset + 4 * 65536 + line1] +
      Lift1D[k3 + 5 * 256] * flux_bnd[elem_face_offset + 5 * 65536 + line1];
  int idx = elem_offset + line0 + k0 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s00 + lift00);
  idx = elem_offset + line0 + k1 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s01 + lift01);
  idx = elem_offset + line0 + k2 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s02 + lift02);
  idx = elem_offset + line0 + k3 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s03 + lift03);
  idx = elem_offset + line1 + k0 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s10 + lift10);
  idx = elem_offset + line1 + k1 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s11 + lift11);
  idx = elem_offset + line1 + k2 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s12 + lift12);
  idx = elem_offset + line1 + k3 * 65536;
  dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * s13 + lift13);
#endif
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
  const dim3 threads_x(16, 8);
  const dim3 threads_y(16, 8);
  const dim3 threads_z(16, 8);
  const int nblock_x = (P255_CC_BPE / 4) * Ne;
  const int nblock_y = (P255_CC_BPE / 4) * Ne;
  const int nblock_z = (P255_CC_BPE / 4) * Ne;
  tendency_x_p255_cc_kernel<<<nblock_x, threads_x, 0, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_y_p255_cc_kernel<<<nblock_y, threads_y, 0, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_z_p255_cc_kernel<<<nblock_z, threads_z, 0, dg_cuda_stream>>>(
      dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, Ne);
  check_cuda_cc_hp("tendency_xyz_p255_cc_kernels");
}
