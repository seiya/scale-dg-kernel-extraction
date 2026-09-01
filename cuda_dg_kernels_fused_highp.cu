#include <cuda_runtime.h>
#include <cstdio>

#include "fused_kernel_geom.h"

// CUDAFORTRAN_FUSED CUDA-core schedule for p=15..255.
// Geometry matches 2dadc41^ Fortran, not fused_kernel_geom.h TC sizes
// (p=31 is 1024 threads, not P31_THREADS=512).

extern cudaStream_t dg_cuda_stream;

#define P15_CC_THREADS 512
#define P31_CC_XZ_THREADS 512
// Number of j planes the p=31 xz kernel keeps resident and reduces together.
#define P31_CC_XZ_NJ 4
#define P31_CC_XZ_SMEM (8 * (1024 + 192 + 2 * P31_CC_XZ_NJ * 1024 + 2048))
#define P31_CC_Y_THREADS 256
#define P31_CC_Y_MINBLK 4
#ifndef P63_CC_THREADS
#define P63_CC_THREADS 512
#endif
#define P63_CC_BPE 64
#define P63_CC_NK (4096 / P63_CC_THREADS)
#define P63_CC_MINBLK (1024 / P63_CC_THREADS)
#ifndef P127_CC_THREADS
#define P127_CC_THREADS 512
#endif
#define P127_CC_BPE 512
#ifndef P127_CC_BK
#define P127_CC_BK 64
#endif
#define P127_CC_NK (4096 / P127_CC_THREADS)
#define P127_CC_KSTRIDE (64 / P127_CC_NK)
#define P127_CC_MINBLK (1024 / P127_CC_THREADS)
#define P127_CC_STAGE (64 * P127_CC_BK / P127_CC_THREADS)
#define P255_CC_BPE 65536
#ifndef P255_CC_ABLATE
#define P255_CC_ABLATE 0
#endif
#ifndef P127_CC_ABLATE
#define P127_CC_ABLATE 0
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

__device__ double p15_face_flux_cc(int fidx, const double *__restrict__ q,
                                   const double *__restrict__ u,
                                   const double *__restrict__ v,
                                   const double *__restrict__ w,
                                   const int *__restrict__ VMapM,
                                   const int *__restrict__ VMapP,
                                   const double *__restrict__ normal_fn,
                                   const double *__restrict__ Fscale,
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

// p=15 CC: 512 threads, 2 blocks/SM, each thread owning a 2i x 2j x 2k
// output tile. Natural-order shared panels and length-16 inner products;
// the tile shares each direction's per-lane-distinct shared operand over
// more outputs (p15_gap_study.md sec 27). Face gathers issue before the
// x-panel barrier so they overlap the remaining panel stores.
__global__ __launch_bounds__(P15_CC_THREADS, 2) void tendency_fused_p15_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  __shared__ double sD1D[256];
  __shared__ double sLift[96];
  __shared__ __align__(16) double sbuf[4096];
  __shared__ double sFace[1536];

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i0 = 2 * (tid & 7);
  const int j0 = 2 * ((tid >> 3) & 7);
  const int kb = tid >> 6;
  const int elem_offset = elem * 4096;
  const int face_offset = elem * 1536;
  const int npoint = 4096 * Ne;
  const int nface = 1536 * Ne;

  if (tid < 256) {
    sD1D[tid] = D1D[tid];
  } else if (tid < 352) {
    sLift[tid - 256] = Lift1D[tid - 256];
  }

  // p[jj][kk] -> point (i0..i0+1, j0+jj, kb+8*kk)
  int nd[2][2], idx[2][2];
  double2 qv[2][2];
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const int k = kb + 8 * kk;
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      nd[jj][kk] = i0 + (j0 + jj) * 16 + k * 256;
      idx[jj][kk] = elem_offset + nd[jj][kk];
      qv[jj][kk] = *reinterpret_cast<const double2 *>(q + idx[jj][kk]);
      const double2 uu =
          *reinterpret_cast<const double2 *>(u + idx[jj][kk]);
      *reinterpret_cast<double2 *>(sbuf + nd[jj][kk]) =
          make_double2(qv[jj][kk].x * uu.x, qv[jj][kk].y * uu.y);
    }
  }
  sFace[tid] = p15_face_flux_cc(face_offset + tid, q, u, v, w, VMapM, VMapP,
                                normal_fn, Fscale, nface);
  sFace[tid + 512] = p15_face_flux_cc(face_offset + tid + 512, q, u, v, w, VMapM,
                                      VMapP, normal_fn, Fscale, nface);
  sFace[tid + 1024] = p15_face_flux_cc(face_offset + tid + 1024, q, u, v, w,
                                       VMapM, VMapP, normal_fn, Fscale, nface);
  __syncthreads();

  double2 a[2][2];
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      a[jj][kk] = make_double2(0.0, 0.0);
    }
  }
  // x: the D1D pair is per-lane distinct and feeds all 8 outputs.
  {
    double2 s[2][2];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) s[jj][kk] = make_double2(0.0, 0.0);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      const double2 dm = *reinterpret_cast<const double2 *>(sD1D + i0 + l * 16);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const int k = kb + 8 * kk;
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const double b = sbuf[l + (j0 + jj) * 16 + k * 256];
          s[jj][kk].x += dm.x * b;
          s[jj][kk].y += dm.y * b;
        }
      }
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const double2 e =
            *reinterpret_cast<const double2 *>(Escale + idx[jj][kk]);
        a[jj][kk].x += e.x * s[jj][kk].x;
        a[jj][kk].y += e.y * s[jj][kk].y;
      }
  }
  __syncthreads();

#pragma unroll
  for (int kk = 0; kk < 2; ++kk)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const double2 vv = *reinterpret_cast<const double2 *>(v + idx[jj][kk]);
      *reinterpret_cast<double2 *>(sbuf + nd[jj][kk]) =
          make_double2(qv[jj][kk].x * vv.x, qv[jj][kk].y * vv.y);
    }
  __syncthreads();

  // y: the panel pair is per-lane distinct and feeds both j outputs.
  {
    double2 s[2][2];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) s[jj][kk] = make_double2(0.0, 0.0);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      const double2 dj = *reinterpret_cast<const double2 *>(sD1D + j0 + l * 16);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const double2 fv = *reinterpret_cast<const double2 *>(
            sbuf + i0 + l * 16 + (kb + 8 * kk) * 256);
        s[0][kk].x += dj.x * fv.x;
        s[0][kk].y += dj.x * fv.y;
        s[1][kk].x += dj.y * fv.x;
        s[1][kk].y += dj.y * fv.y;
      }
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const double2 e = *reinterpret_cast<const double2 *>(
            Escale + idx[jj][kk] + npoint);
        a[jj][kk].x += e.x * s[jj][kk].x;
        a[jj][kk].y += e.y * s[jj][kk].y;
      }
  }
  __syncthreads();

#pragma unroll
  for (int kk = 0; kk < 2; ++kk)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const double2 ww = *reinterpret_cast<const double2 *>(w + idx[jj][kk]);
      *reinterpret_cast<double2 *>(sbuf + nd[jj][kk]) =
          make_double2(qv[jj][kk].x * ww.x, qv[jj][kk].y * ww.y);
    }
  __syncthreads();

  // z: the panel pair feeds both k outputs.
  {
    double2 s[2][2];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) s[jj][kk] = make_double2(0.0, 0.0);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      const double d0 = sD1D[kb + l * 16];
      const double d1 = sD1D[kb + 8 + l * 16];
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const double2 bv = *reinterpret_cast<const double2 *>(
            sbuf + i0 + (j0 + jj) * 16 + l * 256);
        s[jj][0].x += d0 * bv.x;
        s[jj][0].y += d0 * bv.y;
        s[jj][1].x += d1 * bv.x;
        s[jj][1].y += d1 * bv.y;
      }
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const double2 e = *reinterpret_cast<const double2 *>(
            Escale + idx[jj][kk] + 2 * npoint);
        a[jj][kk].x += e.x * s[jj][kk].x;
        a[jj][kk].y += e.y * s[jj][kk].y;
      }
  }

  const double2 l2 = *reinterpret_cast<const double2 *>(sLift + i0 + 16);
  const double2 l4 = *reinterpret_cast<const double2 *>(sLift + i0 + 48);
#pragma unroll
  for (int kk = 0; kk < 2; ++kk) {
    const int k = kb + 8 * kk;
    const double l5 = sLift[k + 64];
    const double l6 = sLift[k + 80];
    const double2 f1 = *reinterpret_cast<const double2 *>(sFace + i0 + k * 16);
    const double2 f3 =
        *reinterpret_cast<const double2 *>(sFace + 512 + i0 + k * 16);
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int j = j0 + jj;
      const double lf1 = sLift[j];
      const double lf3 = sLift[j + 32];
      const double f2 = sFace[256 + j + k * 16];
      const double f4 = sFace[768 + j + k * 16];
      const double2 f5 =
          *reinterpret_cast<const double2 *>(sFace + 1024 + i0 + j * 16);
      const double2 f6 =
          *reinterpret_cast<const double2 *>(sFace + 1280 + i0 + j * 16);
      *reinterpret_cast<double2 *>(dqdt + idx[jj][kk]) = make_double2(
          -(a[jj][kk].x + lf1 * f1.x + l2.x * f2 + lf3 * f3.x + l4.x * f4 +
            l5 * f5.x + l6 * f6.x),
          -(a[jj][kk].y + lf1 * f1.y + l2.y * f2 + lf3 * f3.y + l4.y * f4 +
            l5 * f5.y + l6 * f6.y));
    }
  }
}

__device__ double p31_face_flux_cc(int fidx, const double *__restrict__ q,
                                   const double *__restrict__ u,
                                   const double *__restrict__ v,
                                   const double *__restrict__ w,
                                   const int *__restrict__ VMapM,
                                   const int *__restrict__ VMapP,
                                   const double *__restrict__ normal_fn,
                                   const double *__restrict__ Fscale,
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

// p=31 CC xz: P31_CC_XZ_NJ j planes resident at once, single buffered, so
// the two per-lane-distinct D1D loads of the inner product feed
// 2 x P31_CC_XZ_NJ outputs. 512 threads at 128 registers is the ceiling;
// widening to NJ = 8 spills (p31_gap_study.md sec 27).
__global__ __launch_bounds__(P31_CC_XZ_THREADS, 1) void tendency_fused_p31_xz_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  extern __shared__ __align__(16) double smem[];
  double *sD1D = smem;
  double *sLift = sD1D + 1024;
  double *sQU = sLift + 192;
  double *sQW = sQU + P31_CC_XZ_NJ * 1024;
  double *sfz5 = sQW + P31_CC_XZ_NJ * 1024;
  double *sfz6 = sfz5 + 1024;

  const int elem = (int)blockIdx.x >> 1;
  if (elem >= Ne) {
    return;
  }
  const int j0 = ((int)blockIdx.x & 1) * 16;

  const int tid = (int)threadIdx.x;
  const int i0 = 2 * (tid & 15);
  const int k = tid >> 4;
  const int ik = i0 + k * 32;
  const int elem_offset = elem * 32768;
  const int face_offset = elem * 6144;
  const int npoint = 32768 * Ne;
  const int nface = 6144 * Ne;

  *reinterpret_cast<double2 *>(sD1D + 2 * tid) =
      *reinterpret_cast<const double2 *>(D1D + 2 * tid);
  if (tid < 192) {
    sLift[tid] = Lift1D[tid];
  }

  const double fy1a = p31_face_flux_cc(face_offset + ik, q, u, v, w, VMapM,
                                        VMapP, normal_fn, Fscale, nface);
  const double fy1b = p31_face_flux_cc(face_offset + ik + 1, q, u, v, w, VMapM,
                                        VMapP, normal_fn, Fscale, nface);
  const double fy3a = p31_face_flux_cc(face_offset + 2048 + ik, q, u, v, w,
                                        VMapM, VMapP, normal_fn, Fscale, nface);
  const double fy3b = p31_face_flux_cc(face_offset + 2048 + ik + 1, q, u, v, w,
                                        VMapM, VMapP, normal_fn, Fscale, nface);
  const double fx2a = p31_face_flux_cc(face_offset + 1024 + ik, q, u, v, w,
                                        VMapM, VMapP, normal_fn, Fscale, nface);
  const double fx2b = p31_face_flux_cc(face_offset + 1024 + ik + 1, q, u, v, w,
                                        VMapM, VMapP, normal_fn, Fscale, nface);
  const double fx4a = p31_face_flux_cc(face_offset + 3072 + ik, q, u, v, w,
                                        VMapM, VMapP, normal_fn, Fscale, nface);
  const double fx4b = p31_face_flux_cc(face_offset + 3072 + ik + 1, q, u, v, w,
                                        VMapM, VMapP, normal_fn, Fscale, nface);
  *reinterpret_cast<double2 *>(sfz5 + ik) = make_double2(
      p31_face_flux_cc(face_offset + 4096 + ik, q, u, v, w, VMapM, VMapP,
                       normal_fn, Fscale, nface),
      p31_face_flux_cc(face_offset + 4096 + ik + 1, q, u, v, w, VMapM, VMapP,
                       normal_fn, Fscale, nface));
  *reinterpret_cast<double2 *>(sfz6 + ik) = make_double2(
      p31_face_flux_cc(face_offset + 5120 + ik, q, u, v, w, VMapM, VMapP,
                       normal_fn, Fscale, nface),
      p31_face_flux_cc(face_offset + 5120 + ik + 1, q, u, v, w, VMapM, VMapP,
                       normal_fn, Fscale, nface));
  __syncthreads();

  const double lf2a = sLift[i0 + 32];
  const double lf2b = sLift[i0 + 33];
  const double lf4a = sLift[i0 + 96];
  const double lf4b = sLift[i0 + 97];
  const double lf5 = sLift[k + 128];
  const double lf6 = sLift[k + 160];

  int idx = elem_offset + i0 + j0 * 32 + k * 1024;

  for (int jp = 0; jp < 16 / P31_CC_XZ_NJ; ++jp) {
    const int j = j0 + P31_CC_XZ_NJ * jp;
    if (jp) {
      __syncthreads();
    }
#pragma unroll
    for (int jj = 0; jj < P31_CC_XZ_NJ; ++jj) {
      const double2 qv =
          *reinterpret_cast<const double2 *>(q + idx + 32 * jj);
      const double2 uv =
          *reinterpret_cast<const double2 *>(u + idx + 32 * jj);
      const double2 wv =
          *reinterpret_cast<const double2 *>(w + idx + 32 * jj);
      *reinterpret_cast<double2 *>(sQU + 1024 * jj + ik) =
          make_double2(qv.x * uv.x, qv.y * uv.y);
      *reinterpret_cast<double2 *>(sQW + 1024 * jj + ik) =
          make_double2(qv.x * wv.x, qv.y * wv.y);
    }
    __syncthreads();

    double sx[P31_CC_XZ_NJ][2], sz[P31_CC_XZ_NJ][2];
#pragma unroll
    for (int jj = 0; jj < P31_CC_XZ_NJ; ++jj) {
      sx[jj][0] = 0.0;
      sx[jj][1] = 0.0;
      sz[jj][0] = 0.0;
      sz[jj][1] = 0.0;
    }
    for (int l = 0; l < 32; ++l) {
      const double2 dxi = *reinterpret_cast<const double2 *>(sD1D + i0 + l * 32);
      const double dk = sD1D[k + l * 32];
#pragma unroll
      for (int jj = 0; jj < P31_CC_XZ_NJ; ++jj) {
        const double qu = sQU[1024 * jj + l + k * 32];
        sx[jj][0] += dxi.x * qu;
        sx[jj][1] += dxi.y * qu;
        const double2 qw =
            *reinterpret_cast<const double2 *>(sQW + 1024 * jj + i0 + l * 32);
        sz[jj][0] += dk * qw.x;
        sz[jj][1] += dk * qw.y;
      }
    }

#pragma unroll
    for (int jj = 0; jj < P31_CC_XZ_NJ; ++jj) {
      const int jc = j + jj;
      const int src = (tid & 16) + (jc >> 1);
      const double fx2 = (jc & 1) ? __shfl_sync(0xffffffff, fx2b, src)
                                  : __shfl_sync(0xffffffff, fx2a, src);
      const double fx4 = (jc & 1) ? __shfl_sync(0xffffffff, fx4b, src)
                                  : __shfl_sync(0xffffffff, fx4a, src);
      const double2 ex =
          *reinterpret_cast<const double2 *>(Escale + idx + 32 * jj);
      const double2 ez = *reinterpret_cast<const double2 *>(
          Escale + idx + 32 * jj + 2 * npoint);
      const double lf1 = sLift[jc];
      const double lf3 = sLift[jc + 64];
      *reinterpret_cast<double2 *>(dqdt + idx + 32 * jj) = make_double2(
          -(ex.x * sx[jj][0] + ez.x * sz[jj][0] + lf1 * fy1a + lf2a * fx2 +
            lf3 * fy3a + lf4a * fx4 + lf5 * sfz5[i0 + jc * 32] +
            lf6 * sfz6[i0 + jc * 32]),
          -(ex.y * sx[jj][1] + ez.y * sz[jj][1] + lf1 * fy1b + lf2b * fx2 +
            lf3 * fy3b + lf4b * fx4 + lf5 * sfz5[i0 + 1 + jc * 32] +
            lf6 * sfz6[i0 + 1 + jc * 32]));
    }
    idx += 32 * P31_CC_XZ_NJ;
  }
}
// p=31 CC y: 256 threads covering two k planes, each thread owning a
// 2i x 4j tile, 4 blocks/SM. The per-lane-distinct panel load feeds eight
// outputs and the four D1D rows are two 16 B broadcasts
// (p31_gap_study.md sec 27).
__global__ __launch_bounds__(P31_CC_Y_THREADS, P31_CC_Y_MINBLK) void tendency_fused_p31_y_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  __shared__ __align__(16) double sD1D[1024];
  __shared__ __align__(16) double sQV[2 * 1024];

  const int elem = (int)blockIdx.x >> 1;
  if (elem >= Ne) {
    return;
  }
  const int k0 = ((int)blockIdx.x & 1) * 16;

  const int tid = (int)threadIdx.x;
  const int i0 = 2 * (tid & 15);
  const int j0 = 4 * ((tid >> 4) & 7);
  const int kp = tid >> 7;
  const int ij = i0 + j0 * 32;
  const int pbuf = kp * 1024;
  const int elem_offset = elem * 32768;
  const int npoint = 32768 * Ne;

  *reinterpret_cast<double2 *>(sD1D + 2 * tid) =
      *reinterpret_cast<const double2 *>(D1D + 2 * tid);
  *reinterpret_cast<double2 *>(sD1D + 512 + 2 * tid) =
      *reinterpret_cast<const double2 *>(D1D + 512 + 2 * tid);

  for (int kl = 0; kl < 16; kl += 2) {
    const int idx = elem_offset + ij + (k0 + kl + kp) * 1024;
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const double2 qa =
          *reinterpret_cast<const double2 *>(q + idx + 32 * jj);
      const double2 va =
          *reinterpret_cast<const double2 *>(v + idx + 32 * jj);
      *reinterpret_cast<double2 *>(sQV + pbuf + ij + 32 * jj) =
          make_double2(qa.x * va.x, qa.y * va.y);
    }
    __syncthreads();

    double s[8];
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      s[m] = 0.0;
    }
    for (int l = 0; l < 32; ++l) {
      const double2 d01 =
          *reinterpret_cast<const double2 *>(sD1D + j0 + l * 32);
      const double2 d23 =
          *reinterpret_cast<const double2 *>(sD1D + j0 + 2 + l * 32);
      const double2 fv =
          *reinterpret_cast<const double2 *>(sQV + pbuf + i0 + l * 32);
      s[0] += d01.x * fv.x;
      s[1] += d01.x * fv.y;
      s[2] += d01.y * fv.x;
      s[3] += d01.y * fv.y;
      s[4] += d23.x * fv.x;
      s[5] += d23.x * fv.y;
      s[6] += d23.y * fv.x;
      s[7] += d23.y * fv.y;
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const double2 ey = *reinterpret_cast<const double2 *>(
          Escale + idx + 32 * jj + npoint);
      double2 out = *reinterpret_cast<double2 *>(dqdt + idx + 32 * jj);
      out.x -= ey.x * s[2 * jj];
      out.y -= ey.y * s[2 * jj + 1];
      *reinterpret_cast<double2 *>(dqdt + idx + 32 * jj) = out;
    }
  }
}
// p=63 CC: one 64x64 plane per block. 512 threads x 8 consecutive k
// (or j) fit 2 blocks/SM at 64 registers; 1024 threads were occupancy-1
// and left LDS latency exposed. Natural-order panels (i fastest); not
// the TC fragment layout.
__global__ __launch_bounds__(P63_CC_THREADS, P63_CC_MINBLK) void tendency_fused_p63_xz_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  extern __shared__ __align__(32) double smem_xz[];
  double *sD = smem_xz;
  double *sFU = smem_xz + 64 * 64;
  double *sFW = smem_xz + 2 * 64 * 64;

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

  int kk[P63_CC_NK];
  double sx[P63_CC_NK], sz[P63_CC_NK];
  const int kbase = P63_CC_NK * kb;
  for (int m = 0; m < P63_CC_NK; ++m) {
    kk[m] = kbase + m;
    sx[m] = 0.0;
    sz[m] = 0.0;
  }

  for (int p = 0; p < P63_CC_NK; ++p) {
    const int idxp = tid + P63_CC_THREADS * p;
    const int col = idxp % 64;
    const int row = idxp / 64;
    sD[row * 64 + col] = D1D[col + row * 64];
    const int gidx = elem_offset + col + j * 64 + row * 4096;
    const double qv = q[gidx];
    sFU[row * 64 + col] = qv * u[gidx];
    sFW[row * 64 + col] = qv * w[gidx];
  }
  __syncthreads();

  for (int lc = 0; lc < 64; ++lc) {
    const double dxi = sD[lc * 64 + i];
    const double fwi = sFW[lc * 64 + i];
#pragma unroll
    for (int m = 0; m < P63_CC_NK; ++m) {
      sx[m] += dxi * sFU[kk[m] * 64 + lc];
    }
#pragma unroll
    for (int m = 0; m < P63_CC_NK; m += 2) {
      const double2 dz =
          *reinterpret_cast<const double2 *>(&sD[lc * 64 + kbase + m]);
      sz[m] += fwi * dz.x;
      sz[m + 1] += fwi * dz.y;
    }
  }

  const double lf1 = Lift1D[j];
  const double lf2 = Lift1D[i + 64];
  const double lf3 = Lift1D[j + 128];
  const double lf4 = Lift1D[i + 192];
  const double fb5 = flux_bnd[face_offset + 16384 + i + j * 64];
  const double fb6 = flux_bnd[face_offset + 20480 + i + j * 64];

  for (int m = 0; m < P63_CC_NK; ++m) {
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

__global__ __launch_bounds__(P63_CC_THREADS, P63_CC_MINBLK) void tendency_fused_p63_y_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  extern __shared__ __align__(32) double smem_y[];
  double *sD = smem_y;
  double *sFV = smem_y + 64 * 64;

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

  int jj[P63_CC_NK];
  double sy[P63_CC_NK];
  const int jbase = P63_CC_NK * jb;
  for (int m = 0; m < P63_CC_NK; ++m) {
    jj[m] = jbase + m;
    sy[m] = 0.0;
  }

  for (int p = 0; p < P63_CC_NK; ++p) {
    const int idxp = tid + P63_CC_THREADS * p;
    const int col = idxp % 64;
    const int row = idxp / 64;
    sD[row * 64 + col] = D1D[col + row * 64];
    const int gidx = elem_offset + col + row * 64 + k * 4096;
    sFV[row * 64 + col] = q[gidx] * v[gidx];
  }
  __syncthreads();

  for (int lc = 0; lc < 64; ++lc) {
    const double fvi = sFV[lc * 64 + i];
#pragma unroll
    for (int m = 0; m < P63_CC_NK; m += 2) {
      const double2 dy =
          *reinterpret_cast<const double2 *>(&sD[lc * 64 + jbase + m]);
      sy[m] += fvi * dy.x;
      sy[m + 1] += fvi * dy.y;
    }
  }

  for (int m = 0; m < P63_CC_NK; ++m) {
    const int j = jj[m];
    const int idx = elem_offset + i + j * 64 + k * 4096;
    dqdt[idx] = dqdt[idx] - Escale[idx + npoint] * sy[m];
  }
}

__global__ __launch_bounds__(P127_CC_THREADS, 1) void tendency_fused_p127_xz_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  extern __shared__ __align__(32) double smem_xz[];
  double *sDn = smem_xz;
  double *sDm = smem_xz + 64 * P127_CC_BK;
  double *sFU = smem_xz + 2 * 64 * P127_CC_BK;
  double *sFW = smem_xz + 3 * 64 * P127_CC_BK;

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

  int kk[P127_CC_NK];
  double sx[P127_CC_NK], sz[P127_CC_NK];
  const int kpack = P127_CC_NK * kb;
  for (int m = 0; m < P127_CC_NK; ++m) {
    kk[m] = kpack + m;
    sx[m] = 0.0;
    sz[m] = 0.0;
  }

  for (int l0 = 0; l0 < 128; l0 += P127_CC_BK) {
    if (l0) {
      __syncthreads();
    }
    for (int p = 0; p < P127_CC_STAGE; ++p) {
      const int idxp = tid + P127_CC_THREADS * p;
      const int col = idxp % 64;
      const int row = idxp / 64;
      sDn[row * 64 + col] = D1D[(ibase + col) + (l0 + row) * 128];
      sDm[row * 64 + col] = D1D[(kbase + col) + (l0 + row) * 128];
      int gidx = elem_offset + (l0 + col) + j * 128 + (kbase + row) * 16384;
      sFU[row * 64 + col] = q[gidx] * u[gidx];
      gidx = elem_offset + (ibase + col) + j * 128 + (l0 + row) * 16384;
      sFW[row * 64 + col] = q[gidx] * w[gidx];
    }
    __syncthreads();

    for (int lc = 0; lc < P127_CC_BK; ++lc) {
#if P127_CC_ABLATE
      const double dxi = 1.0;
      const double fwi = 1.0;
      for (int m = 0; m < P127_CC_NK; ++m) {
        sx[m] += dxi * 1.0;
        sz[m] += fwi * 1.0;
      }
#else
      const double dxi = sDn[lc * 64 + i];
      const double fwi = sFW[lc * 64 + i];
      for (int m = 0; m < P127_CC_NK; ++m) {
        sx[m] += dxi * sFU[kk[m] * P127_CC_BK + lc];
      }
#pragma unroll
      for (int m = 0; m < P127_CC_NK; m += 2) {
        const double2 dz =
            *reinterpret_cast<const double2 *>(&sDm[lc * 64 + kpack + m]);
        sz[m] += fwi * dz.x;
        sz[m + 1] += fwi * dz.y;
      }
#endif
    }
  }

  const double lf1 = Lift1D[j];
  const double lf2 = Lift1D[ig + 128];
  const double lf3 = Lift1D[j + 256];
  const double lf4 = Lift1D[ig + 384];
  const double fb5 = flux_bnd[face_offset + 65536 + ig + j * 128];
  const double fb6 = flux_bnd[face_offset + 81920 + ig + j * 128];

  for (int m = 0; m < P127_CC_NK; ++m) {
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

__global__ __launch_bounds__(P127_CC_THREADS, P127_CC_MINBLK) void tendency_fused_p127_y_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  extern __shared__ __align__(32) double smem_y[];
  double *sDm = smem_y;
  double *sFV = smem_y + 64 * P127_CC_BK;

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

  int jj[P127_CC_NK];
  double sy[P127_CC_NK];
  const int jpack = P127_CC_NK * jb;
  for (int m = 0; m < P127_CC_NK; ++m) {
    jj[m] = jpack + m;
    sy[m] = 0.0;
  }

  for (int l0 = 0; l0 < 128; l0 += P127_CC_BK) {
    if (l0) {
      __syncthreads();
    }
    for (int p = 0; p < P127_CC_STAGE; ++p) {
      const int idxp = tid + P127_CC_THREADS * p;
      const int col = idxp % 64;
      const int row = idxp / 64;
      sDm[row * 64 + col] = D1D[(jbase + col) + (l0 + row) * 128];
      const int gidx = elem_offset + (ibase + col) + (l0 + row) * 128 + k * 16384;
      sFV[row * 64 + col] = q[gidx] * v[gidx];
    }
    __syncthreads();

    for (int lc = 0; lc < P127_CC_BK; ++lc) {
#if P127_CC_ABLATE
      const double fvi = 1.0;
      for (int m = 0; m < P127_CC_NK; ++m) {
        sy[m] += fvi * 1.0;
      }
#else
      const double fvi = sFV[lc * 64 + i];
#pragma unroll
      for (int m = 0; m < P127_CC_NK; m += 2) {
        const double2 dy =
            *reinterpret_cast<const double2 *>(&sDm[lc * 64 + jpack + m]);
        sy[m] += fvi * dy.x;
        sy[m + 1] += fvi * dy.y;
      }
#endif
    }
  }

  for (int m = 0; m < P127_CC_NK; ++m) {
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
  tendency_fused_p15_cc_kernel<<<Ne, P15_CC_THREADS, 0, dg_cuda_stream>>>(
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
  static int xz_smem_set = 0;
  if (!xz_smem_set) {
    cudaFuncSetAttribute(tendency_fused_p31_xz_cc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         P31_CC_XZ_SMEM);
    xz_smem_set = 1;
  }
  tendency_fused_p31_xz_cc_kernel<<<2 * Ne, P31_CC_XZ_THREADS, P31_CC_XZ_SMEM,
                                   dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  tendency_fused_p31_y_cc_kernel<<<2 * Ne, P31_CC_Y_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p31_cc_kernels");
}

extern "C" void launch_tendency_fused_p63(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = P63_CC_BPE * Ne;
  const size_t smem_xz = 3 * 64 * 64 * sizeof(double);
  const size_t smem_y = 2 * 64 * 64 * sizeof(double);
  cudaFuncSetAttribute(tendency_fused_p63_xz_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_xz);
  cudaFuncSetAttribute(tendency_fused_p63_y_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_y);
  tendency_fused_p63_xz_cc_kernel<<<nblock, P63_CC_THREADS, smem_xz, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p63_y_cc_kernel<<<nblock, P63_CC_THREADS, smem_y, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p63_cc_kernels");
}

extern "C" void launch_tendency_fused_p127(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = P127_CC_BPE * Ne;
  const size_t smem_xz = 4 * 64 * P127_CC_BK * sizeof(double);
  const size_t smem_y = 2 * 64 * P127_CC_BK * sizeof(double);
  cudaFuncSetAttribute(tendency_fused_p127_xz_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_xz);
  cudaFuncSetAttribute(tendency_fused_p127_y_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_y);
  tendency_fused_p127_xz_cc_kernel<<<nblock, P127_CC_THREADS, smem_xz, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p127_y_cc_kernel<<<nblock, P127_CC_THREADS, smem_y, dg_cuda_stream>>>(
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
