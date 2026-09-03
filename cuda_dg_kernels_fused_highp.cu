#include <cuda_runtime.h>
#include <cstdio>

#include "fused_kernel_geom.h"

// CUDAFORTRAN_FUSED CUDA-core schedule for p=15..255.
// Geometry matches 2dadc41^ Fortran, not fused_kernel_geom.h TC sizes
// (p=31 is 1024 threads, not P31_THREADS=512).

extern cudaStream_t dg_cuda_stream;

// p=15 CC output-tile knob: TKK consecutive-in-stride k outputs per thread
// (the tile is 2i x 2j x TKK).  The block still covers the whole 16^3
// element, so the thread count follows: 1024 / TKK.  TKK = 2 is the sec 27
// shape; shared loads per FMA are (2 + 2*TKK)/(4*TKK) for x and y and
// (TKK + 4)/(4*TKK) for z, i.e. 0.75 everywhere at TKK = 2 and
// 0.625 / 0.625 / 0.50 at TKK = 4.
#ifndef P15_CC_TKK
#define P15_CC_TKK 8
#endif
#define P15_CC_THREADS (1024 / P15_CC_TKK)
#define P15_CC_NKB (P15_CC_THREADS / 64)
#ifndef P15_CC_MINB
#define P15_CC_MINB 1
#endif
// p=31 xz output-tile knob (dfma_register_budget.md sec 12 axis carried over
// to the CC path).  TK is how many consecutive k outputs one thread owns; the
// block still covers the same 32 i x 32 k slab, so the thread count is the
// dependent variable.  TK = 1 is the sec 27 shape.  The shared loads per FMA
// of the inner product are (2 + NJ*TK)/(2*NJ*TK) for x and (TK + 2*NJ)/
// (2*NJ*TK) for z, i.e. 0.9375 at TK = 1 and 0.625 at TK = 2.
#ifndef P31_CC_XZ_TK
#define P31_CC_XZ_TK 4
#endif
#ifndef P31_CC_XZ_MINB
#define P31_CC_XZ_MINB 1
#endif
#define P31_CC_XZ_THREADS (512 / P31_CC_XZ_TK)
// Number of j planes the p=31 xz kernel keeps resident and reduces together.
#ifndef P31_CC_XZ_NJ
#define P31_CC_XZ_NJ 2
#endif
#define P31_CC_XZ_SMEM (8 * (1024 + 192 + 2 * P31_CC_XZ_NJ * 1024 + 2048))
// p=31 y output-tile knob: TJ consecutive j outputs per thread.  The block
// still covers 32 i x 32 j x 2 k, so the thread count follows.  TJ = 4 is the
// sec 27 shape (shared loads per FMA (2 + TJ)/(2*TJ) = 0.75); TJ = 8 gives
// 0.625 at half the threads.
#ifndef P31_CC_Y_TJ
#define P31_CC_Y_TJ 4
#endif
#define P31_CC_Y_THREADS (1024 / P31_CC_Y_TJ)
#ifndef P31_CC_Y_MINBLK
#define P31_CC_Y_MINBLK 4
#endif
#ifndef P63_CC_THREADS
#define P63_CC_THREADS 512
#endif
#define P63_CC_BPE 64
// p=63 CC output-tile knobs (p63_gap_study.md sec 26).  TI is the number of
// consecutive i outputs one thread owns; the k (or j) count per thread then
// follows from the 64x64 plane and the thread count.  TI = 1 with 512 threads
// is the sec 25 shape.  MINBLK is the register budget: 65536 / (THREADS *
// MINBLK) registers per thread.
#ifndef P63_CC_XZ_TI
#define P63_CC_XZ_TI 2
#endif
#ifndef P63_CC_XZ_THREADS
#define P63_CC_XZ_THREADS 256
#endif
#ifndef P63_CC_XZ_MINBLK
#define P63_CC_XZ_MINBLK 2
#endif
#ifndef P63_CC_XZ_BK
#define P63_CC_XZ_BK 64
#endif
#define P63_CC_XZ_STAGE (64 * P63_CC_XZ_BK / P63_CC_XZ_THREADS)
#define P63_CC_XZ_NI (64 / P63_CC_XZ_TI)
#define P63_CC_XZ_NK (4096 / (P63_CC_XZ_THREADS * P63_CC_XZ_TI))
#ifndef P63_CC_Y_TI
#define P63_CC_Y_TI 2
#endif
#ifndef P63_CC_Y_THREADS
#define P63_CC_Y_THREADS 256
#endif
#ifndef P63_CC_Y_MINBLK
#define P63_CC_Y_MINBLK 2
#endif
#ifndef P63_CC_Y_BK
#define P63_CC_Y_BK 64
#endif
#define P63_CC_Y_STAGE (64 * P63_CC_Y_BK / P63_CC_Y_THREADS)
#define P63_CC_Y_NI (64 / P63_CC_Y_TI)
#define P63_CC_Y_NJ (4096 / (P63_CC_Y_THREADS * P63_CC_Y_TI))
#ifndef P127_CC_THREADS
#define P127_CC_THREADS 512
#endif
#define P127_CC_BPE 512
#ifndef P127_CC_BK
#define P127_CC_BK 64
#endif
// p=127 CC output-tile knobs (p127_gap_study.md sec 18).  Same two axes as
// the p=63 knobs above: TI consecutive i outputs per thread, and MINBLK as
// the register budget.  BK is the l-chunk of sec 17.3 item 2, now separate
// for xz and y so a wider output tile can buy back the blocks/SM that the
// shared panel costs.
#ifndef P127_CC_XZ_TI
#define P127_CC_XZ_TI 2
#endif
#ifndef P127_CC_XZ_THREADS
#define P127_CC_XZ_THREADS 256
#endif
#ifndef P127_CC_XZ_MINBLK
#define P127_CC_XZ_MINBLK 2
#endif
#ifndef P127_CC_XZ_BK
#define P127_CC_XZ_BK 16
#endif
#define P127_CC_XZ_NI (64 / P127_CC_XZ_TI)
#define P127_CC_XZ_NK (4096 / (P127_CC_XZ_THREADS * P127_CC_XZ_TI))
#define P127_CC_XZ_STAGE (64 * P127_CC_XZ_BK / P127_CC_XZ_THREADS)
#ifndef P127_CC_Y_TI
#define P127_CC_Y_TI 2
#endif
#ifndef P127_CC_Y_THREADS
#define P127_CC_Y_THREADS 128
#endif
#ifndef P127_CC_Y_MINBLK
#define P127_CC_Y_MINBLK 2
#endif
#ifndef P127_CC_Y_BK
#define P127_CC_Y_BK 32
#endif
#define P127_CC_Y_NI (64 / P127_CC_Y_TI)
#define P127_CC_Y_NJ (4096 / (P127_CC_Y_THREADS * P127_CC_Y_TI))
#define P127_CC_Y_STAGE (64 * P127_CC_Y_BK / P127_CC_Y_THREADS)
#define P255_CC_BPE 65536
// p=255 CC output tile / register budget knobs.  Section 26.6 of
// p255_gap_study.md sweeps them; the defaults below are what it adopted
// (1874.3 -> 1525.0 us/stage, -18.63%, bit identical).  P255_CC_{X,Y}_TI and
// P255_CC_Z_TL are how many i (line, for z) a thread owns -- the j/k count is
// four in all three kernels -- and the MINB knobs are the second
// __launch_bounds__ argument, i.e. the register budget the tile needs.  The
// old form was TI = 2 with minBlocks = 8 (64 registers, 50% occupancy), which
// is section 15.68-15.76; this is the next rung of that same ladder, and it
// stops here (TI = 16 does not fit y's static shared and loses 5-22% in x and
// z).  P255_CC_ZDG is the rejected form of section 26.5: z reads D1D with
// __ldg instead of staging it, which halves shared wavefronts but multiplies
// global requests by 5.2 because z's D index rides threadIdx.y.
#ifndef P255_CC_ZDG
#define P255_CC_ZDG 0
#endif
#ifndef P255_CC_Z_TL
#define P255_CC_Z_TL 8
#endif
#ifndef P255_CC_Z_MINB
#define P255_CC_Z_MINB 4
#endif
// Same ladder for x and y: P255_CC_{X,Y}_TI is how many i a thread owns (the
// j count is 4 in both), P255_CC_{X,Y}_MINB the register budget that has to
// come with it.
#ifndef P255_CC_X_TI
#define P255_CC_X_TI 8
#endif
#ifndef P255_CC_X_MINB
#define P255_CC_X_MINB 3
#endif
#ifndef P255_CC_Y_TI
#define P255_CC_Y_TI 8
#endif
#ifndef P255_CC_Y_MINB
#define P255_CC_Y_MINB 2
#endif
// Section 28 block-shape knobs.  The block is 128 threads shaped
// (BDX, 128/BDX); threadIdx.x indexes the i (line, for z) a thread owns and
// threadIdx.y the four j (k).  (16, 8) was the shape all three kernels used
// through section 27.  Section 28 swept it: x stays at 16 (32 is a wash, 8 is
// +2.2%), but y and z want (8, 16), which shrinks their shared panels and
// their register pressure and buys back occupancy -- worth 5.60% of the stage
// together with z's MINB 2 -> 4.  P255_CC_{Y,Z}_LT is the l-tile depth of the
// shared panels; it must be a multiple of 128/BDX, and y has to halve it at
// BDX = 32 to stay inside 48 KB of static shared (which is why BDX = 32 is
// not a one-axis change for y).
#ifndef P255_CC_X_BDX
#define P255_CC_X_BDX 16
#endif
#ifndef P255_CC_Y_BDX
#define P255_CC_Y_BDX 8
#endif
#ifndef P255_CC_Z_BDX
#define P255_CC_Z_BDX 8
#endif
#ifndef P255_CC_Y_LT
#define P255_CC_Y_LT 16
#endif
#ifndef P255_CC_Z_LT
#define P255_CC_Z_LT 16
#endif
// Where the j offset m of a shared D panel is stored, so that the four j a
// thread owns (m = ty + BDY*a) come back as two aligned double2 loads at
// 2*ty and 2*BDY + 2*ty.  With BDY = 8 this is the 2*(tx&7) + (tx>>3) pair
// swizzle of section 15.12.
#define P255_JPOS(m, BDY)                                                     \
  (((m) / (2 * (BDY))) * (2 * (BDY)) + 2 * (((m) % (2 * (BDY))) % (BDY)) +    \
   (((m) % (2 * (BDY))) / (BDY)))
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 6
#define P255_QV(G) (1.0)
#else
#define P255_QV(G) (q[G] * velocity[G])
#endif
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
#define P255_D(I) (1.0)
#else
#define P255_D(I) (D1D[I])
#endif
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

// p=15 CC: 1024/P15_CC_TKK threads, each thread owning a 2i x 2j x TKK
// output tile.  TKK = 8 (128 threads, 254 registers) beats the 512-thread
// TKK = 2 shape of sec 27 by 13.1% (dfma_register_budget.md sec 13.5). Natural-order shared panels and length-16 inner products;
// the tile shares each direction's per-lane-distinct shared operand over
// more outputs (p15_gap_study.md sec 27). Face gathers issue before the
// x-panel barrier so they overlap the remaining panel stores.
__global__ __launch_bounds__(P15_CC_THREADS, P15_CC_MINB) void tendency_fused_p15_cc_kernel(
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

  if constexpr (P15_CC_THREADS >= 352) {
    if (tid < 256) {
      sD1D[tid] = D1D[tid];
    } else if (tid < 352) {
      sLift[tid - 256] = Lift1D[tid - 256];
    }
  } else {
    for (int t = tid; t < 256; t += P15_CC_THREADS) {
      sD1D[t] = D1D[t];
    }
    for (int t = tid; t < 96; t += P15_CC_THREADS) {
      sLift[t] = Lift1D[t];
    }
  }

  // p[jj][kk] -> point (i0..i0+1, j0+jj, kb+P15_CC_NKB*kk)
  constexpr int TKK = P15_CC_TKK;
  int nd[2][TKK], idx[2][TKK];
  double2 qv[2][TKK];
#pragma unroll
  for (int kk = 0; kk < TKK; ++kk) {
    const int k = kb + P15_CC_NKB * kk;
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
#pragma unroll
  for (int fb = 0; fb < 1536 / P15_CC_THREADS; ++fb) {
    const int fp = tid + P15_CC_THREADS * fb;
    sFace[fp] = p15_face_flux_cc(face_offset + fp, q, u, v, w, VMapM, VMapP,
                                 normal_fn, Fscale, nface);
  }
  __syncthreads();

  double2 a[2][TKK];
#pragma unroll
  for (int jj = 0; jj < 2; ++jj) {
#pragma unroll
    for (int kk = 0; kk < TKK; ++kk) {
      a[jj][kk] = make_double2(0.0, 0.0);
    }
  }
  // x: the D1D pair is per-lane distinct and feeds all 8 outputs.
  {
    double2 s[2][TKK];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) s[jj][kk] = make_double2(0.0, 0.0);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      const double2 dm = *reinterpret_cast<const double2 *>(sD1D + i0 + l * 16);
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) {
        const int k = kb + P15_CC_NKB * kk;
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
      for (int kk = 0; kk < TKK; ++kk) {
        const double2 e =
            *reinterpret_cast<const double2 *>(Escale + idx[jj][kk]);
        a[jj][kk].x += e.x * s[jj][kk].x;
        a[jj][kk].y += e.y * s[jj][kk].y;
      }
  }
  __syncthreads();

#pragma unroll
  for (int kk = 0; kk < TKK; ++kk)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const double2 vv = *reinterpret_cast<const double2 *>(v + idx[jj][kk]);
      *reinterpret_cast<double2 *>(sbuf + nd[jj][kk]) =
          make_double2(qv[jj][kk].x * vv.x, qv[jj][kk].y * vv.y);
    }
  __syncthreads();

  // y: the panel pair is per-lane distinct and feeds both j outputs.
  {
    double2 s[2][TKK];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) s[jj][kk] = make_double2(0.0, 0.0);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      const double2 dj = *reinterpret_cast<const double2 *>(sD1D + j0 + l * 16);
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) {
        const double2 fv = *reinterpret_cast<const double2 *>(
            sbuf + i0 + l * 16 + (kb + P15_CC_NKB * kk) * 256);
        s[0][kk].x += dj.x * fv.x;
        s[0][kk].y += dj.x * fv.y;
        s[1][kk].x += dj.y * fv.x;
        s[1][kk].y += dj.y * fv.y;
      }
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) {
        const double2 e = *reinterpret_cast<const double2 *>(
            Escale + idx[jj][kk] + npoint);
        a[jj][kk].x += e.x * s[jj][kk].x;
        a[jj][kk].y += e.y * s[jj][kk].y;
      }
  }
  __syncthreads();

#pragma unroll
  for (int kk = 0; kk < TKK; ++kk)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const double2 ww = *reinterpret_cast<const double2 *>(w + idx[jj][kk]);
      *reinterpret_cast<double2 *>(sbuf + nd[jj][kk]) =
          make_double2(qv[jj][kk].x * ww.x, qv[jj][kk].y * ww.y);
    }
  __syncthreads();

  // z: the panel pair feeds both k outputs.
  {
    double2 s[2][TKK];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) s[jj][kk] = make_double2(0.0, 0.0);
#pragma unroll
    for (int l = 0; l < 16; ++l) {
      double dz[TKK];
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) {
        dz[kk] = sD1D[kb + P15_CC_NKB * kk + l * 16];
      }
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const double2 bv = *reinterpret_cast<const double2 *>(
            sbuf + i0 + (j0 + jj) * 16 + l * 256);
#pragma unroll
        for (int kk = 0; kk < TKK; ++kk) {
          s[jj][kk].x += dz[kk] * bv.x;
          s[jj][kk].y += dz[kk] * bv.y;
        }
      }
    }
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
#pragma unroll
      for (int kk = 0; kk < TKK; ++kk) {
        const double2 e = *reinterpret_cast<const double2 *>(
            Escale + idx[jj][kk] + 2 * npoint);
        a[jj][kk].x += e.x * s[jj][kk].x;
        a[jj][kk].y += e.y * s[jj][kk].y;
      }
  }

  const double2 l2 = *reinterpret_cast<const double2 *>(sLift + i0 + 16);
  const double2 l4 = *reinterpret_cast<const double2 *>(sLift + i0 + 48);
#pragma unroll
  for (int kk = 0; kk < TKK; ++kk) {
    const int k = kb + P15_CC_NKB * kk;
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

// p=31 CC xz: P31_CC_XZ_NJ j planes resident at once, single buffered, and
// P31_CC_XZ_TK consecutive k per thread, so one thread owns 2i x NJ j x TK k
// outputs. 128 threads x 255 registers (TK = 4, NJ = 2) beats the 512-thread
// 2i x 4j x 1k shape of p31_gap_study.md sec 27 by 16.6%: the "512 threads
// cap registers at 128" premise of sec 27.1 was about the thread count, not
// about the kernel (dfma_register_budget.md sec 13).
__global__ __launch_bounds__(P31_CC_XZ_THREADS, P31_CC_XZ_MINB) void tendency_fused_p31_xz_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  constexpr int TK = P31_CC_XZ_TK;
  constexpr int NJ = P31_CC_XZ_NJ;
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
  const int k0 = TK * (tid >> 4);
  const int elem_offset = elem * 32768;
  const int face_offset = elem * 6144;
  const int npoint = 32768 * Ne;
  const int nface = 6144 * Ne;

#pragma unroll
  for (int t = 0; t < TK; ++t) {
    const int tt = tid + P31_CC_XZ_THREADS * t;
    *reinterpret_cast<double2 *>(sD1D + 2 * tt) =
        *reinterpret_cast<const double2 *>(D1D + 2 * tt);
  }
  for (int t = tid; t < 192; t += P31_CC_XZ_THREADS) {
    sLift[t] = Lift1D[t];
  }

  double fy1a[TK], fy1b[TK], fy3a[TK], fy3b[TK];
  double fx2a[TK], fx2b[TK], fx4a[TK], fx4b[TK];
#pragma unroll
  for (int kk = 0; kk < TK; ++kk) {
    const int ik = i0 + (k0 + kk) * 32;
    fy1a[kk] = p31_face_flux_cc(face_offset + ik, q, u, v, w, VMapM,
                                VMapP, normal_fn, Fscale, nface);
    fy1b[kk] = p31_face_flux_cc(face_offset + ik + 1, q, u, v, w, VMapM,
                                VMapP, normal_fn, Fscale, nface);
    fy3a[kk] = p31_face_flux_cc(face_offset + 2048 + ik, q, u, v, w,
                                VMapM, VMapP, normal_fn, Fscale, nface);
    fy3b[kk] = p31_face_flux_cc(face_offset + 2048 + ik + 1, q, u, v, w,
                                VMapM, VMapP, normal_fn, Fscale, nface);
    fx2a[kk] = p31_face_flux_cc(face_offset + 1024 + ik, q, u, v, w,
                                VMapM, VMapP, normal_fn, Fscale, nface);
    fx2b[kk] = p31_face_flux_cc(face_offset + 1024 + ik + 1, q, u, v, w,
                                VMapM, VMapP, normal_fn, Fscale, nface);
    fx4a[kk] = p31_face_flux_cc(face_offset + 3072 + ik, q, u, v, w,
                                VMapM, VMapP, normal_fn, Fscale, nface);
    fx4b[kk] = p31_face_flux_cc(face_offset + 3072 + ik + 1, q, u, v, w,
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
  }
  __syncthreads();

  const double lf2a = sLift[i0 + 32];
  const double lf2b = sLift[i0 + 33];
  const double lf4a = sLift[i0 + 96];
  const double lf4b = sLift[i0 + 97];
  double lf5[TK], lf6[TK];
#pragma unroll
  for (int kk = 0; kk < TK; ++kk) {
    lf5[kk] = sLift[k0 + kk + 128];
    lf6[kk] = sLift[k0 + kk + 160];
  }

  int idx = elem_offset + i0 + j0 * 32 + k0 * 1024;

  for (int jp = 0; jp < 16 / NJ; ++jp) {
    const int j = j0 + NJ * jp;
    if (jp) {
      __syncthreads();
    }
#pragma unroll
    for (int kk = 0; kk < TK; ++kk) {
      const int ik = i0 + (k0 + kk) * 32;
#pragma unroll
      for (int jj = 0; jj < NJ; ++jj) {
        const int g = idx + 1024 * kk + 32 * jj;
        const double2 qv = *reinterpret_cast<const double2 *>(q + g);
        const double2 uv = *reinterpret_cast<const double2 *>(u + g);
        const double2 wv = *reinterpret_cast<const double2 *>(w + g);
        *reinterpret_cast<double2 *>(sQU + 1024 * jj + ik) =
            make_double2(qv.x * uv.x, qv.y * uv.y);
        *reinterpret_cast<double2 *>(sQW + 1024 * jj + ik) =
            make_double2(qv.x * wv.x, qv.y * wv.y);
      }
    }
    __syncthreads();

    double sx[NJ][TK][2], sz[NJ][TK][2];
#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
#pragma unroll
      for (int kk = 0; kk < TK; ++kk) {
        sx[jj][kk][0] = 0.0;
        sx[jj][kk][1] = 0.0;
        sz[jj][kk][0] = 0.0;
        sz[jj][kk][1] = 0.0;
      }
    }
    for (int l = 0; l < 32; ++l) {
      const double2 dxi = *reinterpret_cast<const double2 *>(sD1D + i0 + l * 32);
      double dk[TK];
#pragma unroll
      for (int kk = 0; kk < TK; ++kk) {
        dk[kk] = sD1D[k0 + kk + l * 32];
      }
#pragma unroll
      for (int jj = 0; jj < NJ; ++jj) {
        const double2 qw =
            *reinterpret_cast<const double2 *>(sQW + 1024 * jj + i0 + l * 32);
#pragma unroll
        for (int kk = 0; kk < TK; ++kk) {
          const double qu = sQU[1024 * jj + l + (k0 + kk) * 32];
          sx[jj][kk][0] += dxi.x * qu;
          sx[jj][kk][1] += dxi.y * qu;
          sz[jj][kk][0] += dk[kk] * qw.x;
          sz[jj][kk][1] += dk[kk] * qw.y;
        }
      }
    }

#pragma unroll
    for (int jj = 0; jj < NJ; ++jj) {
      const int jc = j + jj;
      const int src = (tid & 16) + (jc >> 1);
      const double lf1 = sLift[jc];
      const double lf3 = sLift[jc + 64];
      const double f5 = sfz5[i0 + jc * 32];
      const double f5b = sfz5[i0 + 1 + jc * 32];
      const double f6 = sfz6[i0 + jc * 32];
      const double f6b = sfz6[i0 + 1 + jc * 32];
#pragma unroll
      for (int kk = 0; kk < TK; ++kk) {
        const double fx2 = (jc & 1) ? __shfl_sync(0xffffffff, fx2b[kk], src)
                                    : __shfl_sync(0xffffffff, fx2a[kk], src);
        const double fx4 = (jc & 1) ? __shfl_sync(0xffffffff, fx4b[kk], src)
                                    : __shfl_sync(0xffffffff, fx4a[kk], src);
        const int g = idx + 1024 * kk + 32 * jj;
        const double2 ex = *reinterpret_cast<const double2 *>(Escale + g);
        const double2 ez =
            *reinterpret_cast<const double2 *>(Escale + g + 2 * npoint);
        *reinterpret_cast<double2 *>(dqdt + g) = make_double2(
            -(ex.x * sx[jj][kk][0] + ez.x * sz[jj][kk][0] + lf1 * fy1a[kk] +
              lf2a * fx2 + lf3 * fy3a[kk] + lf4a * fx4 + lf5[kk] * f5 +
              lf6[kk] * f6),
            -(ex.y * sx[jj][kk][1] + ez.y * sz[jj][kk][1] + lf1 * fy1b[kk] +
              lf2b * fx2 + lf3 * fy3b[kk] + lf4b * fx4 + lf5[kk] * f5b +
              lf6[kk] * f6b));
      }
    }
    idx += 32 * NJ;
  }
}
// p=31 CC y: 256 threads covering two k planes, each thread owning a
// 2i x P31_CC_Y_TJ tile, 4 blocks/SM. The per-lane-distinct panel load feeds
// 2*TJ outputs and the TJ D1D rows are TJ/2 16 B broadcasts
// (p31_gap_study.md sec 27).  TJ = 8 halves the warps and lowers the loads
// per FMA but is a wash on wall time (dfma_register_budget.md sec 13.8).
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

  constexpr int TJ = P31_CC_Y_TJ;
  const int tid = (int)threadIdx.x;
  const int i0 = 2 * (tid & 15);
  const int j0 = TJ * ((tid >> 4) & (32 / TJ - 1));
  const int kp = tid / (16 * (32 / TJ));
  const int ij = i0 + j0 * 32;
  const int pbuf = kp * 1024;
  const int elem_offset = elem * 32768;
  const int npoint = 32768 * Ne;

#pragma unroll
  for (int t = 0; t < TJ / 2; ++t) {
    const int tt = tid + P31_CC_Y_THREADS * t;
    *reinterpret_cast<double2 *>(sD1D + 2 * tt) =
        *reinterpret_cast<const double2 *>(D1D + 2 * tt);
  }

  for (int kl = 0; kl < 16; kl += 2) {
    const int idx = elem_offset + ij + (k0 + kl + kp) * 1024;
    __syncthreads();
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
      const double2 qa =
          *reinterpret_cast<const double2 *>(q + idx + 32 * jj);
      const double2 va =
          *reinterpret_cast<const double2 *>(v + idx + 32 * jj);
      *reinterpret_cast<double2 *>(sQV + pbuf + ij + 32 * jj) =
          make_double2(qa.x * va.x, qa.y * va.y);
    }
    __syncthreads();

    double s[2 * TJ];
#pragma unroll
    for (int m = 0; m < 2 * TJ; ++m) {
      s[m] = 0.0;
    }
    for (int l = 0; l < 32; ++l) {
      double dj[TJ];
#pragma unroll
      for (int t = 0; t < TJ; t += 2) {
        const double2 dv =
            *reinterpret_cast<const double2 *>(sD1D + j0 + t + l * 32);
        dj[t] = dv.x;
        dj[t + 1] = dv.y;
      }
      const double2 fv =
          *reinterpret_cast<const double2 *>(sQV + pbuf + i0 + l * 32);
#pragma unroll
      for (int jj = 0; jj < TJ; ++jj) {
        s[2 * jj] += dj[jj] * fv.x;
        s[2 * jj + 1] += dj[jj] * fv.y;
      }
    }
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
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
// Load TI consecutive doubles as ceil(TI/2) 16 B loads.  With TI = 2 the
// compiler otherwise keeps two 8 B loads at lane stride 16 B, and each of them
// touches the whole 512 B warp footprint, so the sector count per request
// doubles (p63_gap_study.md sec 53.8).
template <int N>
__device__ __forceinline__ void cc_ldvec(const double *__restrict__ p,
                                         double (&o)[N])
{
#pragma unroll
  for (int t = 0; t < N; t += 2) {
    if constexpr (N == 1) {
      o[0] = p[0];
    } else {
      const double2 v = *reinterpret_cast<const double2 *>(p + t);
      o[t] = v.x;
      o[t + 1] = v.y;
    }
  }
}

__global__ __launch_bounds__(P63_CC_XZ_THREADS, P63_CC_XZ_MINBLK) void tendency_fused_p63_xz_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  constexpr int TI = P63_CC_XZ_TI;
  constexpr int NK = P63_CC_XZ_NK;
  constexpr int BK = P63_CC_XZ_BK;
  extern __shared__ __align__(32) double smem_xz[];
  double *sD = smem_xz;
  double *sFU = smem_xz + 64 * BK;
  double *sFW = smem_xz + 2 * 64 * BK;

  const int elem = (int)blockIdx.x / P63_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  const int j = (int)blockIdx.x % P63_CC_BPE;
  const int tid = (int)threadIdx.x;
  const int i0 = TI * (tid % P63_CC_XZ_NI);
  const int kb = tid / P63_CC_XZ_NI;
  const int elem_offset = elem * 262144;
  const int face_offset = elem * 24576;
  const int npoint = 262144 * Ne;

  const int kbase = NK * kb;
  double sx[NK][TI], sz[NK][TI];
#pragma unroll
  for (int m = 0; m < NK; ++m) {
#pragma unroll
    for (int t = 0; t < TI; ++t) {
      sx[m][t] = 0.0;
      sz[m][t] = 0.0;
    }
  }

  for (int l0 = 0; l0 < 64; l0 += BK) {
    if (l0) {
      __syncthreads();
    }
    for (int p = 0; p < P63_CC_XZ_STAGE; ++p) {
      const int idxp = tid + P63_CC_XZ_THREADS * p;
      const int col = idxp % 64;
      const int row = idxp / 64;
      sD[row * 64 + col] = D1D[col + (l0 + row) * 64];
      if constexpr (BK == 64) {
        // One chunk: the FU and FW staging addresses coincide, so q is read
        // once (sec 26.4 -- writing them as two expressions costs +47 %
        // global sectors).
        const int gidx = elem_offset + col + j * 64 + row * 4096;
        const double qv = q[gidx];
        sFU[row * 64 + col] = qv * u[gidx];
        sFW[row * 64 + col] = qv * w[gidx];
      } else {
        int gidx = elem_offset + col + j * 64 + (l0 + row) * 4096;
        sFW[row * 64 + col] = q[gidx] * w[gidx];
        const int lcol = idxp % BK;
        const int krow = idxp / BK;
        gidx = elem_offset + (l0 + lcol) + j * 64 + krow * 4096;
        sFU[krow * BK + lcol] = q[gidx] * u[gidx];
      }
    }
    __syncthreads();

    for (int lc = 0; lc < BK; ++lc) {
    double dxi[TI], fwi[TI];
#pragma unroll
    for (int t = 0; t < TI; t += 2) {
      if constexpr (TI == 1) {
        dxi[0] = sD[lc * 64 + i0];
        fwi[0] = sFW[lc * 64 + i0];
      } else {
        const double2 dv =
            *reinterpret_cast<const double2 *>(&sD[lc * 64 + i0 + t]);
        const double2 fv =
            *reinterpret_cast<const double2 *>(&sFW[lc * 64 + i0 + t]);
        dxi[t] = dv.x;
        dxi[t + 1] = dv.y;
        fwi[t] = fv.x;
        fwi[t + 1] = fv.y;
      }
    }
#pragma unroll
    for (int m = 0; m < NK; ++m) {
      const double qu = sFU[(kbase + m) * BK + lc];
#pragma unroll
      for (int t = 0; t < TI; ++t) {
        sx[m][t] += dxi[t] * qu;
      }
    }
#pragma unroll
    for (int m = 0; m < NK; m += 2) {
      const double2 dz =
          *reinterpret_cast<const double2 *>(&sD[lc * 64 + kbase + m]);
#pragma unroll
      for (int t = 0; t < TI; ++t) {
        sz[m][t] += fwi[t] * dz.x;
        sz[m + 1][t] += fwi[t] * dz.y;
      }
    }
    }
  }

  const double lf1 = Lift1D[j];
  const double lf3 = Lift1D[j + 128];
  double lf2[TI], lf4[TI], fb5[TI], fb6[TI];
  cc_ldvec(Lift1D + i0 + 64, lf2);
  cc_ldvec(Lift1D + i0 + 192, lf4);
  cc_ldvec(flux_bnd + face_offset + 16384 + i0 + j * 64, fb5);
  cc_ldvec(flux_bnd + face_offset + 20480 + i0 + j * 64, fb6);

  for (int m = 0; m < NK; ++m) {
    const int k = kbase + m;
    const int idx = elem_offset + i0 + j * 64 + k * 4096;
    const double lf5 = Lift1D[k + 256];
    const double lf6 = Lift1D[k + 320];
    const double fbj2 = flux_bnd[face_offset + 4096 + j + k * 64];
    const double fbj4 = flux_bnd[face_offset + 12288 + j + k * 64];
    double ex[TI], ez[TI], fb1[TI], fb3[TI];
    cc_ldvec(Escale + idx, ex);
    cc_ldvec(Escale + idx + 2 * npoint, ez);
    cc_ldvec(flux_bnd + face_offset + i0 + k * 64, fb1);
    cc_ldvec(flux_bnd + face_offset + 8192 + i0 + k * 64, fb3);
    double out[TI];
#pragma unroll
    for (int t = 0; t < TI; ++t) {
      out[t] = -(ex[t] * sx[m][t] + ez[t] * sz[m][t] + lf1 * fb1[t] +
                 lf2[t] * fbj2 + lf3 * fb3[t] + lf4[t] * fbj4 +
                 lf5 * fb5[t] + lf6 * fb6[t]);
    }
#pragma unroll
    for (int t = 0; t < TI; t += 2) {
      if constexpr (TI == 1) {
        dqdt[idx] = out[0];
      } else {
        *reinterpret_cast<double2 *>(dqdt + idx + t) =
            make_double2(out[t], out[t + 1]);
      }
    }
  }
}

__global__ __launch_bounds__(P63_CC_Y_THREADS, P63_CC_Y_MINBLK) void tendency_fused_p63_y_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  constexpr int TI = P63_CC_Y_TI;
  constexpr int NJ = P63_CC_Y_NJ;
  constexpr int BK = P63_CC_Y_BK;
  extern __shared__ __align__(32) double smem_y[];
  double *sD = smem_y;
  double *sFV = smem_y + 64 * BK;

  const int elem = (int)blockIdx.x / P63_CC_BPE;
  if (elem >= Ne) {
    return;
  }
  const int k = (int)blockIdx.x % P63_CC_BPE;
  const int tid = (int)threadIdx.x;
  const int i0 = TI * (tid % P63_CC_Y_NI);
  const int jb = tid / P63_CC_Y_NI;
  const int elem_offset = elem * 262144;
  const int npoint = 262144 * Ne;

  const int jbase = NJ * jb;
  double sy[NJ][TI];
#pragma unroll
  for (int m = 0; m < NJ; ++m) {
#pragma unroll
    for (int t = 0; t < TI; ++t) {
      sy[m][t] = 0.0;
    }
  }

  for (int l0 = 0; l0 < 64; l0 += BK) {
    if (l0) {
      __syncthreads();
    }
    for (int p = 0; p < P63_CC_Y_STAGE; ++p) {
      const int idxp = tid + P63_CC_Y_THREADS * p;
      const int col = idxp % 64;
      const int row = idxp / 64;
      sD[row * 64 + col] = D1D[col + (l0 + row) * 64];
      const int gidx = elem_offset + col + (l0 + row) * 64 + k * 4096;
      sFV[row * 64 + col] = q[gidx] * v[gidx];
    }
    __syncthreads();

    for (int lc = 0; lc < BK; ++lc) {
    double fvi[TI];
#pragma unroll
    for (int t = 0; t < TI; t += 2) {
      if constexpr (TI == 1) {
        fvi[0] = sFV[lc * 64 + i0];
      } else {
        const double2 fv =
            *reinterpret_cast<const double2 *>(&sFV[lc * 64 + i0 + t]);
        fvi[t] = fv.x;
        fvi[t + 1] = fv.y;
      }
    }
#pragma unroll
    for (int m = 0; m < NJ; m += 2) {
      const double2 dy =
          *reinterpret_cast<const double2 *>(&sD[lc * 64 + jbase + m]);
#pragma unroll
      for (int t = 0; t < TI; ++t) {
        sy[m][t] += fvi[t] * dy.x;
        sy[m + 1][t] += fvi[t] * dy.y;
      }
    }
    }
  }

  for (int m = 0; m < NJ; ++m) {
    const int j = jbase + m;
    const int idx = elem_offset + i0 + j * 64 + k * 4096;
#pragma unroll
    for (int t = 0; t < TI; t += 2) {
      if constexpr (TI == 1) {
        dqdt[idx] = dqdt[idx] - Escale[idx + npoint] * sy[m][0];
      } else {
        const double2 ey =
            *reinterpret_cast<const double2 *>(Escale + idx + t + npoint);
        double2 out = *reinterpret_cast<double2 *>(dqdt + idx + t);
        out.x -= ey.x * sy[m][t];
        out.y -= ey.y * sy[m][t + 1];
        *reinterpret_cast<double2 *>(dqdt + idx + t) = out;
      }
    }
  }
}

__global__ __launch_bounds__(P127_CC_XZ_THREADS, P127_CC_XZ_MINBLK) void tendency_fused_p127_xz_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  constexpr int TI = P127_CC_XZ_TI;
  constexpr int NK = P127_CC_XZ_NK;
  constexpr int BK = P127_CC_XZ_BK;
  extern __shared__ __align__(32) double smem_xz[];
  double *sDn = smem_xz;
  double *sDm = smem_xz + 64 * BK;
  double *sFU = smem_xz + 2 * 64 * BK;
  double *sFW = smem_xz + 3 * 64 * BK;

  const int bidx = (int)blockIdx.x;
  const int ibase = (bidx % 2) * 64;
  const int kbase = ((bidx / 2) % 2) * 64;
  const int j = (bidx / 4) % 128;
  const int elem = bidx / P127_CC_BPE;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i0 = TI * (tid % P127_CC_XZ_NI);
  const int kb = tid / P127_CC_XZ_NI;
  const int ig = ibase + i0;
  const int elem_offset = elem * 2097152;
  const int face_offset = elem * 98304;
  const int npoint = 2097152 * Ne;

  const int kpack = NK * kb;
  double sx[NK][TI], sz[NK][TI];
#pragma unroll
  for (int m = 0; m < NK; ++m) {
#pragma unroll
    for (int t = 0; t < TI; ++t) {
      sx[m][t] = 0.0;
      sz[m][t] = 0.0;
    }
  }

  for (int l0 = 0; l0 < 128; l0 += BK) {
    if (l0) {
      __syncthreads();
    }
    for (int p = 0; p < P127_CC_XZ_STAGE; ++p) {
      const int idxp = tid + P127_CC_XZ_THREADS * p;
      const int col = idxp % 64;
      const int row = idxp / 64;
      sDn[row * 64 + col] = D1D[(ibase + col) + (l0 + row) * 128];
      sDm[row * 64 + col] = D1D[(kbase + col) + (l0 + row) * 128];
      int gidx = elem_offset + (ibase + col) + j * 128 + (l0 + row) * 16384;
      sFW[row * 64 + col] = q[gidx] * w[gidx];
      const int lcol = idxp % BK;
      const int krow = idxp / BK;
      gidx = elem_offset + (l0 + lcol) + j * 128 + (kbase + krow) * 16384;
      sFU[krow * BK + lcol] = q[gidx] * u[gidx];
    }
    __syncthreads();

    for (int lc = 0; lc < BK; ++lc) {
#if P127_CC_ABLATE
#pragma unroll
      for (int m = 0; m < NK; ++m) {
#pragma unroll
        for (int t = 0; t < TI; ++t) {
          sx[m][t] += 1.0;
          sz[m][t] += 1.0;
        }
      }
#else
      double dxi[TI], fwi[TI];
#pragma unroll
      for (int t = 0; t < TI; t += 2) {
        if constexpr (TI == 1) {
          dxi[0] = sDn[lc * 64 + i0];
          fwi[0] = sFW[lc * 64 + i0];
        } else {
          const double2 dv =
              *reinterpret_cast<const double2 *>(&sDn[lc * 64 + i0 + t]);
          const double2 fv =
              *reinterpret_cast<const double2 *>(&sFW[lc * 64 + i0 + t]);
          dxi[t] = dv.x;
          dxi[t + 1] = dv.y;
          fwi[t] = fv.x;
          fwi[t + 1] = fv.y;
        }
      }
#pragma unroll
      for (int m = 0; m < NK; ++m) {
        const double qu = sFU[(kpack + m) * BK + lc];
#pragma unroll
        for (int t = 0; t < TI; ++t) {
          sx[m][t] += dxi[t] * qu;
        }
      }
#pragma unroll
      for (int m = 0; m < NK; m += 2) {
        const double2 dz =
            *reinterpret_cast<const double2 *>(&sDm[lc * 64 + kpack + m]);
#pragma unroll
        for (int t = 0; t < TI; ++t) {
          sz[m][t] += fwi[t] * dz.x;
          sz[m + 1][t] += fwi[t] * dz.y;
        }
      }
#endif
    }
  }

  const double lf1 = Lift1D[j];
  const double lf3 = Lift1D[j + 256];
  double lf2[TI], lf4[TI], fb5[TI], fb6[TI];
  cc_ldvec(Lift1D + ig + 128, lf2);
  cc_ldvec(Lift1D + ig + 384, lf4);
  cc_ldvec(flux_bnd + face_offset + 65536 + ig + j * 128, fb5);
  cc_ldvec(flux_bnd + face_offset + 81920 + ig + j * 128, fb6);

  for (int m = 0; m < NK; ++m) {
    const int k = kbase + kpack + m;
    const int idx = elem_offset + ig + j * 128 + k * 16384;
    const double lf5 = Lift1D[k + 512];
    const double lf6 = Lift1D[k + 640];
    const double fbj2 = flux_bnd[face_offset + 16384 + j + k * 128];
    const double fbj4 = flux_bnd[face_offset + 49152 + j + k * 128];
    double ex[TI], ez[TI], fb1[TI], fb3[TI];
    cc_ldvec(Escale + idx, ex);
    cc_ldvec(Escale + idx + 2 * npoint, ez);
    cc_ldvec(flux_bnd + face_offset + ig + k * 128, fb1);
    cc_ldvec(flux_bnd + face_offset + 32768 + ig + k * 128, fb3);
    double out[TI];
#pragma unroll
    for (int t = 0; t < TI; ++t) {
      out[t] = -(ex[t] * sx[m][t] + ez[t] * sz[m][t] + lf1 * fb1[t] +
                 lf2[t] * fbj2 + lf3 * fb3[t] + lf4[t] * fbj4 +
                 lf5 * fb5[t] + lf6 * fb6[t]);
    }
#pragma unroll
    for (int t = 0; t < TI; t += 2) {
      if constexpr (TI == 1) {
        dqdt[idx] = out[0];
      } else {
        *reinterpret_cast<double2 *>(dqdt + idx + t) =
            make_double2(out[t], out[t + 1]);
      }
    }
  }
}

__global__ __launch_bounds__(P127_CC_Y_THREADS, P127_CC_Y_MINBLK) void tendency_fused_p127_y_cc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  constexpr int TI = P127_CC_Y_TI;
  constexpr int NJ = P127_CC_Y_NJ;
  constexpr int BK = P127_CC_Y_BK;
  extern __shared__ __align__(32) double smem_y[];
  double *sDm = smem_y;
  double *sFV = smem_y + 64 * BK;

  const int bidx = (int)blockIdx.x;
  const int ibase = (bidx % 2) * 64;
  const int jbase = ((bidx / 2) % 2) * 64;
  const int k = (bidx / 4) % 128;
  const int elem = bidx / P127_CC_BPE;
  if (elem >= Ne) {
    return;
  }

  const int tid = (int)threadIdx.x;
  const int i0 = TI * (tid % P127_CC_Y_NI);
  const int jb = tid / P127_CC_Y_NI;
  const int ig = ibase + i0;
  const int elem_offset = elem * 2097152;
  const int npoint = 2097152 * Ne;

  const int jpack = NJ * jb;
  double sy[NJ][TI];
#pragma unroll
  for (int m = 0; m < NJ; ++m) {
#pragma unroll
    for (int t = 0; t < TI; ++t) {
      sy[m][t] = 0.0;
    }
  }

  for (int l0 = 0; l0 < 128; l0 += BK) {
    if (l0) {
      __syncthreads();
    }
    for (int p = 0; p < P127_CC_Y_STAGE; ++p) {
      const int idxp = tid + P127_CC_Y_THREADS * p;
      const int col = idxp % 64;
      const int row = idxp / 64;
      sDm[row * 64 + col] = D1D[(jbase + col) + (l0 + row) * 128];
      const int gidx =
          elem_offset + (ibase + col) + (l0 + row) * 128 + k * 16384;
      sFV[row * 64 + col] = q[gidx] * v[gidx];
    }
    __syncthreads();

    for (int lc = 0; lc < BK; ++lc) {
#if P127_CC_ABLATE
#pragma unroll
      for (int m = 0; m < NJ; ++m) {
#pragma unroll
        for (int t = 0; t < TI; ++t) {
          sy[m][t] += 1.0;
        }
      }
#else
      double fvi[TI];
#pragma unroll
      for (int t = 0; t < TI; t += 2) {
        if constexpr (TI == 1) {
          fvi[0] = sFV[lc * 64 + i0];
        } else {
          const double2 fv =
              *reinterpret_cast<const double2 *>(&sFV[lc * 64 + i0 + t]);
          fvi[t] = fv.x;
          fvi[t + 1] = fv.y;
        }
      }
#pragma unroll
      for (int m = 0; m < NJ; m += 2) {
        const double2 dy =
            *reinterpret_cast<const double2 *>(&sDm[lc * 64 + jpack + m]);
#pragma unroll
        for (int t = 0; t < TI; ++t) {
          sy[m][t] += fvi[t] * dy.x;
          sy[m + 1][t] += fvi[t] * dy.y;
        }
      }
#endif
    }
  }

  for (int m = 0; m < NJ; ++m) {
    const int j = jbase + jpack + m;
    const int idx = elem_offset + ig + j * 128 + k * 16384;
#pragma unroll
    for (int t = 0; t < TI; t += 2) {
      if constexpr (TI == 1) {
        dqdt[idx] = dqdt[idx] - Escale[idx + npoint] * sy[m][0];
      } else {
        const double2 ey =
            *reinterpret_cast<const double2 *>(Escale + idx + t + npoint);
        double2 out = *reinterpret_cast<double2 *>(dqdt + idx + t);
        out.x -= ey.x * sy[m][t];
        out.y -= ey.y * sy[m][t + 1];
        *reinterpret_cast<double2 *>(dqdt + idx + t) = out;
      }
    }
  }
}

// Section 28 block-shape sweep: BDX is the threadIdx.x extent (BDY = 128/BDX
// is threadIdx.y).  tx indexes the i the thread owns, ty the four j.  BDX = 16
// is the HEAD form; 32 makes the D1D __ldg 32 doubles wide and the shared fill
// one store per row, 8 halves both.
__global__ __launch_bounds__(128, P255_CC_X_MINB)
void tendency_x_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  constexpr int TI = P255_CC_X_TI;
  constexpr int BDX = P255_CC_X_BDX;
  constexpr int BDY = 128 / BDX;
  constexpr int JROWS = 4 * BDY;
  constexpr int SQS = 33;
  constexpr int NCOL = 32 / BDX;
  static_assert(NCOL >= 1 && 32 % BDX == 0, "x: BDX must divide 32");
  __shared__ double sQ[JROWS * SQS];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int block0 = (int)blockIdx.x;
  const int nblock_pe = (P255_CC_BPE / 2) / TI;
  const int elem = block0 / nblock_pe;
  if (elem >= Ne) {
    return;
  }
  constexpr int NPAIR = 256 / (BDX * TI);
  constexpr int NQUAD = 256 / JROWS;
  int local_block = block0 % nblock_pe;
  const int k = local_block / (NQUAD * NPAIR);
  local_block %= NQUAD * NPAIR;
  const int quad_j = local_block / NPAIR;
  const int pair_i = local_block % NPAIR;
  int iv[TI];
#pragma unroll
  for (int u = 0; u < TI; ++u) {
    iv[u] = pair_i * (BDX * TI) + tx + BDX * u;
  }
  const int j0 = quad_j * JROWS + ty;
  const int elem_offset = elem * 16777216;

  double acc[TI][4];
#pragma unroll
  for (int u = 0; u < TI; ++u) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      acc[u][a] = 0.0;
    }
  }
  for (int ltile = 0; ltile < 8; ++ltile) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
      for (int c = 0; c < NCOL; ++c) {
        const int row = ty + BDY * r;
        const int col = tx + BDX * c;
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 6
        sQ[row * SQS + col] = 1.0;
#else
        const int gidx = elem_offset + (ltile * 32 + col) +
                         (j0 + BDY * r) * 256 + k * 65536;
        sQ[row * SQS + col] = q[gidx] * velocity[gidx];
#endif
      }
    }
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
      double d[TI];
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
#pragma unroll
      for (int u = 0; u < TI; ++u) {
        d[u] = 1.0;
      }
#else
#pragma unroll
      for (int u = 0; u < TI; ++u) {
        d[u] = __ldg(D1D + iv[u] + (ltile * 32 + t) * 256);
      }
#endif
      double f[4];
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        f[a] = sQ[(ty + BDY * a) * SQS + t];
      }
#pragma unroll
      for (int u = 0; u < TI; ++u) {
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          acc[u][a] += d[u] * f[a];
        }
      }
    }
#if P255_CC_ABLATE != 4
    if (ltile + 1 < 8) {
      __syncthreads();
    }
#endif
  }

  const int elem_face_offset = elem * 6 * 65536;
#if P255_CC_ABLATE == 3
#pragma unroll
  for (int u = 0; u < TI; ++u) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const int idx = elem_offset + iv[u] + (j0 + BDY * a) * 256 + k * 65536;
      dqdt[idx] = -acc[u][a];
    }
  }
#else
  double fb1[4], fb3[4];
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    const int fp = (j0 + BDY * a) + k * 256;
    fb1[a] = flux_bnd[elem_face_offset + 65536 + fp];
    fb3[a] = flux_bnd[elem_face_offset + 3 * 65536 + fp];
  }
#pragma unroll
  for (int u = 0; u < TI; ++u) {
    const double c1 = Lift1D[iv[u] + 256];
    const double c3 = Lift1D[iv[u] + 3 * 256];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const double lift = c1 * fb1[a] + c3 * fb3[a];
      const int idx = elem_offset + iv[u] + (j0 + BDY * a) * 256 + k * 65536;
      dqdt[idx] = -(__ldg(Escale + idx) * acc[u][a] + lift);
    }
  }
#endif
}

// y holds sD double buffered and P255_CC_Y_TI shared flux planes; a thread
// owns TI values of i and four of j.  TI = 2 with eight accumulators and eight
// blocks per SM was sections 15.69 / 15.74 / 15.80 / 15.92; TI = 8 with 168
// registers is section 26.6, worth 5.92% of the stage on its own.  TI = 16
// would need 64 KB of static shared and cannot be built.  P255_CC_Y_BDX is the
// section 28 block-shape knob (BDY = 128/BDX); BDX = 32 needs P255_CC_Y_LT = 8
// because sQ is TI*2*LT*BDX doubles and 48 KB of static shared is the wall.
__global__ __launch_bounds__(128, P255_CC_Y_MINB)
void tendency_y_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  constexpr int TI = P255_CC_Y_TI;
  constexpr int BDX = P255_CC_Y_BDX;
  constexpr int BDY = 128 / BDX;
  constexpr int LT = P255_CC_Y_LT;
  constexpr int JCOL = 4 * BDY;
  constexpr int NLT = 256 / LT;
  constexpr int NFILLD = (LT * JCOL) / 128;
  constexpr int NROW = LT / BDY;
  static_assert(LT % BDY == 0 && NROW >= 1, "y: LT must be a multiple of BDY");
  static_assert((LT * JCOL) % 128 == 0, "y: sD fill must be thread-aligned");
  __shared__ double sD[2][LT * JCOL];
  __shared__ double sQ[TI][2][LT * BDX];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int tid = ty * BDX + tx;
  const int block0 = (int)blockIdx.x;
  const int nblock_pe = (P255_CC_BPE / 2) / TI;
  const int elem = block0 / nblock_pe;
  if (elem >= Ne) {
    return;
  }
  constexpr int NPAIR = 256 / (BDX * TI);
  constexpr int NQUAD = 256 / JCOL;
  int local_block = block0 % nblock_pe;
  const int k = local_block / (NQUAD * NPAIR);
  local_block %= NQUAD * NPAIR;
  const int quad_j = local_block / NPAIR;
  const int pair_i = local_block % NPAIR;
  int iv[TI];
#pragma unroll
  for (int u = 0; u < TI; ++u) {
    iv[u] = pair_i * (BDX * TI) + tx + BDX * u;
  }
  const int j0 = quad_j * JCOL + ty;
  const int jbase = quad_j * JCOL;
  const int elem_offset = elem * 16777216;

  double acc[TI][4];
#pragma unroll
  for (int u = 0; u < TI; ++u) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      acc[u][a] = 0.0;
    }
  }
  int p = 0;
  int ft = 0;
#define P255_Y_FILL                                                           \
  do {                                                                        \
    _Pragma("unroll") for (int u = 0; u < TI; ++u)                            \
    {                                                                         \
      _Pragma("unroll") for (int r = 0; r < NROW; ++r)                        \
      {                                                                       \
        const int l = ty + BDY * r;                                           \
        const int gidx =                                                      \
            elem_offset + iv[u] + (ft * LT + l) * 256 + k * 65536;            \
        sQ[u][p][l * BDX + tx] = P255_QV(gidx);                               \
      }                                                                       \
    }                                                                         \
  } while (0)
#define P255_Y_FILLD                                                          \
  do {                                                                        \
    _Pragma("unroll") for (int s = 0; s < NFILLD; ++s)                        \
    {                                                                         \
      const int e = tid + 128 * s;                                            \
      const int l = e / JCOL;                                                 \
      const int m = e % JCOL;                                                 \
      sD[p][l * JCOL + P255_JPOS(m, BDY)] =                                   \
          P255_D(jbase + m + (ft * LT + l) * 256);                            \
    }                                                                         \
  } while (0)
  P255_Y_FILLD;
  P255_Y_FILL;
#if P255_CC_ABLATE != 4
  __syncthreads();
#endif
  for (int ltile = 0; ltile < NLT; ++ltile) {
#pragma unroll
#if P255_CC_ABLATE == 1
    for (int t = 0; t < 1; ++t)
#else
    for (int t = 0; t < LT; ++t)
#endif
    {
      const double2 dj0 =
          *reinterpret_cast<const double2 *>(&sD[p][t * JCOL + 2 * ty]);
      const double2 dj1 = *reinterpret_cast<const double2 *>(
          &sD[p][t * JCOL + 2 * BDY + 2 * ty]);
      double dj[4];
      dj[0] = dj0.x;
      dj[1] = dj0.y;
      dj[2] = dj1.x;
      dj[3] = dj1.y;
      double f[TI];
#pragma unroll
      for (int u = 0; u < TI; ++u) {
        f[u] = sQ[u][p][t * BDX + tx];
      }
#pragma unroll
      for (int u = 0; u < TI; ++u) {
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          acc[u][a] += f[u] * dj[a];
        }
      }
    }
    if (ltile + 1 < NLT) {
      p ^= 1;
      ft = ltile + 1;
      P255_Y_FILLD;
      P255_Y_FILL;
#if P255_CC_ABLATE != 4
      __syncthreads();
#endif
    }
  }
#undef P255_Y_FILL
#undef P255_Y_FILLD

  const int npoint = 16777216 * Ne;
  const int elem_face_offset = elem * 6 * 65536;
#if P255_CC_ABLATE == 3
#pragma unroll
  for (int u = 0; u < TI; ++u) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const int idx = elem_offset + iv[u] + (j0 + BDY * a) * 256 + k * 65536;
      dqdt[idx] = dqdt[idx] - acc[u][a];
    }
  }
#else
  double c0[4], c2[4];
#pragma unroll
  for (int a = 0; a < 4; ++a) {
    c0[a] = Lift1D[j0 + BDY * a];
    c2[a] = Lift1D[j0 + BDY * a + 2 * 256];
  }
#pragma unroll
  for (int u = 0; u < TI; ++u) {
    const int fp = iv[u] + k * 256;
    const double fb0 = flux_bnd[elem_face_offset + fp];
    const double fb2 = flux_bnd[elem_face_offset + 2 * 65536 + fp];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const double lift = c0[a] * fb0 + c2[a] * fb2;
      const int idx = elem_offset + iv[u] + (j0 + BDY * a) * 256 + k * 65536;
      dqdt[idx] = dqdt[idx] - (Escale[idx + npoint] * acc[u][a] + lift);
    }
  }
#endif
}

// z contracts the whole element at once: a thread owns P255_CC_Z_TL lines of
// the linear (i,j) index and four k values.  2 lines / 8 accumulators / 8
// blocks per SM was sections 15.76 and 15.81; 8 lines with 166 registers is
// section 26.6, worth 2.56% of the stage on its own (4 lines is 1.75%, and at
// the old 64-register budget it spills 8 bytes).  P255_CC_ZDG drops the shared
// D panel and reads D1D through __ldg, which is what section 15.8 adopted for
// x and what section 26.5 measured at +11.26% here.  P255_CC_Z_BDX is the
// section 28 block-shape knob (BDY = 128/BDX).
__global__ __launch_bounds__(128, P255_CC_Z_MINB)
void tendency_z_p255_cc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  constexpr int TL = P255_CC_Z_TL;
  constexpr int BDX = P255_CC_Z_BDX;
  constexpr int BDY = 128 / BDX;
  constexpr int LT = P255_CC_Z_LT;
  constexpr int KCOL = 4 * BDY;
  constexpr int NLT = 256 / LT;
  constexpr int NFILLD = (LT * KCOL) / 128;
  constexpr int NROW = LT / BDY;
  static_assert(LT % BDY == 0 && NROW >= 1, "z: LT must be a multiple of BDY");
  static_assert((LT * KCOL) % 128 == 0, "z: sD fill must be thread-aligned");
#if !P255_CC_ZDG
  __shared__ double sD[LT * KCOL];
#endif
  __shared__ double sQ[TL][LT * BDX];

  const int tx = (int)threadIdx.x;
  const int ty = (int)threadIdx.y;
  const int tid = ty * BDX + tx;
  const int block0 = (int)blockIdx.x;
  const int nblock_pe = (P255_CC_BPE / 2) / TL;
  const int elem = block0 / nblock_pe;
  if (elem >= Ne) {
    return;
  }
  const int local_block = block0 % nblock_pe;
  constexpr int npair = 65536 / (BDX * TL);
  const int quad_k = local_block / npair;
  const int pair = local_block % npair;
  int line[TL];
#pragma unroll
  for (int u = 0; u < TL; ++u) {
    line[u] = pair * (BDX * TL) + tx + BDX * u;
  }
  const int k0 = quad_k * KCOL + ty;
  const int kbase = quad_k * KCOL;
  const int elem_offset = elem * 16777216;

  double acc[TL][4];
#pragma unroll
  for (int u = 0; u < TL; ++u) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      acc[u][a] = 0.0;
    }
  }
  for (int ltile = 0; ltile < NLT; ++ltile) {
#if !P255_CC_ZDG
#pragma unroll
    for (int s = 0; s < NFILLD; ++s) {
      const int e = tid + 128 * s;
      const int l = e / KCOL;
      const int m = e % KCOL;
      sD[l * KCOL + P255_JPOS(m, BDY)] =
          P255_D(kbase + m + (ltile * LT + l) * 256);
    }
#endif
#pragma unroll
    for (int u = 0; u < TL; ++u) {
#pragma unroll
      for (int r = 0; r < NROW; ++r) {
        const int l = ty + BDY * r;
        const int gidx = elem_offset + line[u] + (ltile * LT + l) * 65536;
        sQ[u][l * BDX + tx] = P255_QV(gidx);
      }
    }
#if P255_CC_ABLATE != 4
    __syncthreads();
#endif
#pragma unroll
#if P255_CC_ABLATE == 1
    for (int t = 0; t < 1; ++t)
#else
    for (int t = 0; t < LT; ++t)
#endif
    {
      double dk[4];
#if P255_CC_ZDG
#if P255_CC_ABLATE == 2 || P255_CC_ABLATE == 5
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        dk[a] = 1.0;
      }
#else
#pragma unroll
      for (int a = 0; a < 4; ++a) {
        dk[a] = __ldg(D1D + k0 + BDY * a + (ltile * LT + t) * 256);
      }
#endif
#else
      {
        const double2 dk0 =
            *reinterpret_cast<const double2 *>(&sD[t * KCOL + 2 * ty]);
        const double2 dk1 = *reinterpret_cast<const double2 *>(
            &sD[t * KCOL + 2 * BDY + 2 * ty]);
        dk[0] = dk0.x;
        dk[1] = dk0.y;
        dk[2] = dk1.x;
        dk[3] = dk1.y;
      }
#endif
      double f[TL];
#pragma unroll
      for (int u = 0; u < TL; ++u) {
        f[u] = sQ[u][t * BDX + tx];
      }
#pragma unroll
      for (int u = 0; u < TL; ++u) {
#pragma unroll
        for (int a = 0; a < 4; ++a) {
          acc[u][a] += f[u] * dk[a];
        }
      }
    }
#if P255_CC_ABLATE != 4
    if (ltile + 1 < NLT) {
      __syncthreads();
    }
#endif
  }

  const int npoint = 16777216 * Ne;
  const int elem_face_offset = elem * 6 * 65536;
#if P255_CC_ABLATE == 3
#pragma unroll
  for (int u = 0; u < TL; ++u) {
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const int idx = elem_offset + line[u] + (k0 + BDY * a) * 65536;
      dqdt[idx] = dqdt[idx] - acc[u][a];
    }
  }
#else
#pragma unroll
  for (int u = 0; u < TL; ++u) {
    const double fb4 = flux_bnd[elem_face_offset + 4 * 65536 + line[u]];
    const double fb5 = flux_bnd[elem_face_offset + 5 * 65536 + line[u]];
#pragma unroll
    for (int a = 0; a < 4; ++a) {
      const int k = k0 + BDY * a;
      const double lift =
          Lift1D[k + 4 * 256] * fb4 + Lift1D[k + 5 * 256] * fb5;
      const int idx = elem_offset + line[u] + k * 65536;
      dqdt[idx] = dqdt[idx] - (Escale[idx + 2 * npoint] * acc[u][a] + lift);
    }
  }
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
  const size_t smem_xz = 3 * 64 * P63_CC_XZ_BK * sizeof(double);
  const size_t smem_y = 2 * 64 * P63_CC_Y_BK * sizeof(double);
  cudaFuncSetAttribute(tendency_fused_p63_xz_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_xz);
  cudaFuncSetAttribute(tendency_fused_p63_y_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_y);
  tendency_fused_p63_xz_cc_kernel<<<nblock, P63_CC_XZ_THREADS, smem_xz, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p63_y_cc_kernel<<<nblock, P63_CC_Y_THREADS, smem_y, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p63_cc_kernels");
}

extern "C" void launch_tendency_fused_p127(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = P127_CC_BPE * Ne;
  const size_t smem_xz = 4 * 64 * P127_CC_XZ_BK * sizeof(double);
  const size_t smem_y = 2 * 64 * P127_CC_Y_BK * sizeof(double);
  cudaFuncSetAttribute(tendency_fused_p127_xz_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_xz);
  cudaFuncSetAttribute(tendency_fused_p127_y_cc_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_y);
  tendency_fused_p127_xz_cc_kernel<<<nblock, P127_CC_XZ_THREADS, smem_xz, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p127_y_cc_kernel<<<nblock, P127_CC_Y_THREADS, smem_y, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda_cc_hp("tendency_fused_p127_cc_kernels");
}

extern "C" void launch_tendency_xyz_p255(
    double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, int Ne)
{
  const dim3 threads_x(P255_CC_X_BDX, 128 / P255_CC_X_BDX);
  const dim3 threads_y(P255_CC_Y_BDX, 128 / P255_CC_Y_BDX);
  const dim3 threads_z(P255_CC_Z_BDX, 128 / P255_CC_Z_BDX);
  const int nblock_x = ((P255_CC_BPE / 2) / P255_CC_X_TI) * Ne;
  const int nblock_y = ((P255_CC_BPE / 2) / P255_CC_Y_TI) * Ne;
  const int nblock_z = ((P255_CC_BPE / 2) / P255_CC_Z_TL) * Ne;
  tendency_x_p255_cc_kernel<<<nblock_x, threads_x, 0, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_y_p255_cc_kernel<<<nblock_y, threads_y, 0, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_z_p255_cc_kernel<<<nblock_z, threads_z, 0, dg_cuda_stream>>>(
      dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, Ne);
  check_cuda_cc_hp("tendency_xyz_p255_cc_kernels");
}
