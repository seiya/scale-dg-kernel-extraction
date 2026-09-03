#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#include "fused_kernel_geom.h"

// FP64 Tensor Core GEMM helpers: mma.sync.aligned.m8n8k4.f64
// Fragment map (SM80+):
//   lane = thread % 32
//   A(8x4): A[lane/4][lane%4]
//   B(4x8): B[lane%4][lane/4]
//   C(8x8): C[lane/4][(lane%4)*2] and +1
//
// UseTc=false walks the same fragments with DFMA and warp shuffles so
// CUDAFORTRAN_FUSED_DFMA and CUDAFORTRAN_FUSED_TC share every other instruction.

template <bool UseTc>
__device__ __forceinline__ void mma_m8n8k4_f64(
    double &d0, double &d1, double a, double b, double c0, double c1)
{
  if constexpr (UseTc) {
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
        : "=d"(d0), "=d"(d1)
        : "d"(a), "d"(b), "d"(c0), "d"(c1));
  } else {
    const int lane = (int)threadIdx.x & 31;
    const int row = lane >> 2;
    const int col0 = (lane & 3) * 2;
    double acc0 = c0;
    double acc1 = c1;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const double ak = __shfl_sync(0xffffffff, a, (row << 2) + k);
      const double bk0 = __shfl_sync(0xffffffff, b, (col0 << 2) + k);
      const double bk1 = __shfl_sync(0xffffffff, b, ((col0 + 1) << 2) + k);
      acc0 = fma(ak, bk0, acc0);
      acc1 = fma(ak, bk1, acc1);
    }
    d0 = acc0;
    d1 = acc1;
  }
}

__device__ __forceinline__ void mma_reset(double &c0, double &c1)
{
  c0 = 0.0;
  c1 = 0.0;
}

// Out-of-role ablation (P7_TC_M16N8K8, off by default): the same K = 8
// contraction expressed as one mma.sync.m16n8k8 instead of two
// mma.sync.m8n8k4.  Fragment map (verified against a CPU reference on
// sm_100), g = lane>>2, t = lane&3:
//   A(16x8): a0 = A[g][t]  a1 = A[g+8][t]  a2 = A[g][t+4]  a3 = A[g+8][t+4]
//   B(8x8) : b0 = B[t][g]  b1 = B[t+4][g]
//   C(16x8): c0 = C[g][2t]  c1 = C[g][2t+1]  c2 = C[g+8][2t]  c3 = +1
// The m8n8k4 A and B operands of the k0 = 0 and k0 = 4 halves are exactly
// a0/a2 and b0/b1, so the drop-in sets a1 = a3 = 0 and drops c2, c3.  The
// UseTc = false arm walks the same fragments with DFMA and shuffles, so
// FUSED_DFMA and FUSED_TC stay iso-schedule under the knob as well.
template <bool UseTc>
__device__ __forceinline__ void mma_m16n8k8_f64(
    double &d0, double &d1, double &d2, double &d3, double a0, double a1,
    double a2, double a3, double b0, double b1, double c0, double c1,
    double c2, double c3)
{
  if constexpr (UseTc) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
        : "=d"(d0), "=d"(d1), "=d"(d2), "=d"(d3)
        : "d"(a0), "d"(a1), "d"(a2), "d"(a3), "d"(b0), "d"(b1), "d"(c0),
          "d"(c1), "d"(c2), "d"(c3));
  } else {
    const int lane = (int)threadIdx.x & 31;
    const int g = lane >> 2;
    const int n0 = (lane & 3) * 2;
    double acc0 = c0, acc1 = c1, acc2 = c2, acc3 = c3;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const double alo = (k < 4) ? a0 : a2;
      const double ahi = (k < 4) ? a1 : a3;
      const double bb = (k < 4) ? b0 : b1;
      const int src_a = (g << 2) + (k & 3);
      const double ak0 = __shfl_sync(0xffffffff, alo, src_a);
      const double ak1 = __shfl_sync(0xffffffff, ahi, src_a);
      const double bk0 = __shfl_sync(0xffffffff, bb, (n0 << 2) + (k & 3));
      const double bk1 = __shfl_sync(0xffffffff, bb, ((n0 + 1) << 2) + (k & 3));
      acc0 = fma(ak0, bk0, acc0);
      acc1 = fma(ak0, bk1, acc1);
      acc2 = fma(ak1, bk0, acc2);
      acc3 = fma(ak1, bk1, acc3);
    }
    d0 = acc0;
    d1 = acc1;
    d2 = acc2;
    d3 = acc3;
  }
}

// One K = 8 contraction on an 8x8 output tile: the production form is the two
// m8n8k4 the path is defined by; P7_TC_M16N8K8 swaps in the wider shape.
template <bool UseTc>
__device__ __forceinline__ void mma_k8_8x8(
    double &c0, double &c1, double a0, double b0, double a1, double b1)
{
#if P7_TC_M16N8K8
  double d2 = 0.0, d3 = 0.0;
  mma_m16n8k8_f64<UseTc>(c0, c1, d2, d3, a0, 0.0, a1, 0.0, b0, b1, c0, c1, 0.0,
                         0.0);
#else
  mma_m8n8k4_f64<UseTc>(c0, c1, a0, b0, c0, c1);
  mma_m8n8k4_f64<UseTc>(c0, c1, a1, b1, c0, c1);
#endif
}

// Shared-memory layouts for the p=7 fused Tensor Core kernel.
//
// ncu (Slurm job 43554) showed the previous natural layouts caused a 3.0-way
// bank conflict on shared loads and a 5.8-way conflict on shared stores, so
// 46% of all shared wavefronts were excess. FP64 shared accesses are serviced
// in half-warp phases, so an access is conflict free when the 16 lanes of a
// phase hit 16 distinct addresses modulo 16 doubles.
//
// sDfrag: the 1D derivative matrix in m8n8k4 fragment order,
//   sDfrag[b*32 + r*4 + c] = D1D[r + (b*4 + c)*8],  b = 0,1.
//   Lane L reads sDfrag[b*32 + (L>>2)*4 + (L&3)], so a half-warp covers 16
//   consecutive doubles.
// sFluxX / sFluxY: natural node order i + 8*j + 64*k permuted by sw_xy(),
//   which folds bit 4 of the index into the otherwise unused bit 2 and turns
//   the 2-way operand loads into conflict-free ones.
// sFluxZ: natural node order permuted by sw_z(); the z contraction strides by
//   64 doubles, so bits 6-7 of the index are folded into bits 2-3 to break a
//   4-way conflict.
// sDz: natural node order permuted by sw_dz(), which folds bit 6 (the low bit
//   of the accumulator row) into bit 3, because the m8n8k4 accumulator holds
//   C[r][2c] and C[r][2c+1] with r = lane>>2, so the natural order would make
//   an accumulator store phase hit only 4 distinct banks. The x and y
//   derivatives no longer pass through shared memory at all; see the note on
//   the transposed accumulators in the kernel.
//
// sLift: the separable face lift coefficients Lift1D(Nq,6). Lift_mat(i,j,k,f)
//   varies in one volume index only (j for faces 1 and 3, i for faces 2 and 4,
//   k for faces 5 and 6), which is how mod_mesh.f90 already derives Lift1D for
//   the p=255 and GEMM paths. The 512x6 dense form cost 12 global loads per
//   thread, 768 of the 3684 global sectors an element moves; the 48 distinct
//   values live in shared memory instead. This only pays off once the kernel
//   reaches 8 blocks per SM: at 6 blocks the same substitution measured 1.3%
//   slower, because the epilogue is bound by L1/TEX, which serves the shared
//   loads and the global loads alike.
//
// sDz aliases sFluxZ. sw_z() and sw_dz() move indices across the 8-column
// range a warp owns, so the z accumulators need a block-wide barrier before
// they overwrite the flux. Reusing the flux buffer is what keeps the block at
// 15.87 KB instead of 20 KB and lets __launch_bounds__(256, 8) reach 8 blocks
// per SM; ncu (Slurm job 43734) showed the kernel held at 6 blocks and 72%
// achieved occupancy, limited by both registers and shared memory at once.
//
// Two measured results kept this kernel away from a fully conflict-free store.
// Writing the accumulator pair with one 16-byte store removes the store
// conflicts entirely but is slower (597 us against 504 us, MIO throttle 11.16
// against 1.73). A permutation that also makes the 8-byte stores conflict free
// lowers the store wavefronts from 15.0 M to 11.5 M and is likewise slower
// (596 us). Both are recorded in tc_paper_survey_2407.09621.md section 5.


// Illegal ablations that price the shared-memory store bank conflicts of the
// p=7 fused Tensor Core kernel.  Section 13.5 of
// reports/tc_paper_survey_2407.09621.md recorded that staging the two x-normal
// faces raised the shared store conflicts from 2.41 M to 4.10 M and left the
// mechanism open.  P7_ABL_SHST is a bit mask over the three groups of shared
// stores the kernel has; a set bit replaces that group's addresses with
// lane-linear ones, which are conflict free by construction and keep the
// instruction count, the access width and the number of active lanes.
//   1  the six sFluxX / sFluxY / sFluxZ stores (drops sw_xy / sw_z)
//   2  the eight stage_xface() stores (drops the j + 8k fold and the plane
//      stride of 72)
//   4  the two sDz stores (drops sw_dz)
// The values are wrong in every non-zero setting, so these builds exist only
// to measure the ceiling.
#ifndef P7_ABL_SHST
#define P7_ABL_SHST 0
#endif
// Attribution build: revert faces 2 and 4 to the pre-section-13 form, in which
// the M side is gathered from global through VMapM and nothing is staged in
// shared memory.  Numerically correct; it is what the 2.41 M figure was
// measured on.
#ifndef P7_ABL_NOSTAGE
#define P7_ABL_NOSTAGE 0
#endif

__device__ __forceinline__ int sw_xy(int idx)
{
  return idx ^ (((idx >> 4) & 1) << 2);
}

__device__ __forceinline__ int sw_z(int idx)
{
  return idx ^ (((idx >> 6) & 3) << 2);
}

__device__ __forceinline__ int sw_dz(int idx)
{
  return idx ^ (((idx >> 6) & 1) << 3);
}

#if (P7_ABL_SHST & 1)
#define P7_STXY(i) (i)
#define P7_STZ(i) (i)
#else
#define P7_STXY(i) sw_xy(i)
#define P7_STZ(i) sw_z(i)
#endif

// Stage the M-side fields of the two x-normal faces.  Node i + 8j + 64k lies
// on face 2 when i == 7 and on face 4 when i == 0, and Fmask numbers both
// faces by j + 8k, so the owning thread writes one face point of one plane.
//
// The fields are paired into two double2 arrays rather than kept field major.
// Only eight lanes of a warp own an x-plane node, so the cost of this staging
// is the number of shared instructions the whole warp issues, not the number
// of values it writes: four 8-byte stores per node measured 2.8x the MIO
// throttle of the version without staging.  Paired, one node costs two
// stores and one face point two loads.  The plane stride of 68 double2 keeps
// the two planes off the same banks, and consecutive face points read
// consecutive double2, which is conflict free.
#define XFACE_PLANE 72
__device__ __forceinline__ void stage_xface(double *sM, int node, double q,
                                            double u, double v, double w)
{
  const int i = node & 7;
  if (i == 7 || i == 0) {
#if (P7_ABL_SHST & 2)
    double *const m = sM + (node & 31);
#else
    double *const m = sM + ((i == 7) ? 0 : XFACE_PLANE) + ((node >> 3) & 7) +
                      ((node >> 6) << 3);
#endif
    m[0] = q;
    m[144] = u;
    m[288] = v;
    m[432] = w;
  }
}

template <bool UseTc>
__global__ __launch_bounds__(P7_THREADS, P7_BPSM) void tendency_fused_p7_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sDfrag[64];
  __shared__ __align__(16) double sLift[48];
  __shared__ __align__(16) double sflux_bnd[384];
  // M-side q, u, v and w of the two x-normal faces, indexed by face point.
  // Fmask gives faces 2 and 4 the nodes 8j + 64k with i fixed, so a warp of
  // consecutive face points gathers with a stride of 8 doubles and puts every
  // lane in its own sector: 32 sectors per warp instruction where the y- and
  // z-normal faces need 8.  ncu (job 49589, source page) attributed all
  // 24.77 M excessive load sectors of this kernel to those four gathers.  The
  // values are already in registers here, because the same element's volume
  // loads produced them, so the two planes are staged instead of re-read.
  // Field-major with a padded plane stride of 72, so that a face-point warp
  // reads 32 consecutive doubles and the two planes of one store phase do not
  // land on the same bank.
#if !P7_ABL_NOSTAGE
  __shared__ __align__(16) double sMface[4 * 144];
#endif
  __shared__ __align__(16) double sFluxX[512], sFluxY[512], sFluxZ[512];
  // The z derivative overwrites the z flux it consumes, so the block needs
  // 15.87 KB instead of 20 KB. See the aliasing note above.
  double *const sDz = sFluxZ;

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int node1 = tid;
  const int node2 = tid + 256;
  const int elem_offset = elem * 512;
  const int face_offset = elem * 384;
  const int npoint = 512 * Ne;
  const int nface = 384 * Ne;

  if (tid < 64) {
    const int r = (tid >> 2) & 7;
    const int c = tid & 3;
    const int b = tid >> 5;
    sDfrag[tid] = D1D[r + (b * 4 + c) * 8];
  } else if (tid < 112) {
    sLift[tid - 64] = Lift1D[tid - 64];
  }
  const int idx1 = elem_offset + node1;
  const int idx2 = elem_offset + node2;
  {
    const double q1 = q[idx1], u1 = u[idx1], v1 = v[idx1], w1 = w[idx1];
    sFluxX[P7_STXY(node1)] = q1 * u1;
    sFluxY[P7_STXY(node1)] = q1 * v1;
    sFluxZ[P7_STZ(node1)] = q1 * w1;
#if !P7_ABL_NOSTAGE
    stage_xface(sMface, node1, q1, u1, v1, w1);
#endif
  }
  {
    const double q2 = q[idx2], u2 = u[idx2], v2 = v[idx2], w2 = w[idx2];
    sFluxX[P7_STXY(node2)] = q2 * u2;
    sFluxY[P7_STXY(node2)] = q2 * v2;
    sFluxZ[P7_STZ(node2)] = q2 * w2;
#if !P7_ABL_NOSTAGE
    stage_xface(sMface, node2, q2, u2, v2, w2);
#endif
  }
  // sMface is filled by whichever thread owns the node, which is not the
  // thread that reads it as a face point.
  __syncthreads();

  int fp = tid;
  int fidx = face_offset + fp;
  // Face points 64-127 are face 2 and 192-255 are face 4, so bit 6 of fp
  // selects the x-normal faces and bit 7 selects which of the two planes.
  int iP = VMapP[fidx] - 1;
  const double fn1 = normal_fn[fidx];
  const double fn2 = normal_fn[fidx + nface];
  const double fn3 = normal_fn[fidx + 2 * nface];
  double qM, VelM;
#if P7_ABL_NOSTAGE
  {
    const int iM = VMapM[fidx] - 1;
    qM = q[iM];
    VelM = u[iM] * fn1 + v[iM] * fn2 + w[iM] * fn3;
  }
#else
  if ((fp & 64) != 0) {
    const double *const m =
        sMface + (((fp & 128) != 0) ? XFACE_PLANE : 0) + (fp & 63);
    qM = m[0];
    VelM = m[144] * fn1 + m[288] * fn2 + m[432] * fn3;
  } else {
    const int iM = VMapM[fidx] - 1;
    qM = q[iM];
    VelM = u[iM] * fn1 + v[iM] * fn2 + w[iM] * fn3;
  }
#endif
  double qP = q[iP];
  double VelP = u[iP] * fn1 + v[iP] * fn2 + w[iP] * fn3;
  double alpha = 0.5 * fabs(VelP + VelM);
  sflux_bnd[fp] = 0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  if (tid < 128) {
    // Faces 5 and 6 keep the global gather: Fmask gives them 32 consecutive
    // nodes per warp, which is already the ideal sector count.
    fp = tid + 256;
    fidx = face_offset + fp;
    const int iM = VMapM[fidx] - 1;
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

  const int row = lane >> 2;
  const int colk = lane & 3;
  const int k = warp;
  const int j0_c = colk * 2;
  // The A operand of the z contraction and the D1D operand of the x and y
  // contractions are the same fragment element, D1D[row][colk].
  const int frag = (row << 2) + colk;

  // Every paired shared access below derives the second address from the first
  // with one XOR by a constant, instead of swizzling a second index.  The
  // k0 = 4 operand differs from the k0 = 0 one in a single index bit that no
  // swizzle here reads, and the c1 accumulator element is one node away from
  // the c0 one, which sw_dz() leaves in place. Section 11 of
  // reports/tc_paper_survey_2407.09621.md records what this bought.

  // Dz = D * Fz_panel; warp owns 8 (i,j) columns, all k.  k0 = 4 sets bit 8 of
  // the node index, above the bits sw_z() folds.  The z panel is contracted
  // first because it is the only derivative that has to travel through shared
  // memory: its two barriers then sit before the x and y accumulators exist.
  double c0, c1;
  {
    const int fz = sw_z(((warp << 3) + row) + (colk << 6));
    mma_reset(c0, c1);
    mma_k8_8x8<UseTc>(c0, c1, sDfrag[frag], sFluxZ[fz], sDfrag[frag + 32],
                      sFluxZ[fz ^ 256]);
  }
  // sw_z() and sw_dz() permute across warp boundaries, so the z panel needs a
  // block-wide barrier before it overwrites the flux it was read from.
  __syncthreads();
#if (P7_ABL_SHST & 4)
  sDz[tid] = c0;
  sDz[tid + 256] = c1;
#else
  const int dz_c = sw_dz(((warp << 3) + j0_c) + (row << 6));
  sDz[dz_c] = c0;
  sDz[dz_c ^ 1] = c1;
#endif
  __syncthreads();

  // The x and y derivatives never go through shared memory: the thread that
  // computes them is the thread that assembles them.  That removes four shared
  // stores and four shared loads per thread together with their address
  // arithmetic, and it leaves sFluxX and sFluxY read-only for the whole
  // kernel, so the two __syncwarp() calls that used to guard the in-place
  // overwrite are gone.
  //
  // Both contractions are evaluated transposed, C = (D*Fx)^T and C = (Fy*D^T)^T,
  // which costs nothing: with m8n8k4 the transpose is the same two operand
  // values passed in the opposite order.  It is what makes the epilogue
  // coalesce.  The accumulator holds C[lane>>2][2*(lane&3)] and its neighbour,
  // so the untransposed form gave thread lane the nodes
  //   (lane>>2) + 16*(lane&3) + 64*warp  and  + 8,
  // whose warp footprint is four 64-byte runs spread over 448 bytes: same
  // sectors as a contiguous access but twice the cache lines, and measurably
  // slower on a kernel that sits at 95% L1/TEX.  Transposed, the same thread
  // owns nodes 2*tid and 2*tid + 1, so a warp covers 64 consecutive nodes and
  // each of q's neighbours in the epilogue is one aligned 16-byte access.
  const int n0 = tid << 1;
  const int nidx0 = elem_offset + n0;

  // Dx^T = (D * Fx)^T on this k-plane.  sw_xy() flips bit 2 as a function of
  // bit 4, and k0 = 4 sets bit 2 of the node index, so the operands are fx and
  // fx^4.
  double acc0, acc1;
  {
    const int fx = sw_xy(colk + (row << 3) + (k << 6));
    const double2 es = *reinterpret_cast<const double2 *>(Escale + nidx0);
    mma_reset(c0, c1);
    mma_k8_8x8<UseTc>(c0, c1, sFluxX[fx], sDfrag[frag], sFluxX[fx ^ 4],
                      sDfrag[frag + 32]);
    acc0 = es.x * c0;
    acc1 = es.y * c1;
  }

  // Dy^T = (Fy * D^T)^T.  k0 = 4 sets bit 5 of the node index here, again a
  // bit sw_xy() does not read.
  {
    const int fy = sw_xy(row + (colk << 3) + (k << 6));
    const double2 es =
        *reinterpret_cast<const double2 *>(Escale + nidx0 + npoint);
    mma_reset(c0, c1);
    mma_k8_8x8<UseTc>(c0, c1, sDfrag[frag], sFluxY[fy], sDfrag[frag + 32],
                      sFluxY[fy ^ 32]);
    acc0 += es.x * c0;
    acc1 += es.y * c1;
  }

  // sw_dz() folds bit 6 into bit 3 and n0 is even, so the second node of the
  // pair is the neighbour of the first in sDz as well.
  {
    const double2 dz = *reinterpret_cast<const double2 *>(sDz + sw_dz(n0));
    const double2 es =
        *reinterpret_cast<const double2 *>(Escale + nidx0 + 2 * npoint);
    acc0 += es.x * dz.x;
    acc1 += es.y * dz.y;
  }

  // The node pair differs in i only, so faces 2 and 4 (which vary in j) and
  // the lift coefficients that go with them are shared between the two, while
  // faces 1, 3, 5 and 6 shift by one face point.
  const int i0 = colk * 2;
  const int face1 = i0 + (k << 3);
  const int face2 = 64 + row + (k << 3);
  const int face5 = 256 + (n0 & 63);
  const double lf1 = sLift[row];
  const double lf3 = sLift[row + 16];
  const double lf5 = sLift[k + 32];
  const double lf6 = sLift[k + 40];
  const double fb2 = sflux_bnd[face2];
  const double fb4 = sflux_bnd[face2 + 128];
  const double lift0 = lf1 * sflux_bnd[face1] + sLift[i0 + 8] * fb2 +
                       lf3 * sflux_bnd[face1 + 128] + sLift[i0 + 24] * fb4 +
                       lf5 * sflux_bnd[face5] + lf6 * sflux_bnd[face5 + 64];
  const double lift1 = lf1 * sflux_bnd[face1 + 1] + sLift[i0 + 9] * fb2 +
                       lf3 * sflux_bnd[face1 + 129] + sLift[i0 + 25] * fb4 +
                       lf5 * sflux_bnd[face5 + 1] + lf6 * sflux_bnd[face5 + 65];

  *reinterpret_cast<double2 *>(dqdt + nidx0) =
      make_double2(-(acc0 + lift0), -(acc1 + lift1));
}

// Output-tile (A) variant of the kernel above: one warp owns P7_TC_KP k
// planes instead of one, so the block has 256/KP threads and every thread
// holds 2*KP outputs.  It is a separate function, not a template parameter of
// the production kernel, because the production kernel sits at exactly 32
// registers with 8 blocks per SM and any restructuring of its two node loads
// or its epilogue costs it a spill.  Measured in section 23 of
// reports/tc_paper_survey_2407.09621.md.
template <bool UseTc>
__global__ __launch_bounds__(P7_THREADS, P7_BPSM) void tendency_fused_p7_tile_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sDfrag[64];
  __shared__ __align__(16) double sLift[48];
  __shared__ __align__(16) double sflux_bnd[384];
  // M-side q, u, v and w of the two x-normal faces, indexed by face point.
  // Fmask gives faces 2 and 4 the nodes 8j + 64k with i fixed, so a warp of
  // consecutive face points gathers with a stride of 8 doubles and puts every
  // lane in its own sector: 32 sectors per warp instruction where the y- and
  // z-normal faces need 8.  ncu (job 49589, source page) attributed all
  // 24.77 M excessive load sectors of this kernel to those four gathers.  The
  // values are already in registers here, because the same element's volume
  // loads produced them, so the two planes are staged instead of re-read.
  // Field-major with a padded plane stride of 72, so that a face-point warp
  // reads 32 consecutive doubles and the two planes of one store phase do not
  // land on the same bank.
#if !P7_ABL_NOSTAGE
  __shared__ __align__(16) double sMface[4 * 144];
#endif
  __shared__ __align__(16) double sFluxX[512], sFluxY[512], sFluxZ[512];
  // The z derivative overwrites the z flux it consumes, so the block needs
  // 15.87 KB instead of 20 KB. See the aliasing note above.
  double *const sDz = sFluxZ;

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int KP = P7_TC_KP;
  constexpr int THR = P7_THREADS;
  constexpr int NWARP = P7_NWARP;
  const int elem_offset = elem * 512;
  const int face_offset = elem * 384;
  const int npoint = 512 * Ne;
  const int nface = 384 * Ne;

  // 64 fragment elements and 48 lift coefficients.  With 112 threads or more
  // the one-shot form the production kernel uses is kept verbatim; only KP = 4
  // (64 threads) needs the strided loop, and writing it that way for every KP
  // cost 4% at KP = 2.
  if constexpr (THR >= 112) {
    if (tid < 64) {
      const int r = (tid >> 2) & 7;
      const int c = tid & 3;
      const int b = tid >> 5;
      sDfrag[tid] = D1D[r + (b * 4 + c) * 8];
    } else if (tid < 112) {
      sLift[tid - 64] = Lift1D[tid - 64];
    }
  } else {
    for (int t = tid; t < 112; t += THR) {
      if (t < 64) {
        const int r = (t >> 2) & 7;
        const int c = t & 3;
        const int b = t >> 5;
        sDfrag[t] = D1D[r + (b * 4 + c) * 8];
      } else {
        sLift[t - 64] = Lift1D[t - 64];
      }
    }
  }
  // One thread owns 2*KP nodes: KP = 1 is the two nodes tid and tid + 256 the
  // production form has always used.
#pragma unroll
  for (int t = 0; t < 2 * KP; ++t) {
    const int node = tid + t * THR;
    const int idx = elem_offset + node;
    const double qn = q[idx], un = u[idx], vn = v[idx], wn = w[idx];
    sFluxX[P7_STXY(node)] = qn * un;
    sFluxY[P7_STXY(node)] = qn * vn;
    sFluxZ[P7_STZ(node)] = qn * wn;
#if !P7_ABL_NOSTAGE
    stage_xface(sMface, node, qn, un, vn, wn);
#endif
  }
  // sMface is filled by whichever thread owns the node, which is not the
  // thread that reads it as a face point.
  __syncthreads();

  // 384 face points covered THR at a time.  At THR = 256 this unrolls to the
  // two passes the production form was written as: the whole 0-255 range and
  // then faces 5 and 6 under tid < 128.  Faces 5 and 6 keep the global gather,
  // because Fmask gives them 32 consecutive nodes per warp, which is already
  // the ideal sector count.
#pragma unroll
  for (int f = 0; f < NFPTOT7; f += THR) {
    const int fp = tid + f;
    if (fp >= NFPTOT7) {
      continue;
    }
    const int fidx = face_offset + fp;
    // Face points 64-127 are face 2 and 192-255 are face 4, so bit 6 of fp
    // selects the x-normal faces and bit 7 selects which of the two planes.
    const int iP = VMapP[fidx] - 1;
    const double fn1 = normal_fn[fidx];
    const double fn2 = normal_fn[fidx + nface];
    const double fn3 = normal_fn[fidx + 2 * nface];
    double qM, VelM;
#if P7_ABL_NOSTAGE
    {
      const int iM = VMapM[fidx] - 1;
      qM = q[iM];
      VelM = u[iM] * fn1 + v[iM] * fn2 + w[iM] * fn3;
    }
#else
    if (fp < 256 && (fp & 64) != 0) {
      const double *const m =
          sMface + (((fp & 128) != 0) ? XFACE_PLANE : 0) + (fp & 63);
      qM = m[0];
      VelM = m[144] * fn1 + m[288] * fn2 + m[432] * fn3;
    } else {
      const int iM = VMapM[fidx] - 1;
      qM = q[iM];
      VelM = u[iM] * fn1 + v[iM] * fn2 + w[iM] * fn3;
    }
#endif
    const double qP = q[iP];
    const double VelP = u[iP] * fn1 + v[iP] * fn2 + w[iP] * fn3;
    const double alpha = 0.5 * fabs(VelP + VelM);
    sflux_bnd[fp] =
        0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  }
  __syncthreads();

  const int row = lane >> 2;
  const int colk = lane & 3;
  const int j0_c = colk * 2;
  // The A operand of the z contraction and the D1D operand of the x and y
  // contractions are the same fragment element, D1D[row][colk].
  const int frag = (row << 2) + colk;
#if P7_TC_DREG
  // Section 14 held these two D1D fragment elements in registers and lost:
  // ptxas had to spill to stay inside the 32-register budget the KP = 1 form
  // needs for 8 blocks per SM.  The KP = 2 tile has 80 registers, and it also
  // issues the x/y mma pair KP times, so the reload count this saves is 2*KP,
  // not 2.  Section 23.9 measures it.
  const double dfrag0 = sDfrag[frag];
  const double dfrag1 = sDfrag[frag + 32];
#define P7_DF0 dfrag0
#define P7_DF1 dfrag1
#else
#define P7_DF0 sDfrag[frag]
#define P7_DF1 sDfrag[frag + 32]
#endif

  // Every paired shared access below derives the second address from the first
  // with one XOR by a constant, instead of swizzling a second index.  The
  // k0 = 4 operand differs from the k0 = 0 one in a single index bit that no
  // swizzle here reads, and the c1 accumulator element is one node away from
  // the c0 one, which sw_dz() leaves in place. Section 11 of
  // reports/tc_paper_survey_2407.09621.md records what this bought.

  // Dz = D * Fz_panel; warp owns 8 (i,j) columns, all k.  k0 = 4 sets bit 8 of
  // the node index, above the bits sw_z() folds.  The z panel is contracted
  // first because it is the only derivative that has to travel through shared
  // memory: its two barriers then sit before the x and y accumulators exist.
  // With KP > 1 a warp owns KP column groups; all KP results have to be in
  // registers before the barrier, because the store overwrites the flux the
  // mma read.
  double cz0[KP], cz1[KP];
#pragma unroll
  for (int cc = 0; cc < KP; ++cc) {
    const int wcol = warp + NWARP * cc;
    const int fz = sw_z(((wcol << 3) + row) + (colk << 6));
    mma_reset(cz0[cc], cz1[cc]);
    mma_k8_8x8<UseTc>(cz0[cc], cz1[cc], P7_DF0, sFluxZ[fz],
                      P7_DF1, sFluxZ[fz ^ 256]);
  }
  // sw_z() and sw_dz() permute across warp boundaries, so the z panel needs a
  // block-wide barrier before it overwrites the flux it was read from.
  __syncthreads();
#if (P7_ABL_SHST & 4)
  static_assert(KP == 1, "the shared-store ablation is written for KP == 1");
  sDz[tid] = cz0[0];
  sDz[tid + 256] = cz1[0];
#else
#pragma unroll
  for (int cc = 0; cc < KP; ++cc) {
    const int wcol = warp + NWARP * cc;
    const int dz_c = sw_dz(((wcol << 3) + j0_c) + (row << 6));
    sDz[dz_c] = cz0[cc];
    sDz[dz_c ^ 1] = cz1[cc];
  }
#endif
  __syncthreads();

  // The x and y derivatives never go through shared memory: the thread that
  // computes them is the thread that assembles them.  That removes four shared
  // stores and four shared loads per thread together with their address
  // arithmetic, and it leaves sFluxX and sFluxY read-only for the whole
  // kernel, so the two __syncwarp() calls that used to guard the in-place
  // overwrite are gone.
  //
  // Both contractions are evaluated transposed, C = (D*Fx)^T and C = (Fy*D^T)^T,
  // which costs nothing: with m8n8k4 the transpose is the same two operand
  // values passed in the opposite order.  It is what makes the epilogue
  // coalesce.  The accumulator holds C[lane>>2][2*(lane&3)] and its neighbour,
  // so the untransposed form gave thread lane the nodes
  //   (lane>>2) + 16*(lane&3) + 64*warp  and  + 8,
  // whose warp footprint is four 64-byte runs spread over 448 bytes: same
  // sectors as a contiguous access but twice the cache lines, and measurably
  // slower on a kernel that sits at 95% L1/TEX.  Transposed, the same thread
  // owns nodes 2*tid and 2*tid + 1, so a warp covers 64 consecutive nodes and
  // each of q's neighbours in the epilogue is one aligned 16-byte access.
  // KP k planes per warp: k = warp for the production form.
#pragma unroll
  for (int kk = 0; kk < KP; ++kk) {
  const int k = warp + NWARP * kk;
  const int n0 = (k << 6) + (lane << 1);
  const int nidx0 = elem_offset + n0;
  double c0, c1;

  // Dx^T = (D * Fx)^T on this k-plane.  sw_xy() flips bit 2 as a function of
  // bit 4, and k0 = 4 sets bit 2 of the node index, so the operands are fx and
  // fx^4.
  double acc0, acc1;
  {
    const int fx = sw_xy(colk + (row << 3) + (k << 6));
    const double2 es = *reinterpret_cast<const double2 *>(Escale + nidx0);
    mma_reset(c0, c1);
    mma_k8_8x8<UseTc>(c0, c1, sFluxX[fx], P7_DF0, sFluxX[fx ^ 4],
                      P7_DF1);
    acc0 = es.x * c0;
    acc1 = es.y * c1;
  }

  // Dy^T = (Fy * D^T)^T.  k0 = 4 sets bit 5 of the node index here, again a
  // bit sw_xy() does not read.
  {
    const int fy = sw_xy(row + (colk << 3) + (k << 6));
    const double2 es =
        *reinterpret_cast<const double2 *>(Escale + nidx0 + npoint);
    mma_reset(c0, c1);
    mma_k8_8x8<UseTc>(c0, c1, P7_DF0, sFluxY[fy], P7_DF1,
                      sFluxY[fy ^ 32]);
    acc0 += es.x * c0;
    acc1 += es.y * c1;
  }

  // sw_dz() folds bit 6 into bit 3 and n0 is even, so the second node of the
  // pair is the neighbour of the first in sDz as well.
  {
    const double2 dz = *reinterpret_cast<const double2 *>(sDz + sw_dz(n0));
    const double2 es =
        *reinterpret_cast<const double2 *>(Escale + nidx0 + 2 * npoint);
    acc0 += es.x * dz.x;
    acc1 += es.y * dz.y;
  }

  // The node pair differs in i only, so faces 2 and 4 (which vary in j) and
  // the lift coefficients that go with them are shared between the two, while
  // faces 1, 3, 5 and 6 shift by one face point.
  const int i0 = colk * 2;
  const int face1 = i0 + (k << 3);
  const int face2 = 64 + row + (k << 3);
  const int face5 = 256 + (n0 & 63);
  const double lf1 = sLift[row];
  const double lf3 = sLift[row + 16];
  const double lf5 = sLift[k + 32];
  const double lf6 = sLift[k + 40];
  const double fb2 = sflux_bnd[face2];
  const double fb4 = sflux_bnd[face2 + 128];
  const double lift0 = lf1 * sflux_bnd[face1] + sLift[i0 + 8] * fb2 +
                       lf3 * sflux_bnd[face1 + 128] + sLift[i0 + 24] * fb4 +
                       lf5 * sflux_bnd[face5] + lf6 * sflux_bnd[face5 + 64];
  const double lift1 = lf1 * sflux_bnd[face1 + 1] + sLift[i0 + 9] * fb2 +
                       lf3 * sflux_bnd[face1 + 129] + sLift[i0 + 25] * fb4 +
                       lf5 * sflux_bnd[face5 + 1] + lf6 * sflux_bnd[face5 + 65];

  *reinterpret_cast<double2 *>(dqdt + nidx0) =
      make_double2(-(acc0 + lift0), -(acc1 + lift1));
  }
#undef P7_DF0
#undef P7_DF1
}

// p=255 (Nq=256) tendency, one direction per launch.
//
// Every direction is written as the transposed product
//
//     C^T[m][n] = sum_l A[m][l] * B[n][l]
//
// which makes both operands the same shape -- [outer][l] with outer taken from
// lane/4 and l from lane%4 -- so one shared layout and one loader serve both.
// The transpose is free with m8n8k4 (it is the operand order) and it is what
// makes the epilogue coalesce: a lane owns two nodes adjacent in the fastest
// index, where the one-warp-per-block kernels this replaced owned two nodes
// 256 or 65536 apart.
//
//   x: C^T[j][i], A[j][l] = q*u at (l,j,k)   B[i][l] = D1D(i,l)
//   y: C^T[j][i], A[j][l] = D1D(j,l)         B[i][l] = q*v at (i,l,k)
//   z: C^T[k][p], A[k][l] = D1D(k,l)         B[p][l] = q*w at (p + Nq^2*l)
//
// where p is the linear (i,j) index, so z contracts the whole element at once
// and needs no plane loop.  Faces are split two per direction: x lifts faces 2
// and 4, y faces 1 and 3, z faces 5 and 6.
//
// The block owns a 64x64 output tile with 128 threads (four warps in a 2 by 2
// grid), each warp holding 4x4 mma tiles, and the chunk loop is double
// buffered.  How that shape was chosen is in reports/p255_gap_study.md; the
// two facts that decide it are that the mma loop pays (TM+TN) shared operand
// loads per TM*TN mma, so 4x4 is a third cheaper than the 2x4 it replaced, and
// that 4x4 costs 32 accumulator doubles, which only fits three blocks per SM.
// At that occupancy the un-pipelined loop lost more in exposed global latency
// than the cheaper mma loop won, so the two changes only pay together.

// Shared tiles are stored as l + BK*outer, l in the low log2(BK) bits and
// outer above them.  Three different access patterns hit these arrays: the mma read (l from
// lane%4, outer from lane/4), an outer-fast store (D1D and the y/z fluxes,
// whose global source runs fastest in the outer index) and an l-fast store
// (the x flux, whose source runs fastest in l).  Folding bits 4-5 into bits
// 2-3 fixes the read, and folding bits 6-7 into bits 0-1 fixes the outer-fast
// store; each fold is over bits the other pattern holds constant, so one
// function makes all three conflict free at once.
// LB = log2(BK) is where the outer index starts, so the two folds move with
// the chunk width: BK = 16 reproduces the shifts 4 and 6 this was written with.
template <int BK>
__device__ __forceinline__ int sw255(int idx)
{
  constexpr int LB = (BK == 16) ? 4 : ((BK == 32) ? 5 : 6);
  static_assert(BK == 16 || BK == 32 || BK == 64, "sw255 chunk width");
  return idx ^ (((idx >> LB) & 3) << 2) ^ (((idx >> (LB + 2)) & 3) << 0);
}

// Epilogue: scale the accumulators by Escale, add the two lifted faces this
// direction owns, and write (or accumulate onto) dqdt.
//
// The loop is nested b-outer / a-inner rather than flat over the TM*TN tiles,
// because half of what it loads does not depend on both indices.  For x the
// two face fluxes are indexed by m (the tile row) and the four lift
// coefficients by n (the tile column); for y and z it is the other way round.
// Written flat the compiler reloaded them for every tile -- 112 LDG for the
// sixteen output pairs of one warp at DIR=0.  Hoisting the m-side into
// registers and lifting the n-side out of the inner loop leaves only Escale,
// dqdt and the store inside, and the lift coefficients arrive as double2
// because the node pair a lane owns is adjacent in n.  That is 112 -> 32 LDG
// at DIR=0 and 96 -> 48 at DIR=1/2, worth 8.7% of the kernel; an ablation that
// strips the epilogue to a bare store prices what is left at 9.7%, which is
// the Escale and dqdt traffic the numerical contract requires.
// Illegal ablations that price an x+y fusion without paying its register bill.
// Fusing the two directions would remove exactly two things: x's 16 MB store
// into dqdt and y's 16 MB read back out of it.  The field these leave behind
// is wrong; they are ceiling measurements only.
//
//   1: x's store predicated off and y's load replaced by zero.  Loose: with
//      the store under a never-taken branch ptxas sinks the whole x epilogue
//      into it (168 -> 110 registers), so it also deletes work a fusion keeps.
//   2: x's store folded into one live-but-never-taken scalar store, which
//      keeps every Escale / lift / face load and every flop.  Tight.
//   3: y's dqdt read replaced by zero.  Tight on its own.
//   4: 2 and 3 together -- the fusion's whole prize.
#ifndef P255_XYFUSE_ABL
#define P255_XYFUSE_ABL 0
#endif

template <int DIR, int NQ, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void p255_epilogue(
    double *dqdt, const double *Lift1D, const double *flux_bnd,
    const double *Escale, const double *acc, const double *face24,
    const double *es_pre, int m0, int n0, int wm, int wn, int row, int colk,
    int eo, int efo, int npoint, int plane_off, int kplane)
{
  const int NQ2 = NQ * NQ;

  // Per-row (m) quantities: the two face flux values for x, the two lift
  // coefficients for y and z.
  double ra[TM], rb2[TM];
#pragma unroll
  for (int a = 0; a < TM; ++a) {
    const int m = m0 + 8 * (TM * wm + a) + row;
    if (DIR == 0) {
      ra[a] = face24[m - m0];
      rb2[a] = face24[64 + (m - m0)];
    } else if (DIR == 1) {
      ra[a] = Lift1D[m];
      rb2[a] = Lift1D[m + 2 * NQ];
    } else {
      ra[a] = Lift1D[m + 4 * NQ];
      rb2[a] = Lift1D[m + 5 * NQ];
    }
  }

  double xsink = 0.0;
  (void)xsink;
  double2 dout[TM * TN];
  double2 esy[TM * TN];
  if (DIR != 0) {
#pragma unroll
    for (int bb = 0; bb < TN; ++bb) {
      const int n = n0 + 8 * (TN * wn + bb) + 2 * colk;
#pragma unroll
      for (int a = 0; a < TM; ++a) {
        const int m = m0 + 8 * (TM * wm + a) + row;
        const int node =
            (DIR == 2) ? (eo + n + NQ2 * m) : (eo + n + NQ * m + plane_off);
#if P255_XYFUSE_ABL == 1 || P255_XYFUSE_ABL == 3 || P255_XYFUSE_ABL == 4
        dout[TN * a + bb] =
            (DIR == 1) ? make_double2(0.0, 0.0)
                       : *reinterpret_cast<const double2 *>(dqdt + node);
#else
        dout[TN * a + bb] = *reinterpret_cast<const double2 *>(dqdt + node);
#endif
        esy[TN * a + bb] = *reinterpret_cast<const double2 *>(
            Escale + node + (DIR == 1 ? npoint : 2 * npoint));
      }
    }
  }

#pragma unroll
  for (int bb = 0; bb < TN; ++bb) {
    const int n = n0 + 8 * (TN * wn + bb) + 2 * colk;
    // Per-column (n) quantities: the lift coefficient pairs for x, the face
    // flux pairs for y and z.  Both are adjacent in n, hence double2.
    double2 c0n, c1n;
    if (DIR == 0) {
      c0n = *reinterpret_cast<const double2 *>(Lift1D + n + NQ);
      c1n = *reinterpret_cast<const double2 *>(Lift1D + n + 3 * NQ);
    } else if (DIR == 1) {
      const int fp = n + NQ * kplane;
      c0n = *reinterpret_cast<const double2 *>(flux_bnd + efo + fp);
      c1n = *reinterpret_cast<const double2 *>(flux_bnd + efo + 2 * NQ2 + fp);
    } else {
      c0n = *reinterpret_cast<const double2 *>(flux_bnd + efo + 4 * NQ2 + n);
      c1n = *reinterpret_cast<const double2 *>(flux_bnd + efo + 5 * NQ2 + n);
    }
#pragma unroll
    for (int a = 0; a < TM; ++a) {
      const int e8 = TN * a + bb;
      const int m = m0 + 8 * (TM * wm + a) + row;
      const double c0 = acc[2 * e8];
      const double c1 = acc[2 * e8 + 1];
      const int node =
          (DIR == 2) ? (eo + n + NQ2 * m) : (eo + n + NQ * m + plane_off);
      if (DIR == 0) {
        // Faces 2 and 4 vary in (j,k), so the node pair shares the flux value
        // and differs only through the Lift1D coefficient, which varies in i.
        const double2 es = make_double2(es_pre[2 * e8], es_pre[2 * e8 + 1]);
        const double l0 = c0n.x * ra[a] + c1n.x * rb2[a];
        const double l1 = c0n.y * ra[a] + c1n.y * rb2[a];
#if P255_XYFUSE_ABL == 1
        if (npoint < 0)
          *reinterpret_cast<double2 *>(dqdt + node) =
              make_double2(-(es.x * c0 + l0), -(es.y * c1 + l1));
#elif P255_XYFUSE_ABL == 2 || P255_XYFUSE_ABL == 4
        xsink += -(es.x * c0 + l0) - (es.y * c1 + l1);
#else
        *reinterpret_cast<double2 *>(dqdt + node) =
            make_double2(-(es.x * c0 + l0), -(es.y * c1 + l1));
#endif
      } else {
        // Faces 1 and 3 vary in (i,k) and faces 5 and 6 in the linear (i,j)
        // point: here the pair shares the coefficient and the two flux values
        // are one aligned double2.
        const double2 es = esy[e8];
        double2 out = dout[e8];
        out.x -= es.x * c0 + ra[a] * c0n.x + rb2[a] * c1n.x;
        out.y -= es.y * c1 + ra[a] * c0n.y + rb2[a] * c1n.y;
        *reinterpret_cast<double2 *>(dqdt + node) = out;
      }
    }
  }
#if P255_XYFUSE_ABL == 2 || P255_XYFUSE_ABL == 4
  if (DIR == 0 && xsink == 1.2345e300) {
    dqdt[eo] = xsink;
  }
#endif
}

// DIR: 0 = x, 1 = y, 2 = z.
//
// The chunk loop is double buffered:
//
//   issue(k+1) -> mma(buf) -> store(buf^1) -> barrier
//
// One barrier per chunk instead of two, and the next chunk's global loads are
// in flight across the whole mma loop.  The loaded values stay raw in
// registers and the q*vel multiply happens at the store, because multiplying
// at issue time would make the pipeline wait on the loads exactly where it is
// trying not to.  Against the single-buffered loop this is worth 27.8% here,
// and it is what makes the 4x4 warp shape usable at all.
//
// Measured and rejected (reports/p255_gap_study.md):
//   - moving the barrier to the head of the body, which is what won at p=63
//     and p=127: +10.7% here, because with sixteen chunks the barrier saved is
//     one of many while the guard is a branch in every iteration;
//   - hoisting the staging addresses out of the chunk loop (the shared
//     destinations do not move and the global sources are affine in the chunk
//     index): -34% instructions in the loop body, +2.5% time;
//   - bigger tiles (64x128, 128x64, 128x128) in every thread-count and
//     launch-bound combination that fits: 10-35% slower, all of them through
//     registers and occupancy, never through the operand traffic they save.
template <int DIR, int NQ, bool UseTc, bool FuseFace24>
__global__ __launch_bounds__(TH255, MINB255) void tendency_p255_kernel(
    double *__restrict__ dqdt, const double *__restrict__ q,
    const double *__restrict__ velocity, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ flux_bnd,
    const double *__restrict__ Escale, const double *__restrict__ vel_v,
    const double *__restrict__ vel_w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, int Ne)
{
  constexpr int BM = BM255;
  constexpr int BN = BN255;
  constexpr int BK = BK255;
  constexpr int TM = TM255;
  constexpr int TN = TN255;
  constexpr int THREADS = TH255;
  constexpr int WM = BM / (8 * TM);
  constexpr int WN = BN / (8 * TN);
  // Staging iterations per thread.  Each one moves a double2, so the pair a
  // thread holds is adjacent in whichever index runs fastest in global and the
  // wavefront stays fully coalesced with half the addresses formed.
  constexpr int NA = (BM * BK) / THREADS;
  constexpr int NB = (BN * BK) / THREADS;
  static_assert(THREADS == 32 * WM * WN, "warp grid must tile the block");
  static_assert(NA % 2 == 0 && NB % 2 == 0, "staging is vectorized");

#if P255_DYNSMEM
  // BK > 16 needs 2*(BM+BN)*BK doubles, which is past the 48 KB static limit.
  extern __shared__ __align__(16) double p255_smem[];
  double(*sA)[BM * BK] = reinterpret_cast<double(*)[BM * BK]>(p255_smem);
  double(*sB)[BN * BK] =
      reinterpret_cast<double(*)[BN * BK]>(p255_smem + 2 * BM * BK);
#else
  __shared__ __align__(16) double sA[2][BM * BK];
  __shared__ __align__(16) double sB[2][BN * BK];
#endif

  const int NP = NQ * NQ * NQ;
  const int NQ2 = NQ * NQ;
  // x and y: (Nq/BM) m-tiles * (Nq/BN) n-tiles * Nq planes.
  // z:       (Nq/BM) m-tiles * (Nq^2/BN) n-tiles, no plane loop.  Both are
  // Nq^3/(BM*BN), so the grid is the same shape for all three.
  const int blocks_per_elem = (NQ / BM) * (NQ2 / BN);

  const int elem = (int)blockIdx.x / blocks_per_elem;
  if (elem >= Ne) {
    return;
  }
  const int b = (int)blockIdx.x - elem * blocks_per_elem;
  constexpr int MTILES = NQ / BM;
  constexpr int NTILES = (DIR == 2) ? (NQ * NQ / BN) : (NQ / BN);
  const int tm = b % MTILES;
  const int tn = (b / MTILES) % NTILES;
  const int kplane = (DIR == 2) ? 0 : (b / MTILES) / NTILES;
  const int m0 = tm * BM;
  const int n0 = tn * BN;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP;
  const int efo = elem * 6 * NQ2;
  const int npoint = NP * Ne;
  const int plane_off = kplane * NQ2;

  const int wm = warp % WM;
  const int wn = warp / WM;
  // Which half of the vectorized pair a lane stores first.  The two elements a
  // lane holds in an outer-fast panel are o and o+1, and c(o+1) = c(o) ^ 4 in
  // the swizzle, so if every lane stored its even element first the sixteen
  // lanes of a half warp would reach only eight banks -- the 2-way store
  // conflict ncu found once the staging was vectorized (8.45 M conflicts on
  // 16.8 M store wavefronts).  Letting the upper eight lanes of each half warp
  // store their odd element first makes both store instructions cover all
  // sixteen banks.  Worth 0.9%.
  const bool pswap = (lane & 8) != 0;

  double acc[2 * TM * TN];
#pragma unroll
  for (int i = 0; i < 2 * TM * TN; ++i) {
    acc[i] = 0.0;
  }

  // Prefetch registers.  The panel that is a flux needs q and the velocity
  // held separately until the store; the panel that is D1D needs one value.
  double raq[NA], rav[NA], rb[NB], rbv[NB];

  // Hoisted shared addresses for the mma loop.  sw255 only ever touches the
  // low four bits of the index, so with l = 4*ks + colk and outer = 8*t + row
  //
  //   sw255(l + BK*outer) = BK*outer + (colk ^ c) + ((4*ks) ^ ((row & 3) << 2))
  //
  // where c = (2*t + (row >> 2)) & 3: the three fields land in disjoint bits,
  // colk in 0-1, 4*ks in 2-3, 16*outer above.  Both terms are loop invariant.
  // Worth 1.7% -- but only once the loop was pipelined; on the single-buffered
  // loop the same change cost 4.6%, which is the clearest evidence here that
  // this kernel is not bound by instruction issue.
  int abase[TM], bbase[TN], koff[BK / 4];
#pragma unroll
  for (int a = 0; a < TM; ++a) {
    const int t = TM * wm + a;
    abase[a] = BK * (8 * t + row) + (colk ^ ((2 * t + (row >> 2)) & 3));
  }
#pragma unroll
  for (int bb = 0; bb < TN; ++bb) {
    const int t = TN * wn + bb;
    bbase[bb] = BK * (8 * t + row) + (colk ^ ((2 * t + (row >> 2)) & 3));
  }
#pragma unroll
  for (int ks = 0; ks < BK / 4; ++ks) {
    koff[ks] = (4 * ks) ^ ((row & 3) << 2);
  }

#define P255_ISSUE(KK)                                                        \
  do {                                                                        \
    _Pragma("unroll") for (int p = 0; p < NA / 2; ++p)                        \
    {                                                                         \
      const int pr = tid + THREADS * p;                                       \
      if (DIR == 0) {                                                         \
        const int ll = 2 * (pr % (BK / 2));                                   \
        const int o = pr / (BK / 2);                                          \
        const int g = eo + ((KK) + ll) + NQ * (m0 + o) + plane_off;           \
        const double2 vq = *reinterpret_cast<const double2 *>(q + g);         \
        const double2 vv = *reinterpret_cast<const double2 *>(velocity + g);  \
        raq[2 * p] = vq.x;                                                    \
        raq[2 * p + 1] = vq.y;                                                \
        rav[2 * p] = vv.x;                                                    \
        rav[2 * p + 1] = vv.y;                                                \
      } else {                                                                \
        const int o = 2 * (pr % (BM / 2));                                    \
        const int ll = pr / (BM / 2);                                         \
        const double2 vd = *reinterpret_cast<const double2 *>(                \
            D1D + (m0 + o) + NQ * ((KK) + ll));                               \
        raq[2 * p] = vd.x;                                                    \
        raq[2 * p + 1] = vd.y;                                                \
      }                                                                       \
    }                                                                         \
    _Pragma("unroll") for (int p = 0; p < NB / 2; ++p)                        \
    {                                                                         \
      const int pr = tid + THREADS * p;                                       \
      const int o = 2 * (pr % (BN / 2));                                      \
      const int ll = pr / (BN / 2);                                           \
      if (DIR == 0) {                                                         \
        const double2 vd = *reinterpret_cast<const double2 *>(                \
            D1D + (n0 + o) + NQ * ((KK) + ll));                               \
        rb[2 * p] = vd.x;                                                     \
        rb[2 * p + 1] = vd.y;                                                 \
      } else {                                                                \
        const int g = (DIR == 1)                                              \
                          ? eo + (n0 + o) + NQ * ((KK) + ll) + plane_off      \
                          : eo + (n0 + o) + NQ2 * ((KK) + ll);                \
        const double2 vq = *reinterpret_cast<const double2 *>(q + g);         \
        const double2 vv = *reinterpret_cast<const double2 *>(velocity + g);  \
        rb[2 * p] = vq.x;                                                     \
        rb[2 * p + 1] = vq.y;                                                 \
        rbv[2 * p] = vv.x;                                                    \
        rbv[2 * p + 1] = vv.y;                                                \
      }                                                                       \
    }                                                                         \
  } while (0)

#define P255_STORE(BUF)                                                       \
  do {                                                                        \
    _Pragma("unroll") for (int p = 0; p < NA / 2; ++p)                        \
    {                                                                         \
      const int pr = tid + THREADS * p;                                       \
      if (DIR == 0) {                                                         \
        /* An adjacent l pair stays adjacent in shared under sw255, so it      \
           goes out as one 16-byte store; the xor may swap the two halves. */ \
        const int ll = 2 * (pr % (BK / 2));                                   \
        const int o = pr / (BK / 2);                                          \
        const int i0 = sw255<BK>(ll + BK * o);                                    \
        const double v0 = raq[2 * p] * rav[2 * p];                            \
        const double v1 = raq[2 * p + 1] * rav[2 * p + 1];                    \
        *reinterpret_cast<double2 *>(&sA[BUF][i0 & ~1]) =                     \
            (i0 & 1) ? make_double2(v1, v0) : make_double2(v0, v1);           \
      } else {                                                                \
        const int o = 2 * (pr % (BM / 2));                                    \
        const int ll = pr / (BM / 2);                                         \
        const int i0 = sw255<BK>(ll + BK * o);                                    \
        const int i1 = sw255<BK>(ll + BK * (o + 1));                              \
        sA[BUF][pswap ? i1 : i0] = pswap ? raq[2 * p + 1] : raq[2 * p];       \
        sA[BUF][pswap ? i0 : i1] = pswap ? raq[2 * p] : raq[2 * p + 1];       \
      }                                                                       \
    }                                                                         \
    _Pragma("unroll") for (int p = 0; p < NB / 2; ++p)                        \
    {                                                                         \
      const int pr = tid + THREADS * p;                                       \
      const int o = 2 * (pr % (BN / 2));                                      \
      const int ll = pr / (BN / 2);                                           \
      const double w0 = (DIR == 0) ? rb[2 * p] : rb[2 * p] * rbv[2 * p];      \
      const double w1 =                                                       \
          (DIR == 0) ? rb[2 * p + 1] : rb[2 * p + 1] * rbv[2 * p + 1];        \
      const int j0 = sw255<BK>(ll + BK * o);                                      \
      const int j1 = sw255<BK>(ll + BK * (o + 1));                                \
      sB[BUF][pswap ? j1 : j0] = pswap ? w1 : w0;                             \
      sB[BUF][pswap ? j0 : j1] = pswap ? w0 : w1;                             \
    }                                                                         \
  } while (0)

  P255_ISSUE(0);
  P255_STORE(0);
  __syncthreads();

  int cur = 0;
  for (int kk = 0; kk < NQ; kk += BK) {
    const bool more = (kk + BK) < NQ;
    if (more) {
      P255_ISSUE(kk + BK);
    }
#pragma unroll
    for (int ks = 0; ks < BK / 4; ++ks) {
      double av[TM], bv[TN];
#pragma unroll
      for (int a = 0; a < TM; ++a) {
        av[a] = sA[cur][abase[a] + koff[ks]];
      }
#pragma unroll
      for (int bb = 0; bb < TN; ++bb) {
        bv[bb] = sB[cur][bbase[bb] + koff[ks]];
      }
#pragma unroll
      for (int a = 0; a < TM; ++a) {
#pragma unroll
        for (int bb = 0; bb < TN; ++bb) {
          const int e = 2 * (TN * a + bb);
          mma_m8n8k4_f64<UseTc>(acc[e], acc[e + 1], av[a], bv[bb], acc[e],
                         acc[e + 1]);
        }
      }
    }
    if (more) {
      P255_STORE(cur ^ 1);
      __syncthreads();
      cur ^= 1;
    }
  }
#undef P255_ISSUE
#undef P255_STORE

  double esx[2 * TM * TN];
  const double *face24 = nullptr;
  if (DIR == 0) {
    const int j = m0 + (tid & 63);
    const int fp = j + NQ * kplane;
#pragma unroll
    for (int bb = 0; bb < TN; ++bb) {
      const int n = n0 + 8 * (TN * wn + bb) + 2 * colk;
#pragma unroll
      for (int a = 0; a < TM; ++a) {
        const int e8 = TN * a + bb;
        const int m = m0 + 8 * (TM * wm + a) + row;
        const int node = eo + n + NQ * m + plane_off;
        const double2 es = *reinterpret_cast<const double2 *>(Escale + node);
        esx[2 * e8] = es.x;
        esx[2 * e8 + 1] = es.y;
      }
    }
    const int fidx = efo + ((tid < 64) ? NQ2 : 3 * NQ2) + fp;
    if constexpr (FuseFace24) {
      const int nface = 6 * NQ2 * Ne;
      const int iP = VMapP[fidx] - 1;
      const double fn1 = normal_fn[fidx];
      const double fn2 = normal_fn[fidx + nface];
      const double fn3 = normal_fn[fidx + 2 * nface];
      const double qP = q[iP];
      const double VelP =
          velocity[iP] * fn1 + vel_v[iP] * fn2 + vel_w[iP] * fn3;
      const int iM = eo + ((tid < 64) ? (NQ - 1) : 0) + NQ * j + plane_off;
      const double qM = q[iM];
      const double VelM =
          velocity[iM] * fn1 + vel_v[iM] * fn2 + vel_w[iM] * fn3;
      const double alpha = 0.5 * fabs(VelP + VelM);
      sA[0][tid] =
          0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
    } else {
      sA[0][tid] = flux_bnd[fidx];
    }
    face24 = sA[0];
    __syncthreads();
  }

  p255_epilogue<DIR, NQ, BM, BN, TM, TN>(dqdt, Lift1D, flux_bnd, Escale, acc, face24,
                                     esx, m0, n0, wm, wn, row, colk, eo, efo,
                                     npoint, plane_off, kplane);
}

static void check_cuda(const char *what)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
  }
}

//- Stream that every kernel of the CUDA path is launched on.  The Fortran
//  side sets it to the stream of the OpenACC queue used by the time-stepping
//  loop, so that the two kinds of kernels keep their order without the host
//  synchronizing in between.  Zero (the default stream) until it is set.
cudaStream_t dg_cuda_stream = 0;

extern "C" void dg_set_cuda_stream(void *stream)
{
  dg_cuda_stream = static_cast<cudaStream_t>(stream);
}

template <bool UseTc>
void launch_tendency_fused_p7_impl(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
#if P7_TC_KP > 1
  tendency_fused_p7_tile_kernel<UseTc><<<Ne, P7_THREADS, 0, dg_cuda_stream>>>(
#else
  tendency_fused_p7_kernel<UseTc><<<Ne, P7_THREADS, 0, dg_cuda_stream>>>(
#endif
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda("tendency_fused_p7_kernel");
}

extern "C" void launch_tendency_fused_p7_dfma(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  launch_tendency_fused_p7_impl<false>(dqdt, D1D, Lift1D, q, u, v, w, VMapM,
                                       VMapP, normal_fn, Fscale, Escale, Ne);
}

extern "C" void launch_tendency_fused_p7_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  launch_tendency_fused_p7_impl<true>(dqdt, D1D, Lift1D, q, u, v, w, VMapM,
                                      VMapP, normal_fn, Fscale, Escale, Ne);
}

#if P255_DYNSMEM
#define P255_DYN_BYTES ((int)P255_SMEM_BYTES)
#else
#define P255_DYN_BYTES 0
#endif

// BK255 > 16 puts the two double-buffered panels past the 48 KB static limit,
// so they move to dynamic shared memory and the opt-in has to be requested
// once per instantiation.
template <int DIR, int NQ, bool UseTc, bool FuseFace24>
static void p255_set_smem()
{
#if P255_DYNSMEM
  static bool done = false;
  if (!done) {
    cudaFuncSetAttribute(tendency_p255_kernel<DIR, NQ, UseTc, FuseFace24>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         P255_DYN_BYTES);
    done = true;
  }
#endif
}

template <bool UseTc>
void launch_tendency_xyz_p255_impl(
    double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, int Ne)
{
  const int nblock = (NQ255 * NQ255 * NQ255 / (BM255 * BN255)) * Ne;
  p255_set_smem<0, NQ255, UseTc, false>();
  p255_set_smem<1, NQ255, UseTc, false>();
  p255_set_smem<2, NQ255, UseTc, false>();
  tendency_p255_kernel<0, NQ255, UseTc, false><<<nblock, TH255, P255_DYN_BYTES, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, v, w, nullptr, nullptr, nullptr,
      nullptr, Ne);
  tendency_p255_kernel<1, NQ255, UseTc, false><<<nblock, TH255, P255_DYN_BYTES, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, Ne);
  tendency_p255_kernel<2, NQ255, UseTc, false><<<nblock, TH255, P255_DYN_BYTES, dg_cuda_stream>>>(
      dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, Ne);
  check_cuda("p255 fused tendency kernels");
}

extern "C" void launch_tendency_xyz_p255_dfma(
    double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, int Ne)
{
  launch_tendency_xyz_p255_impl<false>(dqdt, q, u, v, w, D1D, Lift1D, flux_bnd,
                                       Escale, Ne);
}

extern "C" void launch_tendency_xyz_p255_tc(
    double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, int Ne)
{
  launch_tendency_xyz_p255_impl<true>(dqdt, q, u, v, w, D1D, Lift1D, flux_bnd,
                                      Escale, Ne);
}

// The tile-type kernel above carries Nq only in its grid and in the number of
// reduction chunks: shared (32,768 B), registers (168, no spill), the 128
// threads and the three blocks per SM are all Nq-independent.  That is what
// lets the same source serve Nq = 256 and Nq = 512; the counting is in
// reports/p511_gap_study.md section 14.3, and section 15 measures what it buys.
// Nq = 512 still fits 32-bit node indices (Escale reaches 3*Np = 4.03e8);
// only Nq = 1024 would need the 64-bit offsets p1023_gap_study.md section 1
// gave the GEMM path.
template <bool UseTc, int NQ>
void launch_tendency_dir_p255_nq(
    int dir, double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale, int Ne)
{
  const int nblock = (NQ / BM255) * (NQ * NQ / BN255) * Ne;
  p255_set_smem<0, NQ, UseTc, true>();
  p255_set_smem<1, NQ, UseTc, false>();
  p255_set_smem<2, NQ, UseTc, false>();
  if (dir == 0) {
    tendency_p255_kernel<0, NQ, UseTc, true><<<nblock, TH255, P255_DYN_BYTES, dg_cuda_stream>>>(
        dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, v, w, VMapM, VMapP, normal_fn,
        Fscale, Ne);
  } else if (dir == 1) {
    tendency_p255_kernel<1, NQ, UseTc, false><<<nblock, TH255, P255_DYN_BYTES, dg_cuda_stream>>>(
        dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, v, w, VMapM, VMapP, normal_fn,
        Fscale, Ne);
  } else {
    tendency_p255_kernel<2, NQ, UseTc, false><<<nblock, TH255, P255_DYN_BYTES, dg_cuda_stream>>>(
        dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, v, w, VMapM, VMapP, normal_fn,
        Fscale, Ne);
  }
  check_cuda("tile fused tendency dir kernel");
}

template <bool UseTc>
void launch_tendency_dir_p255_impl(
    int dir, double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale, int Ne,
    int nq)
{
  if (nq == 512) {
    launch_tendency_dir_p255_nq<UseTc, 512>(dir, dqdt, q, u, v, w, D1D, Lift1D,
                                            flux_bnd, Escale, VMapM, VMapP,
                                            normal_fn, Fscale, Ne);
  } else {
    launch_tendency_dir_p255_nq<UseTc, NQ255>(dir, dqdt, q, u, v, w, D1D,
                                              Lift1D, flux_bnd, Escale, VMapM,
                                              VMapP, normal_fn, Fscale, Ne);
  }
}

extern "C" void launch_tendency_dir_p255_dfma(
    int dir, double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale, int Ne,
    int nq)
{
  launch_tendency_dir_p255_impl<false>(dir, dqdt, q, u, v, w, D1D, Lift1D,
                                       flux_bnd, Escale, VMapM, VMapP, normal_fn,
                                       Fscale, Ne, nq);
}

extern "C" void launch_tendency_dir_p255_tc(
    int dir, double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale, int Ne,
    int nq)
{
  launch_tendency_dir_p255_impl<true>(dir, dqdt, q, u, v, w, D1D, Lift1D,
                                      flux_bnd, Escale, VMapM, VMapP, normal_fn,
                                      Fscale, Ne, nq);
}

//============================================================================
// p=15 (Nq=16) fused Tensor Core tendency
//============================================================================
//
// Shared memory strategy is the one the CUDA-core p=15 kernel established:
// laying out three directional flux panels the way the p=7 kernel does would
// need 3*4096*8 = 96 KB and push the block past the 48 KB static limit into a
// carveout that costs more L1 than it buys (section 13.4 of
// tc_paper_survey_2407.09621.md).  One 4096-double buffer is reused for the x,
// y and z panels in turn and then for the face fluxes, and q is held in
// registers so every field is still read from global exactly once.
//
// The m8n8k4 tile is 8x8 while a plane here is 16x16, so one plane needs four
// output tiles and four k-steps instead of one tile and two steps.  A warp
// owns one j-half of one plane (both i-halves), so each lane still ends up
// with four nodes: i = tn*8 + 2*colk and +1 for tn = 0, 1.
//
// The same fragment array serves all three directions.  x reads it as the B
// operand D[i][l], y as the A operand D[j_out][j_in] and z as the A operand
// D[k_out][l]; in every case a lane wants D[tile*8 + row][colk + 4*ks], so one
// layout indexed by (tile, ks, row, colk) covers them all.

// x reads the panel at (i = colk + 4*ks, j = tm*8 + row): colk lands in bits
// 0-1 and row in bits 4-5, leaving bits 2-3 dead for a 4-way conflict.  y
// reads at (i = tn*8 + row, j = colk + 4*ks), which is the same picture with
// the roles swapped, so folding node bits 4-5 into address bits 2-3 fixes
// both with one function -- the same trick sw_xy() plays at Nq=8, one bit
// wider.  Neither the x k-step offset (bits 2-3) nor the y one (bits 6-7) is
// read here, so both stay a plain XOR on the swizzled address.
__device__ __forceinline__ int sw_xy15(int idx)
{
  return idx ^ (((idx >> 4) & 3) << 2);
}

// z strides by 256 doubles, so its contraction index sits in bits 8-9.
__device__ __forceinline__ int sw_z15(int idx)
{
  return idx ^ (((idx >> 8) & 3) << 2);
}

// The z derivative is written under the mma output map and read back under
// the x/y one, and the two disagree about where the varying bits live: the
// store has 2*colk in bits 1-2 and the output k in bits 8-10, while the read
// has 2*colk in bits 1-2 and j in bits 4-6.  Bits 0 and 3 are dead in both,
// so folding the store's bits 8-9 and the read's bits 4-5 into them makes
// both phases conflict free at once.  Each phase only ever sees the other's
// source bits as warp-invariant, so the extra terms are a constant XOR there
// and do no harm.  Bit 0 is only ever XORed with warp-invariant bits, which
// keeps the c0/c1 pair adjacent so the pair still moves as one double2.
//
// Unlike Nq=8, where section 10.3 of the survey found the fully conflict-free
// permutation slower, this one is not a trade: at Nq=16 the read was 4-way,
// and ncu (job 55570) put 3.21 M of the kernel's shared-load conflicts here.
__device__ __forceinline__ int sw_dz15(int idx)
{
  return idx ^ (((idx >> 8) & 1) << 3) ^ (((idx >> 9) & 1) << 0) ^
         (((idx >> 4) & 1) << 0) ^ (((idx >> 5) & 1) << 3);
}

// Face-flux staging.  The six faces are read back with three different index
// shapes: faces 1 and 3 vary in (i,k), faces 2 and 4 in (j,k) and faces 5 and
// 6 in (i,j).  Within a warp k is constant, so the first two shapes broadcast,
// but the (i,j) one has 2*colk in bits 1-2 and j in bits 4-7 with bits 0 and 3
// dead, which is a 4-way conflict on eight of the epilogue's loads.  Folding
// the j bits into the dead ones fixes faces 5 and 6 and leaves the other four
// broadcasting, because there those bits are warp-invariant.  The write side
// is fp = tid, i.e. sixteen consecutive indices per half warp, and a fold of
// bits that are constant across them is still a permutation of the sixteen.
__device__ __forceinline__ int sw_f15(int fp)
{
  return fp ^ (((fp >> 4) & 1) << 0) ^ (((fp >> 5) & 1) << 3);
}

// How many of the element's six boundary planes of q, u, v and w are staged
// in shared memory for the M side of the face flux.  2 is the production form
// of section 16.6 of p15_gap_study.md (the two i-normal planes, 16 KB); 6
// stages all of them (48 KB) and is the candidate section 16.6 rejected
// without writing.  Section 30 of the same report measures it.
#ifndef P15_MPLANES
#define P15_MPLANES 2
#endif
#if P15_MPLANES == 6
#define P15_MSTRIDE 1536
#else
#define P15_MSTRIDE 512
#endif

// Illegal ablation: read the M side at the coalesced node index instead of
// the one VMapM gave.  Measures the ceiling of every M-side gather at once
// (section 16.9's "M side fully coalesced").  Numerically wrong.
#ifndef P15_ABL_MCOAL
#define P15_ABL_MCOAL 0
#endif

// Read q, u, v and w at an M-side face node.  The staged boundary planes of
// the element are in shared memory; every other node still goes to global.
// The index is the one VMapM gave, so the test is on the map's own answer.
#if P15_MPLANES == 6
#define LOAD_M(iM, qv, uv, vv, wv)                                             \
  {                                                                            \
    const int loc = (iM) - elem_offset;                                        \
    int sidx = -1;                                                             \
    if ((unsigned)loc < (unsigned)NP15) {                                      \
      const int ii = loc & 15;                                                 \
      const int jj = (loc >> 4) & 15;                                          \
      const int kk = loc >> 8;                                                 \
      if (ii == 0) sidx = jj + 16 * kk;                                        \
      else if (ii == 15) sidx = 256 + jj + 16 * kk;                            \
      else if (jj == 0) sidx = 512 + ii + 16 * kk;                             \
      else if (jj == 15) sidx = 768 + ii + 16 * kk;                            \
      else if (kk == 0) sidx = 1024 + ii + 16 * jj;                            \
      else if (kk == 15) sidx = 1280 + ii + 16 * jj;                           \
    }                                                                          \
    if (sidx >= 0) {                                                           \
      qv = sMq[sidx]; uv = sMu[sidx]; vv = sMv[sidx]; wv = sMw[sidx];          \
    } else {                                                                   \
      qv = q[iM]; uv = u[iM]; vv = v[iM]; wv = w[iM];                          \
    }                                                                          \
  }
#else
#define LOAD_M(iM, qv, uv, vv, wv)                                             \
  {                                                                            \
    const int loc = (iM) - elem_offset;                                        \
    const int im = loc & 15;                                                   \
    if ((unsigned)loc < (unsigned)NP15 && (im == 0 || im == 15)) {             \
      const int sidx = ((im >> 3) << 8) + ((loc >> 4) & 255);                  \
      qv = sMq[sidx]; uv = sMu[sidx]; vv = sMv[sidx]; wv = sMw[sidx];          \
    } else {                                                                   \
      qv = q[iM]; uv = u[iM]; vv = v[iM]; wv = w[iM];                          \
    }                                                                          \
  }
#endif

// Write the boundary-plane copies on the way past.  The volume phase has
// already read these nodes; nothing is assumed about VMapM.  The pair (n,n+1)
// is adjacent in i, so the j- and k-normal planes take it as one double2 and
// only the i-normal planes split it.
#define STAGE_M_PAIR(n, QV, UV, VV, WV)                                        \
  {                                                                            \
    const int ii = (n) & 15;                                                   \
    const int jj = ((n) >> 4) & 15;                                            \
    const int kk = (n) >> 8;                                                   \
    if (ii == 0) {                                                             \
      const int s = jj + 16 * kk;                                              \
      sMq[s] = QV.x; sMu[s] = UV.x; sMv[s] = VV.x; sMw[s] = WV.x;              \
    } else if (ii == 14) {                                                     \
      const int s = 256 + jj + 16 * kk;                                        \
      sMq[s] = QV.y; sMu[s] = UV.y; sMv[s] = VV.y; sMw[s] = WV.y;              \
    }                                                                          \
    int sj = -1;                                                               \
    if (jj == 0) sj = 512 + ii + 16 * kk;                                      \
    else if (jj == 15) sj = 768 + ii + 16 * kk;                                \
    if (sj >= 0) {                                                             \
      *reinterpret_cast<double2 *>(sMq + sj) = QV;                             \
      *reinterpret_cast<double2 *>(sMu + sj) = UV;                             \
      *reinterpret_cast<double2 *>(sMv + sj) = VV;                             \
      *reinterpret_cast<double2 *>(sMw + sj) = WV;                             \
    }                                                                          \
    int sk = -1;                                                               \
    if (kk == 0) sk = 1024 + ii + 16 * jj;                                     \
    else if (kk == 15) sk = 1280 + ii + 16 * jj;                               \
    if (sk >= 0) {                                                             \
      *reinterpret_cast<double2 *>(sMq + sk) = QV;                             \
      *reinterpret_cast<double2 *>(sMu + sk) = UV;                             \
      *reinterpret_cast<double2 *>(sMv + sk) = VV;                             \
      *reinterpret_cast<double2 *>(sMw + sk) = WV;                             \
    }                                                                          \
  }

template <bool UseTc>
__global__ __launch_bounds__(P15_THREADS, 1) void tendency_fused_p15_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  // Three panels at once, 113408 B, past the 48 KB static limit and so an
  // opt-in dynamic allocation.  One buffer reused for x, y and z saves shared
  // memory that this kernel does not need -- occupancy is fixed at one block
  // per SM by the 64 registers times 1024 threads -- and pays for it in
  // barriers: every reuse needs the whole block to finish reading before the
  // next panel is stored.  Holding all three lets the u, v and w loads issue
  // together and takes the kernel from eight __syncthreads to two plus one
  // __syncwarp.  Section 16.4 of p15_gap_study.md: 7.0%, and the bank
  // conflicts of the shared buffer go to zero because the three access maps
  // no longer overlap in one array.
  extern __shared__ __align__(16) double smem[];
  double *const sbufX = smem;
  double *const sbufY = smem + NP15;
  double *const sbufZ = smem + 2 * NP15;
  double *const sflux = smem + 3 * NP15;
  double *const sDfrag = sflux + NFPTOT15;
  double *const sLift = sDfrag + 256;
  // The two i-boundary planes of q, u, v and w, 2048 doubles = 16 KB.  The
  // face points of the two faces that hold i fixed are 16 doubles = 128 B
  // apart, so a warp gathering them pulls 32 cache lines where it wants 8;
  // ncu job 66332 put that at sector/request 17.5.  Those nodes are read once
  // already by the volume phase, so the block writes them down on the way past
  // and the face phase reads them out of shared instead of going back to
  // global.  Nothing is assumed about VMapM: the node index it returns decides
  // whether the staged copy exists, and the global path is still there when it
  // does not.  Staging the whole element instead would need 128 KB more and
  // section 16.2 of p15_gap_study.md measures that carveout at +12%.
  double *const sMq = sLift + 96;
  double *const sMu = sMq + P15_MSTRIDE;
  double *const sMv = sMu + P15_MSTRIDE;
  double *const sMw = sMv + P15_MSTRIDE;

  const int elem = (int)blockIdx.x;
  if (elem >= Ne) {
    return;
  }
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = lane >> 2;
  const int colk = lane & 3;
  const int k = warp >> 1;
  const int tm = warp & 1;

  const int elem_offset = elem * NP15;
  const int face_offset = elem * NFPTOT15;
  const int npoint = NP15 * Ne;
  const int nface = NFPTOT15 * Ne;

  if (tid < 256) {
    const int t = tid >> 7;
    const int ks = (tid >> 5) & 3;
    const int r = (tid >> 2) & 7;
    const int c = tid & 3;
    sDfrag[tid] = D1D[(t * 8 + r) + (c + 4 * ks) * NQ15];
  } else if (tid < 352) {
    sLift[tid - 256] = Lift1D[tid - 256];
  }

  // Linear, coalesced ownership for the loads; the mma fragment map decides
  // who computes what, which is a different mapping and does not have to
  // agree with this one.
  //
  // The two nodes of a pair are adjacent rather than 1024 apart, so q, u, v
  // and w each move as one double2 instead of two doubles and the volume
  // phases issue half the global load instructions.  What this kernel pays
  // for is the number of instructions that traverse L1/TEX, not the bytes:
  // the sector count is identical either way.  No swizzle here reads or
  // writes bit 0 and na is even, so a pair stays a pair through them and the
  // shared store is one aligned double2 as well.
  const int na = tid << 1;
  const int nb = na + 2048;
  const int iag = elem_offset + na;
  const int ibg = elem_offset + nb;
  const double2 qa = *reinterpret_cast<const double2 *>(q + iag);
  const double2 qb = *reinterpret_cast<const double2 *>(q + ibg);

  // Output nodes of this lane: j = tm*8 + row, i = tn*8 + 2*colk and +1.
  const int jout = tm * 8 + row;
  const int outA = (2 * colk) + NQ15 * jout + 256 * k;
  const int outB = outA + 8;
  const int gA = elem_offset + outA;
  const int gB = elem_offset + outB;

  double acc0, acc1, acc2, acc3;
  double c0, c1, c2, c3;

  //- the three flux panels ----------------------------------------------
  {
    const double2 ua = *reinterpret_cast<const double2 *>(u + iag);
    const double2 ub = *reinterpret_cast<const double2 *>(u + ibg);
    const double2 va = *reinterpret_cast<const double2 *>(v + iag);
    const double2 vb = *reinterpret_cast<const double2 *>(v + ibg);
    const double2 wa = *reinterpret_cast<const double2 *>(w + iag);
    const double2 wb = *reinterpret_cast<const double2 *>(w + ibg);
    *reinterpret_cast<double2 *>(sbufX + sw_xy15(na)) =
        make_double2(qa.x * ua.x, qa.y * ua.y);
    *reinterpret_cast<double2 *>(sbufX + sw_xy15(nb)) =
        make_double2(qb.x * ub.x, qb.y * ub.y);
    *reinterpret_cast<double2 *>(sbufY + sw_xy15(na)) =
        make_double2(qa.x * va.x, qa.y * va.y);
    *reinterpret_cast<double2 *>(sbufY + sw_xy15(nb)) =
        make_double2(qb.x * vb.x, qb.y * vb.y);
    *reinterpret_cast<double2 *>(sbufZ + sw_z15(na)) =
        make_double2(qa.x * wa.x, qa.y * wa.y);
    *reinterpret_cast<double2 *>(sbufZ + sw_z15(nb)) =
        make_double2(qb.x * wb.x, qb.y * wb.y);

#if P15_MPLANES == 6
    STAGE_M_PAIR(na, qa, ua, va, wa);
    STAGE_M_PAIR(nb, qb, ub, vb, wb);
#else
    // na is even and nb = na + 2048, so the two nodes of a pair have i = na&15
    // and that plus one: the low plane can only be the first of a pair and the
    // high plane only the second.  One thread in eight writes.
    const int ia = na & 15;
    if (ia == 0) {
      const int s0 = (na >> 4) & 255;
      const int s1 = (nb >> 4) & 255;
      sMq[s0] = qa.x; sMu[s0] = ua.x; sMv[s0] = va.x; sMw[s0] = wa.x;
      sMq[s1] = qb.x; sMu[s1] = ub.x; sMv[s1] = vb.x; sMw[s1] = wb.x;
    } else if (ia == 14) {
      const int s0 = 256 + ((na >> 4) & 255);
      const int s1 = 256 + ((nb >> 4) & 255);
      sMq[s0] = qa.y; sMu[s0] = ua.y; sMv[s0] = va.y; sMw[s0] = wa.y;
      sMq[s1] = qb.y; sMu[s1] = ub.y; sMv[s1] = vb.y; sMw[s1] = wb.y;
    }
#endif
  }

  __syncthreads();
  //- numerical flux on the six faces ------------------------------------
  //
  // This runs before any of the three contractions is consumed, so the
  // gathers have the x, y and z phases to complete in; section 15 of
  // p15_gap_study.md measured that placement at 5.3%.  It now sits just after
  // the barrier rather than just before it, because the M-side planes it
  // reads are written on the other side of that barrier.
  //
  // Two face points per thread, 768 threads covering all 1536.  The four
  // coalesced fields (three normal components and Fscale) and the two maps
  // then move as double2 and int2 instead of eight doubles and four ints, so
  // the phase issues six fewer L1/TEX instructions per pair -- section 15.7
  // named MIO throttle plus LG throttle, i.e. the number of memory
  // instructions rather than their latency, as what was left.  face_offset
  // and 2*tid are both even and the device allocations are 256 B aligned, so
  // every pair load is aligned.  Section 16.3: request count falls 12.7% and
  // sector count rises 13.7%, for a net 3.0%.
  if (tid < 768) {
    const int fp0 = tid << 1;
    const int fidx = face_offset + fp0;
    const int2 mM = *reinterpret_cast<const int2 *>(VMapM + fidx);
    const int2 mP = *reinterpret_cast<const int2 *>(VMapP + fidx);
    const double2 n0 = *reinterpret_cast<const double2 *>(normal_fn + fidx);
    const double2 n1 =
        *reinterpret_cast<const double2 *>(normal_fn + fidx + nface);
    const double2 n2 =
        *reinterpret_cast<const double2 *>(normal_fn + fidx + 2 * nface);
    const double2 fs = *reinterpret_cast<const double2 *>(Fscale + fidx);

    const int iM0 = mM.x - 1, iP0 = mP.x - 1;
    const int iM1 = mM.y - 1, iP1 = mP.y - 1;

    double qM0, uM0, vM0, wM0;
#if P15_ABL_MCOAL
    { const int ic = elem_offset + fp0;
      qM0 = q[ic]; uM0 = u[ic]; vM0 = v[ic]; wM0 = w[ic]; }
#else
    LOAD_M(iM0, qM0, uM0, vM0, wM0);
#endif
    const double qP0 = q[iP0];
    const double VelM0 = uM0 * n0.x + vM0 * n1.x + wM0 * n2.x;
    const double VelP0 = u[iP0] * n0.x + v[iP0] * n1.x + w[iP0] * n2.x;
    const double a0 = 0.5 * fabs(VelP0 + VelM0);
    sflux[sw_f15(fp0)] =
        0.5 * fs.x * (qP0 * VelP0 - qM0 * VelM0 - a0 * (qP0 - qM0));

    double qM1, uM1, vM1, wM1;
#if P15_ABL_MCOAL
    { const int ic = elem_offset + fp0 + 1;
      qM1 = q[ic]; uM1 = u[ic]; vM1 = v[ic]; wM1 = w[ic]; }
#else
    LOAD_M(iM1, qM1, uM1, vM1, wM1);
#endif
    const double qP1 = q[iP1];
    const double VelM1 = uM1 * n0.y + vM1 * n1.y + wM1 * n2.y;
    const double VelP1 = u[iP1] * n0.y + v[iP1] * n1.y + w[iP1] * n2.y;
    const double a1 = 0.5 * fabs(VelP1 + VelM1);
    sflux[sw_f15(fp0 + 1)] =
        0.5 * fs.y * (qP1 * VelP1 - qM1 * VelM1 - a1 * (qP1 - qM1));
  }
  {
    // A = flux panel at (i = colk + 4*ks, j = jout); the k-step moves bits
    // 2-3, which sw_xy15 does not read.
    const int ax = sw_xy15(colk + NQ15 * jout + 256 * k);
    const int fbase = row * 4 + colk;
    mma_reset(c0, c1);
    mma_reset(c2, c3);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const double a = sbufX[ax ^ (4 * ks)];
      mma_m8n8k4_f64<UseTc>(c0, c1, a, sDfrag[(ks * 8 << 2) + fbase], c0, c1);
      mma_m8n8k4_f64<UseTc>(c2, c3, a, sDfrag[128 + (ks * 8 << 2) + fbase], c2, c3);
    }
    const double2 ea = *reinterpret_cast<const double2 *>(Escale + gA);
    const double2 eb = *reinterpret_cast<const double2 *>(Escale + gB);
    acc0 = ea.x * c0;
    acc1 = ea.y * c1;
    acc2 = eb.x * c2;
    acc3 = eb.y * c3;
  }

  //- y -----------------------------------------------------------------
  {
    // B = flux panel at (i = tn*8 + row, j = colk + 4*ks); the k-step moves
    // bits 6-7, again not read by sw_xy15.
    const int byA = sw_xy15(row + NQ15 * colk + 256 * k);
    const int byB = sw_xy15((row + 8) + NQ15 * colk + 256 * k);
    const int fbase = (tm << 7) + row * 4 + colk;
    mma_reset(c0, c1);
    mma_reset(c2, c3);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const double a = sDfrag[(ks * 8 << 2) + fbase];
      mma_m8n8k4_f64<UseTc>(c0, c1, a, sbufY[byA ^ (64 * ks)], c0, c1);
      mma_m8n8k4_f64<UseTc>(c2, c3, a, sbufY[byB ^ (64 * ks)], c2, c3);
    }
    const double2 ea = *reinterpret_cast<const double2 *>(Escale + gA + npoint);
    const double2 eb = *reinterpret_cast<const double2 *>(Escale + gB + npoint);
    acc0 += ea.x * c0;
    acc1 += ea.y * c1;
    acc2 += eb.x * c2;
    acc3 += eb.y * c3;
  }

  //- z -----------------------------------------------------------------
  {
    // The z output map is not the x/y one, so the z derivative is the only one
    // that travels back through shared memory, as it is at Nq=8.  This warp
    // owns the eight columns 8*warp .. 8*warp+7 and both halves of k.
    const int bz = sw_z15(warp * 8 + row + 256 * colk);
    const int fbase = row * 4 + colk;
    mma_reset(c0, c1);
    mma_reset(c2, c3);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const double b = sbufZ[bz ^ (1024 * ks)];
      mma_m8n8k4_f64<UseTc>(c0, c1, sDfrag[(ks * 8 << 2) + fbase], b, c0, c1);
      mma_m8n8k4_f64<UseTc>(c2, c3, sDfrag[128 + (ks * 8 << 2) + fbase], b, c2, c3);
    }
    // The 128 nodes this warp writes are exactly the 128 it just read: the z
    // mma covers ij in [8*warp, 8*warp+8) for every k, and so does the round
    // trip.  Addressing both with the same swizzle therefore makes the store
    // land inside the warp's own read set, so no other warp's live data is
    // clobbered and the barrier that used to guard the overwrite becomes a
    // __syncwarp().  The price is that the round trip no longer gets its own
    // permutation: sw_dz15 existed to make the store and the read conflict
    // free, and sw_z15 leaves the read 4-way.
    __syncwarp();
    const int dzA = sw_z15((warp * 8 + 2 * colk) + 256 * row);
    const int dzB = sw_z15((warp * 8 + 2 * colk) + 256 * (row + 8));
    sbufZ[dzA] = c0;
    sbufZ[dzA ^ 1] = c1;
    sbufZ[dzB] = c2;
    sbufZ[dzB ^ 1] = c3;
  }
  __syncthreads();
  {
    const int dA = sw_z15(outA);
    const int dB = sw_z15(outB);
    const double2 ea =
        *reinterpret_cast<const double2 *>(Escale + gA + 2 * npoint);
    const double2 eb =
        *reinterpret_cast<const double2 *>(Escale + gB + 2 * npoint);
    acc0 += ea.x * sbufZ[dA];
    acc1 += ea.y * sbufZ[dA ^ 1];
    acc2 += eb.x * sbufZ[dB];
    acc3 += eb.y * sbufZ[dB ^ 1];
  }
  // No barrier here: sflux has a buffer of its own and nothing below writes
  // shared memory.

  //- lift and assembly --------------------------------------------------
  {
    const int iA = 2 * colk;
    const int iB = iA + 8;
    const double lf1 = sLift[jout];
    const double lf3 = sLift[32 + jout];
    const double lf5 = sLift[64 + k];
    const double lf6 = sLift[80 + k];
    // Faces 2 and 4 vary in j and k only, so all four nodes share the value
    // and differ only through the Lift1D coefficient, which varies in i.
    // Every index below is even and the swizzle only ever XORs bit 0 with
    // warp-invariant bits here, so the +1 neighbour stays the XOR neighbour;
    // likewise +256 and +512 move bits the fold does not read.
    const int sb2 = sw_f15(256 + jout + NQ15 * k);
    const double fb2 = sflux[sb2];
    const double fb4 = sflux[sb2 + 512];
    const int s1A = sw_f15(iA + NQ15 * k);
    const int s1B = sw_f15(iB + NQ15 * k);
    const int s5A = sw_f15(1024 + iA + NQ15 * jout);
    const int s5B = sw_f15(1024 + iB + NQ15 * jout);

    const double l0 = lf1 * sflux[s1A] + sLift[16 + iA] * fb2 +
                      lf3 * sflux[512 + s1A] + sLift[48 + iA] * fb4 +
                      lf5 * sflux[s5A] + lf6 * sflux[s5A + 256];
    const double l1 = lf1 * sflux[s1A ^ 1] + sLift[17 + iA] * fb2 +
                      lf3 * sflux[512 + (s1A ^ 1)] + sLift[49 + iA] * fb4 +
                      lf5 * sflux[s5A ^ 1] + lf6 * sflux[(s5A ^ 1) + 256];
    const double l2 = lf1 * sflux[s1B] + sLift[16 + iB] * fb2 +
                      lf3 * sflux[512 + s1B] + sLift[48 + iB] * fb4 +
                      lf5 * sflux[s5B] + lf6 * sflux[s5B + 256];
    const double l3 = lf1 * sflux[s1B ^ 1] + sLift[17 + iB] * fb2 +
                      lf3 * sflux[512 + (s1B ^ 1)] + sLift[49 + iB] * fb4 +
                      lf5 * sflux[s5B ^ 1] + lf6 * sflux[(s5B ^ 1) + 256];

    *reinterpret_cast<double2 *>(dqdt + gA) =
        make_double2(-(acc0 + l0), -(acc1 + l1));
    *reinterpret_cast<double2 *>(dqdt + gB) =
        make_double2(-(acc2 + l2), -(acc3 + l3));
  }
}

template <bool UseTc>
void launch_tendency_fused_p15_impl(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  const int p15_smem =
      (3 * NP15 + NFPTOT15 + 256 + 96 + 4 * P15_MSTRIDE) * (int)sizeof(double);
  static bool p15_optin = false;
  if (!p15_optin) {
    cudaFuncSetAttribute(tendency_fused_p15_kernel<UseTc>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, p15_smem);
    p15_optin = true;
  }
  tendency_fused_p15_kernel<UseTc>
      <<<Ne, P15_THREADS, p15_smem, dg_cuda_stream>>>(
          dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale,
          Escale, Ne);
  check_cuda("tendency_fused_p15_kernel");
}

extern "C" void launch_tendency_fused_p15_dfma(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  launch_tendency_fused_p15_impl<false>(dqdt, D1D, Lift1D, q, u, v, w, VMapM,
                                        VMapP, normal_fn, Fscale, Escale, Ne);
}

extern "C" void launch_tendency_fused_p15_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  launch_tendency_fused_p15_impl<true>(dqdt, D1D, Lift1D, q, u, v, w, VMapM,
                                       VMapP, normal_fn, Fscale, Escale, Ne);
}

// One 16-byte global-to-shared copy that never passes through a register.
// The y epilogue's read-modify-write of dqdt is the one load in these kernels
// that the mma cannot cover, and staging it in registers is blocked by the
// register file (section 16.6); cp.async is the way in that costs nothing.
__device__ __forceinline__ void cp_async_16(void *dst, const void *src)
{
  const unsigned sm = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sm),
               "l"(src));
}

//============================================================================
// p=31 (Nq=32) fused Tensor Core tendency
//============================================================================
//
// At Nq=32 one element of q is 32768 doubles = 256 KB, which does not fit in
// the 228 KB of shared memory an SM has, let alone in one block's share of it.
// The p=15 strategy (whole element in one reused buffer, q in registers) is
// therefore structurally impossible, and this kernel keeps the structure the
// CUDA-core p=31 pair established instead: sweep one plane at a time and split
// the y term into a second kernel, because at fixed j the y contraction needs
// data from every j.  See mod_cuda_dg_kernels.cuf:1964 for that argument and
// reports/p31_gap_study.md section 5 for the measurements behind it.
//
// What changes here is only the contraction.  Section 13.5 of the same report
// measured that 87-91% of these kernels' L1/TEX wavefronts are shared, at 21.6
// FLOP per shared wavefront, so the mma attacks the measured limiter directly.
//
// Two structural facts make Nq=32 easier than Nq=8 and Nq=16 were:
//
//   x: C(i,k) = sum_l D(i,l) * FU(l,j,k)      FU = q*u
//   z: C(i,k) = sum_l FW(i,j,l) * D(k,l)      FW = q*w
//
// 1. Both land on the SAME output index pair (i,k).  At Nq=8 and Nq=16 the z
//    derivative was the one that had to travel back through shared memory
//    (sDz, sw_dz, sw_dz15); here it does not, and with it goes the accumulator
//    store whose conflict-free forms measured slower on GB200 (see the note at
//    the top of this file and section 5 of tc_paper_survey_2407.09621.md).
//    The two results are still scaled by DIFFERENT Escale components, so they
//    need two accumulator pairs and are summed only in the epilogue.
// 2. Evaluated transposed -- C^T, which with m8n8k4 is the same two operand
//    values passed in the opposite order -- the D1D operand of both directions
//    is D[8*t + lane/4][4*ks + lane%4], which does not depend on j.  Sixteen
//    doubles per lane hold it in registers for the whole j loop, so D1D never
//    goes through shared memory and each mma costs exactly one shared load.
//
// A block owns half an element: j = 0..15 or j = 16..31.  That is not a shared
// memory constraint (the block uses 42.5 KB either way, since the four face
// planes halve when the j range does) but a wave one: 152 SMs and one block
// resident each make 512 blocks four waves at 84% efficiency and 1024 blocks
// seven waves at 96%.  Only faces 1 and 3, which do not vary in j, are
// evaluated by both slabs; the face phase is about 28 us of the CUDA-core
// kernel, so that redundancy is affordable at two slabs and not at four.

// Every plane in this kernel is addressed as low + 32*high, and every mma
// operand read has the contraction index in one of those two fields and the
// tile row in the other.  Whichever way round it is, address bits 2-3 are
// invariant across an FP64 half-warp phase and bits 5-6 vary with the lane, so
// folding the latter into the former makes the phase cover sixteen distinct
// banks.  One function therefore serves the x panel, the z panel, the y panel
// and the two x-normal face planes, where p=15 needed three.
//
// Faces 5 and 6 are indexed (i,j) with j loop-uniform, so their epilogue loads
// are already a four-way broadcast and are left unswizzled: applying the fold
// there would be harmless but would buy nothing, and section 10.4 of
// tc_paper_survey_2407.09621.md records that this kernel family loses whenever
// a memory instruction is bought with extra integer ones.
__device__ __forceinline__ int sw31(int idx)
{
  return idx ^ (((idx >> 5) & 3) << 2);
}

// Illegal ablations that price the shared-memory bank conflicts of the p=31
// fused Tensor Core kernels.  Every shared address that the mma or the
// epilogue reads (P31_ABL_XZSH) or that the y kernel reads and writes
// (P31_ABL_YSH) is replaced by a lane-linear one, which is conflict free by
// construction and keeps the instruction count and the access widths.  The
// values are wrong, so these builds exist only to measure the ceiling; see
// section 32 of p31_gap_study.md.
#ifndef P31_ABL_XZSH
#define P31_ABL_XZSH 0
#endif
#ifndef P31_ABL_YSH
#define P31_ABL_YSH 0
#endif
// Attribution build: drop the sDQ staging buffer and read dqdt straight from
// global in the y epilogue.  Numerically correct (it is the pre-cp.async form
// of the kernel), and it tells apart the two shared buffers the y kernel has.
#ifndef P31_ABL_YNOCPA
#define P31_ABL_YNOCPA 0
#endif

// Illegal / attribution ablations that price faces 2 and 4 of the p=31 xz
// kernel.  Section 18.2 of p31_gap_study.md measured a ceiling of -17.8% for
// deleting them, and sections 18.3 / 19.1 / 26 measured five implementable
// forms that all lost.  These knobs separate the two things a form can change:
// how much traffic there is, and what shape the M-side address has.
//   0  production: M side through VMapM, node i + 32*j + 1024*k (stride 32)
//   1  faces 2 and 4 are not evaluated at all (sf2 = sf4 = 0).  The ceiling of
//      section 18.2, re-measured at HEAD.  Illegal.
//   2  evaluated, but the M-side index is the coalesced elem_offset + pl.
//      Same loads, same fields, same face points; only the address stride
//      changes from 32 doubles to 1.  This is the prize every one of the five
//      forms was trying to collect, priced with no toll attached.  Illegal.
//   3  same as 2 with the P side made coalesced as well.  Illegal.
//   4  the direct M-side address of section 26 (elem_offset + 31 + 32*pl and
//      elem_offset + 32*pl), which drops the VMapM load but keeps the stride.
//      Numerically correct; it reproduces the fifth form (+2.48%) so that it
//      can be compared with 2 in one job.
//   6  production M side, but the P-side index is the coalesced
//      elem_offset + pl.  The one-variable counterpart of 2 for the P side,
//      with all eight field loads still distinct.  Illegal.
//   5  M side reuses the P-side index (iM = iP), which deletes the four
//      strided loads and the VMapM load but keeps the P gather and the flux
//      arithmetic.  Illegal.
#ifndef P31_ABL_F24
#define P31_ABL_F24 0
#endif

// Programmatic Dependent Launch for the p=31 fused Tensor Core path.  Here the
// stage is two grids, xz -> y (the face fluxes are evaluated inside xz), so
// only the last rung of the p=127 ladder applies: y is made a PDL dependent of
// xz and waits immediately before its only read of dqdt.
//   0  ordinary stream order (control)
//   1  y dependent on xz
#ifndef P31_PDL_STAGE
#define P31_PDL_STAGE 1
#endif

__device__ __forceinline__ double p31_face_flux_core(
    int fidx, int iM, int iP, const double *q, const double *u,
    const double *v, const double *w, const double *normal_fn,
    const double *Fscale, int nface)
{
  const double n1 = normal_fn[fidx];
  const double n2 = normal_fn[fidx + nface];
  const double n3 = normal_fn[fidx + 2 * nface];
  const double qM = q[iM];
  const double qP = q[iP];
  const double VelM = u[iM] * n1 + v[iM] * n2 + w[iM] * n3;
  const double VelP = u[iP] * n1 + v[iP] * n2 + w[iP] * n3;
  const double alpha = 0.5 * fabs(VelP + VelM);
  return 0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
}

__device__ __forceinline__ double p31_face_flux_tc(
    int fidx, const double *q, const double *u, const double *v,
    const double *w, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface)
{
  return p31_face_flux_core(fidx, VMapM[fidx] - 1, VMapP[fidx] - 1, q, u, v, w,
                            normal_fn, Fscale, nface);
}

// Warp mma tile and chunk-loop pipelining for the p=31 fused Tensor Core
// kernels.  The block tile is the whole 32x32 output plane in both kernels, so
// P31_XZ_TM / P31_XZ_TN (and the y pair) fix the warp grid and therefore the
// thread count: widening a warp tile can only be paid for with warps.
//
//   xz per k step: TM + TN shared loads for 2*TM*TN mma
//   y  per k step: TM      shared loads for   TM*TN mma
//
// P31_*_DB switches the plane loop to the double-buffered form of
// tendency_p255_kernel: the plane for jl+1 is stored into the other shared
// buffer before the mma of jl reads its own, so the loop needs one barrier per
// plane instead of two.  Section 30 of p31_gap_study.md has the measurements.
template <bool UseTc>
__global__ __launch_bounds__(P31_XZ_THREADS, P31_XZ_MINB) void
tendency_fused_p31_xz_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  constexpr int TM = P31_XZ_TM;
  constexpr int TN = P31_XZ_TN;
  constexpr int NT = P31_XZ_THREADS;
  constexpr int NBUF = P31_XZ_NBUF;
  constexpr int LDIT = 512 / NT;

  // sFU and sFW hold the current j plane under ONE address map, node index
  // i + 32*k.  For x the contraction index l is the i field and for z it is
  // the k field, which is why one store path and one global load map serve
  // both panels.
  // Faces 2 and 4 are (j,k) planes and faces 5 and 6 are (i,j) planes, so
  // under the mma output map a consumer thread needs 16 j values of the first
  // pair and 32 of the second.  Faces 1 and 3 are (i,k) planes and stay in
  // registers; see the bijection note below.
  //
  // Single buffered this is 41.5 KB and stays static.  The double-buffered
  // form is 57.5 KB, over the 48 KB static limit, so it has to be dynamic --
  // and that conversion is not free at the 1x1 tile: on its own, with an
  // otherwise bit-identical kernel, it costs 1.70% (section 30.8 of
  // p31_gap_study.md).  The two are therefore separate knobs, so the
  // pipelining A/B is not charged for it.  At the 2x2 tile the sign flips.
#if P31_XZ_DYN
  extern __shared__ __align__(16) double smem31[];
  double *const sFU = smem31;
  double *const sFW = smem31 + NBUF * 1024;
  double *const sf2 = smem31 + 2 * NBUF * 1024;
  double *const sf4 = sf2 + 1024;
  double *const sf5 = sf4 + 1024;
  double *const sf6 = sf5 + 512;
  double *const sLift = sf6 + 512;
#else
  __shared__ __align__(16) double sFU[NBUF * 1024], sFW[NBUF * 1024];
  __shared__ __align__(16) double sf2[1024], sf4[1024];
  __shared__ __align__(16) double sf5[512], sf6[512];
  __shared__ __align__(16) double sLift[192];
#endif

  const int elem = (int)blockIdx.x >> 1;
  if (elem >= Ne) {
    return;
  }
  const int j0 = ((int)blockIdx.x & 1) * JSLAB31;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ti = warp % P31_XZ_WM;
  const int tk = warp / P31_XZ_WM;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int elem_offset = elem * NP31;
  const int face_offset = elem * NFPTOT31;
  const int npoint = NP31 * Ne;
  const int nface = NFPTOT31 * Ne;

  for (int t = tid; t < 192; t += NT) {
    sLift[t] = Lift1D[t];
  }


  // D1D is Fortran column major, so D(r,l) is at r + 32*l.  x wants the B
  // operand D[i][l] with i = 8*(ti*TM+m) + row and z the A operand D[k][l]
  // with k = 8*(tk*TN+n) + row; both read column 4*ks + colk, and neither
  // depends on j.
  double Dx[TM][8], Dz[TN][8];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      Dx[m][ks] = D1D[((ti * TM + m) * 8 + row) + NQ31 * (4 * ks + colk)];
    }
  }
#pragma unroll
  for (int n = 0; n < TN; ++n) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      Dz[n][ks] = D1D[((tk * TN + n) * 8 + row) + NQ31 * (4 * ks + colk)];
    }
  }

  // Output nodes of this lane, for every j: i = 8*(ti*TM+m) + 2*colk and +1 at
  // k = 8*(tk*TN+n) + row.  The pair is adjacent in i, so dqdt and Escale move
  // as aligned double2.
  int i0[TM], kout[TN];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    i0[m] = (ti * TM + m) * 8 + 2 * colk;
  }
#pragma unroll
  for (int n = 0; n < TN; ++n) {
    kout[n] = (tk * TN + n) * 8 + row;
  }

  // Faces 1 and 3 are (i,k) planes, and (thread, tile, {0,1}) -> i0 + 32*kout
  // and +1 is a bijection onto the 1024 face points, so each thread evaluates
  // exactly the points it will consume.  No redistribution, and the gather is
  // 64-byte runs per warp, the same shape as the epilogue.
  double f1a[TM][TN], f1b[TM][TN], f3a[TM][TN], f3b[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      const int p13 = i0[m] + NQ31 * kout[n];
      f1a[m][n] = p31_face_flux_tc(face_offset + p13, q, u, v, w, VMapM, VMapP,
                                   normal_fn, Fscale, nface);
      f1b[m][n] = p31_face_flux_tc(face_offset + p13 + 1, q, u, v, w, VMapM,
                                   VMapP, normal_fn, Fscale, nface);
      f3a[m][n] = p31_face_flux_tc(face_offset + 2048 + p13, q, u, v, w, VMapM,
                                   VMapP, normal_fn, Fscale, nface);
      f3b[m][n] = p31_face_flux_tc(face_offset + 2048 + p13 + 1, q, u, v, w,
                                   VMapM, VMapP, normal_fn, Fscale, nface);
    }
  }

  // Faces 2 and 4: 16 j by 32 k is exactly 512 points.  The producer takes k
  // from the high nibble so a half warp holds k fixed and walks 16 consecutive
  // j, which is one 128-byte global run and, since the fold source is then
  // constant, a conflict-free shared store.
#pragma unroll
  for (int t2 = 0; t2 < 512 / NT; ++t2) {
    const int t = tid + t2 * NT;
    const int jl = t & 15;
    const int kk = t >> 4;
    const int pl = (j0 + jl) + NQ31 * kk;
    const int sa = sw31(jl + NQ31 * kk);
#if P31_ABL_F24 == 1
    (void)pl;
    sf2[sa] = 0.0;
    sf4[sa] = 0.0;
#elif P31_ABL_F24 == 0
    sf2[sa] = p31_face_flux_tc(face_offset + 1024 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
    sf4[sa] = p31_face_flux_tc(face_offset + 3072 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
#else
    {
      const int fi2 = face_offset + 1024 + pl;
      const int fi4 = face_offset + 3072 + pl;
#if P31_ABL_F24 == 2 || P31_ABL_F24 == 3
      const int m2 = elem_offset + pl;
      const int m4 = m2;
#elif P31_ABL_F24 == 6
      const int m2 = VMapM[fi2] - 1;
      const int m4 = VMapM[fi4] - 1;
#elif P31_ABL_F24 == 4
      const int m2 = elem_offset + (NQ31 - 1) + NQ31 * pl;
      const int m4 = elem_offset + NQ31 * pl;
#else
      const int m2 = VMapP[fi2] - 1;
      const int m4 = VMapP[fi4] - 1;
#endif
#if P31_ABL_F24 == 3
      const int p2 = m2;
      const int p4 = m4;
#elif P31_ABL_F24 == 6
      const int p2 = elem_offset + pl;
      const int p4 = p2;
#else
      const int p2 = VMapP[fi2] - 1;
      const int p4 = VMapP[fi4] - 1;
#endif
      sf2[sa] = p31_face_flux_core(fi2, m2, p2, q, u, v, w, normal_fn, Fscale,
                                   nface);
      sf4[sa] = p31_face_flux_core(fi4, m4, p4, q, u, v, w, normal_fn, Fscale,
                                   nface);
    }
#endif
  }
  // Faces 5 and 6: 32 i by 16 j.  A warp walks 32 consecutive i at fixed j.
#pragma unroll
  for (int t2 = 0; t2 < 512 / NT; ++t2) {
    const int t = tid + t2 * NT;
    const int ii = t & 31;
    const int jl = t >> 5;
    const int pl = ii + NQ31 * (j0 + jl);
    const int sa = ii + NQ31 * jl;
    sf5[sa] = p31_face_flux_tc(face_offset + 4096 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
    sf6[sa] = p31_face_flux_tc(face_offset + 5120 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
  }

  // Operand base addresses.  For sFU the fold source is kout and for sFW it is
  // colk, and neither depends on ks, so the k-step is one XOR by a constant on
  // the x side and one add on the z side.
  int ax[TN], bz[TM];
#pragma unroll
  for (int n = 0; n < TN; ++n) {
    ax[n] = sw31(colk + NQ31 * kout[n]);
  }
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    bz[m] = sw31(((ti * TM + m) * 8 + row) + NQ31 * colk);
  }

  // The plane load is linear and coalesced, which the mma fragment map is not
  // and does not have to be: a half warp covers 32 consecutive nodes of one k
  // line, one 256-byte run, and each thread moves a double2 of q, u and w.
  int ldsh[LDIT], gidx[LDIT];
#pragma unroll
  for (int t = 0; t < LDIT; ++t) {
    const int s = tid + t * NT;
    const int ldi = 2 * (s & 15);
    const int ldk = s >> 4;
    ldsh[t] = sw31(ldi + NQ31 * ldk);
    gidx[t] = elem_offset + ldi + NQ31 * j0 + (NQ31 * NQ31) * ldk;
  }

  double2 qp[LDIT], up[LDIT], wp[LDIT];
#define P31_XZ_LOADREGS()                                                     \
  do {                                                                        \
    _Pragma("unroll") for (int t = 0; t < LDIT; ++t)                          \
    {                                                                         \
      qp[t] = *reinterpret_cast<const double2 *>(q + gidx[t]);                \
      up[t] = *reinterpret_cast<const double2 *>(u + gidx[t]);                \
      wp[t] = *reinterpret_cast<const double2 *>(w + gidx[t]);                \
      gidx[t] += NQ31;                                                        \
    }                                                                         \
  } while (0)
#define P31_XZ_STORE(BUF)                                                     \
  do {                                                                        \
    _Pragma("unroll") for (int t = 0; t < LDIT; ++t)                          \
    {                                                                         \
      *reinterpret_cast<double2 *>(sFU + (BUF) * 1024 + ldsh[t]) =            \
          make_double2(qp[t].x * up[t].x, qp[t].y * up[t].y);                 \
      *reinterpret_cast<double2 *>(sFW + (BUF) * 1024 + ldsh[t]) =            \
          make_double2(qp[t].x * wp[t].x, qp[t].y * wp[t].y);                 \
    }                                                                         \
  } while (0)

  // One block per SM leaves no second block to interleave with, so the plane
  // for j+1 is issued before the mma of j consumes the plane for j.
#if P31_XZ_EARLY
  P31_XZ_LOADREGS();
#if P31_XZ_DB
  P31_XZ_STORE(0);
  P31_XZ_LOADREGS();
#endif
#endif
  __syncthreads();

#if P31_PDL_STAGE >= 1
  // The face phase above is this grid's only DRAM-latency block; let the y
  // grid start now.  Its mma does not read dqdt, and the wait sits in front of
  // the prefetch that does.
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // Lift1D(Nq,6) varies in j for faces 1 and 3, in i for faces 2 and 4 and in
  // k for faces 5 and 6, so the coefficients that do not vary in j are read
  // once here.
  double lf2a[TM], lf2b[TM], lf4a[TM], lf4b[TM], lf5[TN], lf6[TN];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    lf2a[m] = sLift[32 + i0[m]];
    lf2b[m] = sLift[33 + i0[m]];
    lf4a[m] = sLift[96 + i0[m]];
    lf4b[m] = sLift[97 + i0[m]];
  }
#pragma unroll
  for (int n = 0; n < TN; ++n) {
    lf5[n] = sLift[128 + kout[n]];
    lf6[n] = sLift[160 + kout[n]];
  }

  // P31_XZ_EARLY chooses which side of the face barrier the first plane load
  // sits on.  At the production 1x1 tile the late form wins by 1.5%: the face
  // gather is what saturates this kernel's L1/global path (section 18.1), and
  // a plane load hoisted in front of the barrier competes with it.  Wide warp
  // tiles move three times as many doubles per thread and prefer the early
  // form.  The pipelined preamble follows the same choice, paying one extra
  // barrier when it is late (18 barriers a slab against 33).
#if !P31_XZ_EARLY
  P31_XZ_LOADREGS();
#if P31_XZ_DB
  P31_XZ_STORE(0);
  P31_XZ_LOADREGS();
  __syncthreads();
#endif
#endif

  for (int jl = 0; jl < JSLAB31; ++jl) {
#if P31_XZ_DB
    const int cur = jl & 1;
    if (jl + 1 < JSLAB31) {
      P31_XZ_STORE(cur ^ 1);
      if (jl + 2 < JSLAB31) {
        P31_XZ_LOADREGS();
      }
    }
#else
    const int cur = 0;
    P31_XZ_STORE(0);
    if (jl + 1 < JSLAB31) {
      P31_XZ_LOADREGS();
    }
    __syncthreads();
#endif

    // x^T = (D * FU)^T and z^T = (FW * D^T)^T on this j plane.  Same output
    // map, separate accumulators, because Escale differs by direction.
    double cx0[TM][TN], cx1[TM][TN], cz0[TM][TN], cz1[TM][TN];
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        mma_reset(cx0[m][n], cx1[m][n]);
        mma_reset(cz0[m][n], cz1[m][n]);
      }
    }
    const double *const pFU = sFU + cur * 1024;
    const double *const pFW = sFW + cur * 1024;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      double av[TN], bv[TM];
#pragma unroll
      for (int n = 0; n < TN; ++n) {
#if P31_ABL_XZSH
        av[n] = pFU[(lane + 64 * ks + 256 * n) & 1023];
#else
        av[n] = pFU[ax[n] ^ (4 * ks)];
#endif
      }
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#if P31_ABL_XZSH
        bv[m] = pFW[(lane + 64 * ks + 256 * m) & 1023];
#else
        bv[m] = pFW[bz[m] + 128 * ks];
#endif
      }
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          mma_m8n8k4_f64<UseTc>(cx0[m][n], cx1[m][n], av[n], Dx[m][ks],
                                cx0[m][n], cx1[m][n]);
          mma_m8n8k4_f64<UseTc>(cz0[m][n], cz1[m][n], Dz[n][ks], bv[m],
                                cz0[m][n], cz1[m][n]);
        }
      }
    }

    const int j = j0 + jl;
    const double lf1 = sLift[j];
    const double lf3 = sLift[64 + j];
    // Face values that depend on only one of the two tile indices are read
    // once outside the tile pair, not once per tile.
    double fb2[TN], fb4[TN];
#pragma unroll
    for (int n = 0; n < TN; ++n) {
#if P31_ABL_XZSH
      const int a24 = (lane + 32 * n) & 1023;
#else
      const int a24 = sw31(jl + NQ31 * kout[n]);
#endif
      fb2[n] = sf2[a24];
      fb4[n] = sf4[a24];
    }
    double2 fb5[TM], fb6[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#if P31_ABL_XZSH
      const int a56 = (2 * lane + 64 * m) & 511;
#else
      const int a56 = i0[m] + NQ31 * jl;
#endif
      fb5[m] = *reinterpret_cast<const double2 *>(sf5 + a56);
      fb6[m] = *reinterpret_cast<const double2 *>(sf6 + a56);
    }

    // Same summation order as the CUDA-core p=31 kernels.
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        const int nidx =
            elem_offset + i0[m] + NQ31 * j + (NQ31 * NQ31) * kout[n];
        const double2 ex = *reinterpret_cast<const double2 *>(Escale + nidx);
        const double2 ez =
            *reinterpret_cast<const double2 *>(Escale + nidx + 2 * npoint);
        *reinterpret_cast<double2 *>(dqdt + nidx) = make_double2(
            -(ex.x * cx0[m][n] + ez.x * cz0[m][n] + lf1 * f1a[m][n] +
              lf2a[m] * fb2[n] + lf3 * f3a[m][n] + lf4a[m] * fb4[n] +
              lf5[n] * fb5[m].x + lf6[n] * fb6[m].x),
            -(ex.y * cx1[m][n] + ez.y * cz1[m][n] + lf1 * f1b[m][n] +
              lf2b[m] * fb2[n] + lf3 * f3b[m][n] + lf4b[m] * fb4[n] +
              lf5[n] * fb5[m].y + lf6[n] * fb6[m].y));
      }
    }
    __syncthreads();
  }
#undef P31_XZ_LOADREGS
#undef P31_XZ_STORE
}

//> p=31 y volume term, accumulated onto what the xz kernel wrote.
//
// Threads are (i,j) with k the inner loop, and the contraction
// C(i,j) = sum_l FV(i,k,l) * D(j,l) is structurally the z contraction of the
// first kernel: transposed it wants D as the A operand from registers and the
// plane as the B operand from shared, under the same address map and the same
// swizzle.  The shared operand depends only on the i tile, so widening the
// warp tile in j is the direction that pays here.  The block owns half an
// element in k, for the same wave reason.
__device__ __forceinline__ int ystsh(int idx, int t)
{
#if P31_ABL_YSH
  const int tid = (int)threadIdx.x;
  return (2 * (tid & 31) + 64 * (tid >> 5) + 512 * t) & 1023;
#else
  (void)t;
  return idx;
#endif
}

template <bool UseTc>
__global__ __launch_bounds__(P31_Y_THREADS, P31_Y_MINB) void
tendency_fused_p31_y_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  constexpr int TM = P31_Y_TM;
  constexpr int TN = P31_Y_TN;
  constexpr int NT = P31_Y_THREADS;
  constexpr int NBUF = P31_Y_NBUF;
  constexpr int LDIT = 512 / NT;
  constexpr int NTILE = TM * TN;

  // Two 8 KB stages for the dqdt tile the epilogue reads back.  Each thread
  // asks for the 16-byte slots it will read, so the buffer needs no barrier of
  // its own; see section 19.4 of p63_gap_study.md.
  // At most 32 KB in every configuration, so this one stays static.
  __shared__ __align__(16) double sFV[NBUF * 1024];
  __shared__ __align__(16) double sDQ[4 * NTILE * NT];

  const int elem = (int)blockIdx.x >> 1;
  if (elem >= Ne) {
    return;
  }
  const int k0 = ((int)blockIdx.x & 1) * JSLAB31;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ti = warp % P31_Y_WM;
  const int tj = warp / P31_Y_WM;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int elem_offset = elem * NP31;
  const int npoint = NP31 * Ne;

  double Dy[TN][8];
#pragma unroll
  for (int n = 0; n < TN; ++n) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      Dy[n][ks] = D1D[((tj * TN + n) * 8 + row) + NQ31 * (4 * ks + colk)];
    }
  }

  int i0[TM], jout[TN], by[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    i0[m] = (ti * TM + m) * 8 + 2 * colk;
    by[m] = sw31(((ti * TM + m) * 8 + row) + NQ31 * colk);
  }
#pragma unroll
  for (int n = 0; n < TN; ++n) {
    jout[n] = (tj * TN + n) * 8 + row;
  }

  int ldsh[LDIT], gidx[LDIT];
#pragma unroll
  for (int t = 0; t < LDIT; ++t) {
    const int s = tid + t * NT;
    const int ldi = 2 * (s & 15);
    const int ldj = s >> 4;
    ldsh[t] = sw31(ldi + NQ31 * ldj);
    gidx[t] = elem_offset + ldi + NQ31 * ldj + (NQ31 * NQ31) * k0;
  }

  double2 qp[LDIT], vp[LDIT];
#define P31_Y_LOADREGS()                                                      \
  do {                                                                        \
    _Pragma("unroll") for (int t = 0; t < LDIT; ++t)                          \
    {                                                                         \
      qp[t] = *reinterpret_cast<const double2 *>(q + gidx[t]);                \
      vp[t] = *reinterpret_cast<const double2 *>(v + gidx[t]);                \
      gidx[t] += NQ31 * NQ31;                                                 \
    }                                                                         \
  } while (0)
#define P31_Y_STORE(BUF)                                                      \
  do {                                                                        \
    _Pragma("unroll") for (int t = 0; t < LDIT; ++t)                          \
    {                                                                         \
      *reinterpret_cast<double2 *>(sFV + (BUF) * 1024 + ystsh(ldsh[t], t)) =            \
          make_double2(qp[t].x * vp[t].x, qp[t].y * vp[t].y);                 \
    }                                                                         \
  } while (0)

  int nidx0[TM][TN];
#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      nidx0[m][n] =
          elem_offset + i0[m] + NQ31 * jout[n] + (NQ31 * NQ31) * k0;
    }
  }

  P31_Y_LOADREGS();

#if P31_PDL_STAGE >= 1
  // dqdt is what the xz grid writes, and this prefetch is the first read of it.
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
#if !P31_ABL_YNOCPA
#pragma unroll
  for (int m = 0; m < TM; ++m) {
#pragma unroll
    for (int n = 0; n < TN; ++n) {
      cp_async_16(sDQ + 2 * (tid + NT * (m * TN + n)), dqdt + nidx0[m][n]);
    }
  }
  asm volatile("cp.async.commit_group;\n" ::);
#endif

#if P31_Y_DB
  P31_Y_STORE(0);
  P31_Y_LOADREGS();
  __syncthreads();
#endif

  for (int kl = 0; kl < JSLAB31; ++kl) {
#if P31_Y_DB
    const int cur = kl & 1;
    if (kl + 1 < JSLAB31) {
      P31_Y_STORE(cur ^ 1);
      if (kl + 2 < JSLAB31) {
        P31_Y_LOADREGS();
      }
    }
#else
    const int cur = 0;
    P31_Y_STORE(0);
    if (kl + 1 < JSLAB31) {
      P31_Y_LOADREGS();
    }
#endif
#if !P31_ABL_YNOCPA
    if (kl + 1 < JSLAB31) {
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          cp_async_16(sDQ + 2 * NTILE * NT * ((kl + 1) & 1) +
                          2 * (tid + NT * (m * TN + n)),
                      dqdt + nidx0[m][n] + (NQ31 * NQ31) * (kl + 1));
        }
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
#endif
#if !P31_Y_DB
    __syncthreads();
#endif

    double c0[TM][TN], c1[TM][TN];
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        mma_reset(c0[m][n], c1[m][n]);
      }
    }
    const double *const pFV = sFV + cur * 1024;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      double bv[TM];
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#if P31_ABL_YSH
        bv[m] = pFV[(lane + 64 * ks + 256 * m) & 1023];
#else
        bv[m] = pFV[by[m] + 128 * ks];
#endif
      }
#pragma unroll
      for (int m = 0; m < TM; ++m) {
#pragma unroll
        for (int n = 0; n < TN; ++n) {
          mma_m8n8k4_f64<UseTc>(c0[m][n], c1[m][n], Dy[n][ks], bv[m],
                                c0[m][n], c1[m][n]);
        }
      }
    }

#if !P31_ABL_YNOCPA
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
#pragma unroll
    for (int m = 0; m < TM; ++m) {
#pragma unroll
      for (int n = 0; n < TN; ++n) {
        const int nidx = nidx0[m][n] + (NQ31 * NQ31) * kl;
        const double2 ey =
            *reinterpret_cast<const double2 *>(Escale + nidx + npoint);
#if P31_ABL_YNOCPA
        double2 out = *reinterpret_cast<const double2 *>(dqdt + nidx);
#else
        double2 out = *reinterpret_cast<const double2 *>(
            sDQ + 2 * NTILE * NT * (kl & 1) + 2 * (tid + NT * (m * TN + n)));
#endif
        out.x -= ey.x * c0[m][n];
        out.y -= ey.y * c1[m][n];
        *reinterpret_cast<double2 *>(dqdt + nidx) = out;
      }
    }
    __syncthreads();
  }
#undef P31_Y_LOADREGS
#undef P31_Y_STORE
}

template <bool UseTc>
void launch_tendency_fused_p31_impl(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
#if P31_XZ_DYN
  constexpr size_t smem_xz = (2 * P31_XZ_NBUF * 1024 + 3264) * sizeof(double);
  static bool opted_in = false;
  if (!opted_in) {
    cudaFuncSetAttribute(tendency_fused_p31_xz_kernel<UseTc>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_xz);
    opted_in = true;
  }
#else
  constexpr size_t smem_xz = 0;
#endif
  tendency_fused_p31_xz_kernel<UseTc>
      <<<2 * Ne, P31_XZ_THREADS, smem_xz, dg_cuda_stream>>>(
          dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale,
          Escale, Ne);
#if P31_PDL_STAGE >= 1
  {
    cudaLaunchConfig_t ycfg = {};
    cudaLaunchAttribute yattr[1];
    ycfg.gridDim = dim3(2 * Ne);
    ycfg.blockDim = dim3(P31_Y_THREADS);
    ycfg.dynamicSmemBytes = 0;
    ycfg.stream = dg_cuda_stream;
    yattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    yattr[0].val.programmaticStreamSerializationAllowed = 1;
    ycfg.attrs = yattr;
    ycfg.numAttrs = 1;
    cudaLaunchKernelEx(&ycfg, tendency_fused_p31_y_kernel<UseTc>, dqdt, D1D, q,
                       v, Escale, Ne);
  }
#else
  tendency_fused_p31_y_kernel<UseTc>
      <<<2 * Ne, P31_Y_THREADS, 0, dg_cuda_stream>>>(dqdt, D1D, q, v, Escale,
                                                     Ne);
#endif
  check_cuda("tendency_fused_p31 kernels");
}

extern "C" void launch_tendency_fused_p31_dfma(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  launch_tendency_fused_p31_impl<false>(dqdt, D1D, Lift1D, q, u, v, w, VMapM,
                                        VMapP, normal_fn, Fscale, Escale, Ne);
}

extern "C" void launch_tendency_fused_p31_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  launch_tendency_fused_p31_impl<true>(dqdt, D1D, Lift1D, q, u, v, w, VMapM,
                                       VMapP, normal_fn, Fscale, Escale, Ne);
}

//============================================================================
// p=63 (Nq=64) fused Tensor Core tendency
//============================================================================
//
// Nq=64 is where the p=31 design stops working, for two reasons, and where the
// p=255 tile GEMM starts fitting exactly.
//
// The p=31 kernel evaluated its own face fluxes and held a whole plane in
// shared.  Here Ne is 4**3 against 152 SMs, so an element has to be spread
// over many blocks and in-kernel face evaluation would repeat the (i,k) faces
// once per block; the fluxes come from flux_bnd instead.  And a plane is
// 32 KB, so the x and z panels together would be 64 KB.
//
// Chunking the contraction with BK=16 fixes both: three 64x16 panels are
// 24 KB, and one block owns one (element, j plane), which is 64*64 = 4096
// blocks, 27 waves.  This is exactly the sA/sB arrangement of
// tendency_p255_tc_kernel, so sw255 applies unchanged -- the address is
// l + 16*outer in both.
//
// At Nq=64 the p=31 trick of keeping the D1D fragment in registers is gone:
// there are 16 k-steps and two tile rows per warp, so a lane would need 64
// doubles.  D goes through shared like the flux panels.  It is read as the B
// operand of x and the A operand of z, one panel serving both.
//
// Transposed, with n the fast (i) index and m the slow one:
//   x: C[m=k][n=i] = sum_l FU[k][l] * D[i][l]      A = sFU, B = sD
//   z: C[m=k][n=i] = sum_l D[k][l]  * FW[i][l]     A = sD,  B = sFW
// Escale differs by direction, so the two accumulator sets stay separate.

// Depth of the contraction chunk.  All three legal values were measured
// (Slurm 60560, Ne=4**3, nstep=20):
//
//   BK63   panels    shared   us/stage
//     16   3 x 1024   24 KB      660.5
//     32   3 x 2048   48 KB      615.1
//     64   3 x 4096   96 KB      572.3   <- kept
//
// Deeper wins for a reason that has nothing to do with reuse: the kernel is
// latency bound, not bandwidth or issue bound (section 13.4), and a chunk loop
// with no prefetch stalls once per chunk because the loads for chunk k+1 are
// not issued until the mma of chunk k has consumed chunk k.  At BK63=64 there
// is no chunk loop at all, so every global load of the plane is in flight
// before the first mma and the memory-level parallelism is the whole panel
// instead of a quarter of it.  Section 16 of p63_gap_study.md.

// Two shared layouts, because the transpose between global and the mma has to
// be paid somewhere.  A flux panel arrives from global with i fast and is
// wanted by the mma with the contraction index fast; the two are orthogonal,
// so either the shared store takes the conflict (lanes walking the outer
// index all land in one bank) or the read does.
//
//   sD layout   flux layout   us/stage
//   l-fast      l-fast           564.0
//   outer-fast  l-fast           556.7   <- kept
//   l-fast      outer-fast       637.2
//   outer-fast  outer-fast       656.7
//
// A first attempt folded l into bits 3-4 and looked conflict free by the
// usual test -- 32 lanes, 32 distinct banks -- but measured a uniform 2 extra
// wavefronts per LDS.64.  The usual test is the wrong one for 8-byte shared
// accesses: 32 lanes times 8 B is 256 B against 128 B of banks, so the access
// is two phases of 16 lanes and what has to be distinct is d mod 16 within a
// half warp.  There only two bits of row and two of colk vary, so a fold into
// bits 3-4 leaves bit 4 outside the window and the 16 lanes cover 8 banks.
// Folding into bits 2-3 instead puts both varying pairs inside it.  That took
// the load conflicts from 8.4 M back to 0 and the kernel from 555.6 to 544.4
// us/stage; section 16.5.
//
// Which panels want which layout was then swept at nstep=400:
//
//   sFU (xz)    sD          sFW (xz)    sFV (y)    us/stage
//   l-fast      l-fast      l-fast      l-fast        563.8
//   l-fast      outer-fast  l-fast      l-fast        544.4
//   l-fast      outer-fast  outer-fast  l-fast        539.0
//   outer-fast  outer-fast  outer-fast  l-fast        533.7   <- kept
//   l-fast      outer-fast  outer-fast  outer-fast    629.2
//
// Everything the mma reads wants the outer-fast layout except sFV, and sFV is
// not an exception to the layout rule at all: its transpose makes the y kernel
// 7% faster once the epilogue's read-modify-write of dqdt is taken away
// (233 against 250 us under ncu).  With that load present, the shorter mma
// phase no longer covers it and long scoreboard goes 31% to 50%.  Moving the
// read-modify-write to the xz kernel, which has twice the mma to hide it
// behind, was tried and just moves the cost (541.0 against 538.9).  Section
// 16.6.

// l-fast panels: idx = l + 64*outer.  The read has l in bits 0-1 (colk) and
// the outer index in bits 6-8 (row), so folding the latter into bits 2-4
// spreads it over all 32 banks; the store has the lanes walking l, which is
// already 32 consecutive addresses.
__device__ __forceinline__ int sw63(int idx)
{
  return idx ^ (((idx >> 6) & 7) << 2);
}

// outer-fast panel: idx = outer + 64*l.  Now the store is the contiguous side
// and l sits in bits 6-11, so its low two bits fold into bits 3-4 instead.
__device__ __forceinline__ int swt63(int idx)
{
  return idx ^ (((idx >> 6) & 3) << 2);
}

// sFU is the one panel whose store conflicts.  Its address is k + 64*l with
// the lanes walking l at store time and k at read time, so swt63 -- which
// only folds l's low two bits into bits 2-3 -- leaves the sixteen lanes of a
// store phase on four banks.  Folding l's bits 2-3 into bits 0-1 as well
// makes the map bijective on l's low four bits, so a store phase covers all
// sixteen banks; the read is unharmed because l = 4*ks + colk there, which
// puts colk in bits 2-3 as before and ks -- a compile-time constant of the
// unrolled k loop -- in bits 0-1.  Section 19 of p63_gap_study.md.
__device__ __forceinline__ int swu63(int idx)
{
  return idx ^ ((((idx >> 6) & 3) << 2) | ((idx >> 8) & 3));
}

// Warp shape of the volume kernels.  The warp grid is 4 rows by P63_WN
// columns and each warp owns 2 by P63_TN of the 8x8 mma tiles, so the block
// always covers the whole 64x64 plane: 4*2*8 rows and P63_WN*P63_TN*8 columns
// with P63_WN*P63_TN = 8.
//
// All three legal shapes were measured (Slurm 59919, 59924, Ne=4**3):
//
//   P63_WN  threads  blocking  reg (xz)  occupancy  Main [ms/step]
//        2      256       2x4       198      12.5%       2.41994
//        4      512       2x2       124      25.0%       2.21588   <- kept
//        8     1024       2x1        64      50.0%       2.22658
//
// The p=255 shape (2x4, the one that minimizes operand loads per k-step) is
// the worst here, because at 198 registers only one block fits an SM and a
// 12.5% occupancy kernel has nothing to hide latency with.  Halving the
// accumulators buys 9.4%.  Halving them again doubles occupancy once more and
// buys nothing: 2x1 needs 2+1 operand loads per k-step against the 2+2 of
// 2x2 for half the tiles, so the extra warps spend their slots on shared
// traffic.  Occupancy is worth chasing only until the operand loads start
// paying for it.

// The y kernel carries one accumulator set where the xz kernel carries two,
// so its shape is set separately.  Two blocks per SM is what that buys: 512
// threads asking for 64 registers is exactly the register file twice over,
// and the y kernel reaches 64 with no spill where the xz kernel spills 128 to
// 176 bytes and loses 8%.  Section 19 of p63_gap_study.md.

// Programmatic Dependent Launch stage for the p=63 fused Tensor Core path.
// The three grids of a stage are face flux -> xz -> y, the same shape p=127
// has, so the same ladder applies (p127_gap_study.md section 19):
//   0  C++ face kernel, ordinary stream order, no griddepcontrol  (control)
//   1  face -> xz PDL, hint left implicit at face grid exit
//   2  as 1 plus griddepcontrol.launch_dependents at the top of the face grid
//   3  as 2 plus y made a PDL dependent of xz
// Moving y's wait behind its first panel staging (a fifth rung) was measured
// and is not a difference; see section 50.4 of p63_gap_study.md.
#ifndef P63_PDL_STAGE
#define P63_PDL_STAGE 3
#endif

// Same numerical flux as the Fortran elembnd_flux_kernel (pair_nq2 = 0).
// Launched from C++ so it can hint PDL; the dependent xz kernel waits before
// it reads flux_bnd.  HintFirst puts the hint at the top of the kernel instead
// of leaving it implicit at grid exit, which is what lets the dependent grid
// cover this one rather than only its tail (p127_gap_study.md section 19.1).
template <bool HintFirst>
__global__ void pdl_elembnd_flux_kernel(
    double *__restrict__ flux, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, int nface)
{
  if (HintFirst) {
    asm volatile("griddepcontrol.launch_dependents;");
  }
  const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (idx < nface) {
    const int iM = VMapM[idx] - 1;
    const int iP = VMapP[idx] - 1;
    const double qM = q[iM];
    const double qP = q[iP];
    const double fn1 = normal_fn[idx];
    const double fn2 = normal_fn[idx + nface];
    const double fn3 = normal_fn[idx + 2 * nface];
    const double VelM = u[iM] * fn1 + v[iM] * fn2 + w[iM] * fn3;
    const double VelP = u[iP] * fn1 + v[iP] * fn2 + w[iP] * fn3;
    const double alpha = 0.5 * fabs(VelP + VelM);
    flux[idx] = 0.5 * Fscale[idx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
  }
}

template <bool UseTc>
__global__ __launch_bounds__(P63_THREADS, P63_BPSM) void tendency_fused_p63_xz_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  // Dynamic, because at BK63=32 the three panels are exactly the 48 KB static
  // limit and at BK63=64 they are twice it.  Occupancy is capped by registers
  // (512 threads times 124) long before shared memory matters, so the extra
  // shared costs nothing; see section 16 of p63_gap_study.md.
  constexpr int P63_PANEL = NQ63 * BK63;
  extern __shared__ __align__(16) double smem63[];
  double *const sFU = smem63;
  double *const sD = smem63 + P63_XZ_NBUF * P63_PANEL;
  double *const sFW = smem63 + 2 * P63_XZ_NBUF * P63_PANEL;

  const int elem = (int)blockIdx.x / NQ63;
  if (elem >= Ne) {
    return;
  }
  const int jp = (int)blockIdx.x - elem * NQ63;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp & 3;
  const int wn = warp >> 2;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP63;
  const int efo = elem * NFPTOT63;
  const int npoint = NP63 * Ne;
  const int plane_off = NQ63 * jp;

  // Eight 8x8 tiles per warp in a 2x4 arrangement, for each of the two
  // directions: 2 + 4 operand loads per k-step instead of the 1 + 8 that a
  // 1x8 shape would need.
  double ax[2 * 2 * P63_TN], az[2 * 2 * P63_TN];
#pragma unroll
  for (int e = 0; e < 2 * 2 * P63_TN; ++e) {
    ax[e] = 0.0;
    az[e] = 0.0;
  }

#if P63_PDL_STAGE >= 3
  // Let the y grid start.  Its mma does not read dqdt; the wait is in front of
  // the only load that does, the epilogue's read-modify-write prefetch.
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // sFU[k][l] = q*u at (l, jp, k).  l is fast in global, so sixteen lanes
  // walk l and cover one 128-byte run.  sD[r][l] = D1D(r, l), r fast in the
  // Fortran column-major operator.  sFW[i][l] = q*w at (i, jp, l), i fast in
  // global.
#define P63_XZ_FU_IDX(P) \
  const int ll = tid & (BK63 - 1);                                            \
  const int o = (tid / BK63) + (P63_THREADS / BK63) * (P)
#define P63_XZ_DW_IDX(P) \
  const int o = tid & 63;                                                     \
  const int ll = (tid >> 6) + (P63_THREADS / 64) * (P)

#if P63_XZ_DB
  // Double buffered, the form section 4.1 of p255_gap_study.md measured:
  // issue(k+1) -> mma(buf) -> store(buf^1) -> barrier.  One barrier per chunk
  // instead of two, and the next chunk's global loads are in flight across
  // the whole mma loop.  The values stay raw in registers and the q*u and q*w
  // products happen at the store.
  double pq[P63_STAGE_ITERS], pu[P63_STAGE_ITERS];
  double pd[P63_STAGE_ITERS], pqw[P63_STAGE_ITERS], pw[P63_STAGE_ITERS];
#define P63_XZ_ISSUE(KK)                                                      \
  do {                                                                        \
    _Pragma("unroll") for (int p = 0; p < P63_STAGE_ITERS; ++p)               \
    {                                                                         \
      P63_XZ_FU_IDX(p);                                                       \
      const int g = eo + ((KK) + ll) + plane_off + NQ2_63 * o;                \
      pq[p] = q[g];                                                           \
      pu[p] = u[g];                                                           \
    }                                                                         \
    _Pragma("unroll") for (int p = 0; p < P63_STAGE_ITERS; ++p)               \
    {                                                                         \
      P63_XZ_DW_IDX(p);                                                       \
      pd[p] = D1D[o + NQ63 * ((KK) + ll)];                                    \
      const int g = eo + o + plane_off + NQ2_63 * ((KK) + ll);                \
      pqw[p] = q[g];                                                          \
      pw[p] = w[g];                                                           \
    }                                                                         \
  } while (0)
#define P63_XZ_STORE(BUF)                                                     \
  do {                                                                        \
    _Pragma("unroll") for (int p = 0; p < P63_STAGE_ITERS; ++p)               \
    {                                                                         \
      P63_XZ_FU_IDX(p);                                                       \
      sFU[(BUF) * P63_PANEL + swu63(o + BK63 * ll)] = pq[p] * pu[p];          \
    }                                                                         \
    _Pragma("unroll") for (int p = 0; p < P63_STAGE_ITERS; ++p)               \
    {                                                                         \
      P63_XZ_DW_IDX(p);                                                       \
      sD[(BUF) * P63_PANEL + swt63(o + BK63 * ll)] = pd[p];                   \
      sFW[(BUF) * P63_PANEL + swt63(o + BK63 * ll)] = pqw[p] * pw[p];         \
    }                                                                         \
  } while (0)
  P63_XZ_ISSUE(0);
  P63_XZ_STORE(0);
  __syncthreads();
  int buf = 0;
  for (int kk = 0; kk < NQ63; kk += BK63) {
    const bool more = (kk + BK63) < NQ63;
    if (more) {
      P63_XZ_ISSUE(kk + BK63);
    }
#else
  int buf = 0;
  for (int kk = 0; kk < NQ63; kk += BK63) {
    // The barrier that protects the panels from being overwritten belongs at
    // the head of the body, not at its end: written at the end it also runs
    // after the last chunk, where nothing follows it.  At BK63 = NQ63 the
    // loop runs once and that saves one barrier out of two.
    if (kk) {
      __syncthreads();
    }
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      P63_XZ_FU_IDX(p);
      const int g = eo + (kk + ll) + plane_off + NQ2_63 * o;
      sFU[swu63(o + BK63 * ll)] = q[g] * u[g];
    }
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      P63_XZ_DW_IDX(p);
      sD[swt63(o + BK63 * ll)] = D1D[o + NQ63 * (kk + ll)];
      const int g = eo + o + plane_off + NQ2_63 * (kk + ll);
      sFW[swt63(o + BK63 * ll)] = q[g] * w[g];
    }
    __syncthreads();
#endif

#pragma unroll
    for (int ks = 0; ks < BK63 / 4; ++ks) {
      const int l = 4 * ks + colk;
      double av[2], bv[P63_TN], avz[2], bvz[P63_TN];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int m = 8 * (2 * wm + a) + row;
        av[a] = sFU[buf * P63_PANEL + swu63(m + BK63 * l)];
        avz[a] = sD[buf * P63_PANEL + swt63(m + BK63 * l)];
      }
#pragma unroll
      for (int bb = 0; bb < P63_TN; ++bb) {
        const int n = 8 * (P63_TN * wn + bb) + row;
        bv[bb] = sD[buf * P63_PANEL + swt63(n + BK63 * l)];
        bvz[bb] = sFW[buf * P63_PANEL + swt63(n + BK63 * l)];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < P63_TN; ++bb) {
          const int e = 2 * (P63_TN * a + bb);
          mma_m8n8k4_f64<UseTc>(ax[e], ax[e + 1], av[a], bv[bb], ax[e], ax[e + 1]);
          mma_m8n8k4_f64<UseTc>(az[e], az[e + 1], avz[a], bvz[bb], az[e], az[e + 1]);
        }
      }
    }
#if P63_XZ_DB
    if (more) {
      P63_XZ_STORE(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
#endif
  }
#undef P63_XZ_FU_IDX
#undef P63_XZ_DW_IDX
#if P63_XZ_DB
#undef P63_XZ_ISSUE
#undef P63_XZ_STORE
#endif

#if P63_PDL_STAGE >= 1
  // The epilogue below is the first thing that reads flux_bnd, so this is the
  // latest point the face grid's stores have to be visible.
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  // j is block uniform, so the faces 1 and 3 coefficients and the whole (i,j)
  // face contribution are loop invariant.
  const double lf1 = Lift1D[jp];
  const double lf3 = Lift1D[jp + 2 * NQ63];

#pragma unroll
  for (int e8 = 0; e8 < 2 * P63_TN; ++e8) {
    const int a = e8 / P63_TN;
    const int bb = e8 % P63_TN;
    const int m = 8 * (2 * wm + a) + row;                 // k
    const int n = 8 * (P63_TN * wn + bb) + 2 * colk;      // i, and i+1
    const int node = eo + n + plane_off + NQ2_63 * m;

    const double2 ex = *reinterpret_cast<const double2 *>(Escale + node);
    const double2 ez =
        *reinterpret_cast<const double2 *>(Escale + node + 2 * npoint);

    // Faces 1 and 3 are (i,k) planes, so the pair is one aligned double2 and
    // the coefficient is shared.  Faces 2 and 4 are (j,k) planes, so the pair
    // shares the flux value and the coefficient varies in i.  Faces 5 and 6
    // are (i,j) planes, so the pair is a double2 again.
    const int fp13 = n + NQ63 * m;
    const double2 fb1 = *reinterpret_cast<const double2 *>(flux_bnd + efo + fp13);
    const double2 fb3 =
        *reinterpret_cast<const double2 *>(flux_bnd + efo + 2 * NQ2_63 + fp13);
    const int fp24 = jp + NQ63 * m;
    const double fb2 = flux_bnd[efo + NQ2_63 + fp24];
    const double fb4 = flux_bnd[efo + 3 * NQ2_63 + fp24];
    const int fp56 = n + plane_off;
    const double2 fb5 =
        *reinterpret_cast<const double2 *>(flux_bnd + efo + 4 * NQ2_63 + fp56);
    const double2 fb6 =
        *reinterpret_cast<const double2 *>(flux_bnd + efo + 5 * NQ2_63 + fp56);
    const double lf2a = Lift1D[n + NQ63];
    const double lf2b = Lift1D[n + 1 + NQ63];
    const double lf4a = Lift1D[n + 3 * NQ63];
    const double lf4b = Lift1D[n + 1 + 3 * NQ63];
    const double lf5 = Lift1D[m + 4 * NQ63];
    const double lf6 = Lift1D[m + 5 * NQ63];

    // Same summation order as tendency_fused_p63_xz_kernel.
    *reinterpret_cast<double2 *>(dqdt + node) = make_double2(
        -(ex.x * ax[2 * e8] + ez.x * az[2 * e8] + lf1 * fb1.x + lf2a * fb2 +
          lf3 * fb3.x + lf4a * fb4 + lf5 * fb5.x + lf6 * fb6.x),
        -(ex.y * ax[2 * e8 + 1] + ez.y * az[2 * e8 + 1] + lf1 * fb1.y +
          lf2b * fb2 + lf3 * fb3.y + lf4b * fb4 + lf5 * fb5.y + lf6 * fb6.y));
  }
}

//> p=63 y volume term, accumulated onto what the xz kernel wrote.
//
// One block per (element, k plane).  C[m=j][n=i] = sum_l D[j][l] * FV[i][l],
// which is the z contraction of the first kernel with (i,j) in place of (i,k),
// so the same two panel shapes and the same swizzle serve it.
template <bool UseTc>
__global__ __launch_bounds__(P63Y_THREADS, P63Y_BPSM) void tendency_fused_p63_y_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  extern __shared__ __align__(16) double smem63[];
  double *const sD = smem63;
  double *const sFV = smem63 + NQ63 * BK63;
  // The dqdt tile the epilogue reads back, 64x64 doubles addressed exactly as
  // the plane is in global, so a thread reads back the same slot it asked for
  // and no barrier is needed.
  double *const sDQ = smem63 + 2 * NQ63 * BK63;

  const int elem = (int)blockIdx.x / NQ63;
  if (elem >= Ne) {
    return;
  }
  const int kp = (int)blockIdx.x - elem * NQ63;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp & 3;
  const int wn = warp >> 2;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP63;
  const int npoint = NP63 * Ne;
  const int plane_off = NQ2_63 * kp;

  double acc[2 * 2 * P63Y_TN];
#pragma unroll
  for (int e = 0; e < 2 * 2 * P63Y_TN; ++e) {
    acc[e] = 0.0;
  }

  // Ask for the four dqdt pairs this thread will need before the first mma.
  // The epilogue's read-modify-write is the only load here the mma cannot
  // cover: with it in place the y kernel stalls on long_scoreboard 2.54 of 32
  // warps, and prefetching it takes that to 0.34.  Section 19.4.
  // The whole plane is 4096 doubles at eo + plane_off and the 512 threads
  // cover it exactly, four 16-byte slots each.
#if P63_PDL_STAGE >= 3
  // dqdt is what the xz grid writes, and the prefetch below is this kernel's
  // only read of it, so this is the latest point the wait can sit.
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
#pragma unroll
  for (int e8 = 0; e8 < 2 * P63Y_TN; ++e8) {
    const int a = e8 / P63Y_TN;
    const int bb = e8 % P63Y_TN;
    const int loc = (8 * (P63Y_TN * wn + bb) + 2 * colk) +
                    NQ63 * (8 * (2 * wm + a) + row);
    cp_async_16(sDQ + loc, dqdt + eo + plane_off + loc);
  }
  asm volatile("cp.async.commit_group;\n" ::);

  for (int kk = 0; kk < NQ63; kk += BK63) {
    if (kk) {
      __syncthreads();
    }
#pragma unroll
    for (int p = 0; p < P63Y_STAGE_ITERS; ++p) {
      const int o = tid & 63;
      const int ll = (tid >> 6) + (P63Y_THREADS / 64) * p;
      sD[swt63(o + BK63 * ll)] = D1D[o + NQ63 * (kk + ll)];
      const int g = eo + o + NQ63 * (kk + ll) + plane_off;
      sFV[sw63(ll + BK63 * o)] = q[g] * v[g];
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK63 / 4; ++ks) {
      const int l = 4 * ks + colk;
      double av[2], bv[P63Y_TN];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        av[a] = sD[swt63((8 * (2 * wm + a) + row) + BK63 * l)];
      }
#pragma unroll
      for (int bb = 0; bb < P63Y_TN; ++bb) {
        bv[bb] = sFV[sw63(l + BK63 * (8 * (P63Y_TN * wn + bb) + row))];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < P63Y_TN; ++bb) {
          const int e = 2 * (P63Y_TN * a + bb);
          mma_m8n8k4_f64<UseTc>(acc[e], acc[e + 1], av[a], bv[bb], acc[e], acc[e + 1]);
        }
      }
    }
  }

  asm volatile("cp.async.wait_group 0;\n" ::);

#pragma unroll
  for (int e8 = 0; e8 < 2 * P63Y_TN; ++e8) {
    const int a = e8 / P63Y_TN;
    const int bb = e8 % P63Y_TN;
    const int m = 8 * (2 * wm + a) + row;                 // j
    const int n = 8 * (P63Y_TN * wn + bb) + 2 * colk;      // i, and i+1
    const int node = eo + n + NQ63 * m + plane_off;
    const double2 ey =
        *reinterpret_cast<const double2 *>(Escale + node + npoint);
    double2 out = *reinterpret_cast<const double2 *>(sDQ + n + NQ63 * m);
    out.x -= ey.x * acc[2 * e8];
    out.y -= ey.y * acc[2 * e8 + 1];
    *reinterpret_cast<double2 *>(dqdt + node) = out;
  }
}

template <bool UseTc>
void launch_tendency_fused_p63_impl(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, double *flux_bnd,
    const double *Escale, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface, int Ne)
{
  const int nblock = NQ63 * Ne;
  constexpr int FACE_THREADS = 256;
  const int nblock_face = (nface + FACE_THREADS - 1) / FACE_THREADS;
  const size_t smem_xz =
      3 * P63_XZ_NBUF * NQ63 * BK63 * sizeof(double);
  const size_t smem_y = (2 * NQ63 * BK63 + NQ2_63) * sizeof(double);
  static bool opted_in = false;
  if (!opted_in) {
    cudaFuncSetAttribute(tendency_fused_p63_xz_kernel<UseTc>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_xz);
    cudaFuncSetAttribute(tendency_fused_p63_y_kernel<UseTc>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_y);
    opted_in = true;
  }
#if P63_PDL_STAGE >= 1
  constexpr bool HINT_FIRST = (P63_PDL_STAGE >= 2);
  {
    cudaLaunchConfig_t fcfg = {};
    cudaLaunchAttribute fattr[1];
    fcfg.gridDim = dim3(nblock_face);
    fcfg.blockDim = dim3(FACE_THREADS);
    fcfg.dynamicSmemBytes = 0;
    fcfg.stream = dg_cuda_stream;
    fattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    fattr[0].val.programmaticStreamSerializationAllowed = 1;
    fcfg.attrs = fattr;
    fcfg.numAttrs = 1;
    cudaLaunchKernelEx(&fcfg, pdl_elembnd_flux_kernel<HINT_FIRST>, flux_bnd, q,
                       u, v, w, VMapM, VMapP, normal_fn, Fscale, nface);
  }
  {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    cfg.gridDim = dim3(nblock);
    cfg.blockDim = dim3(P63_THREADS);
    cfg.dynamicSmemBytes = (unsigned)smem_xz;
    cfg.stream = dg_cuda_stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, tendency_fused_p63_xz_kernel<UseTc>, dqdt, D1D,
                       Lift1D, q, u, w, flux_bnd, Escale, Ne);
  }
#else
  pdl_elembnd_flux_kernel<false>
      <<<nblock_face, FACE_THREADS, 0, dg_cuda_stream>>>(
          flux_bnd, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, nface);
  tendency_fused_p63_xz_kernel<UseTc>
      <<<nblock, P63_THREADS, smem_xz, dg_cuda_stream>>>(
          dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
#endif
#if P63_PDL_STAGE >= 3
  {
    cudaLaunchConfig_t ycfg = {};
    cudaLaunchAttribute yattr[1];
    ycfg.gridDim = dim3(nblock);
    ycfg.blockDim = dim3(P63Y_THREADS);
    ycfg.dynamicSmemBytes = (unsigned)smem_y;
    ycfg.stream = dg_cuda_stream;
    yattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    yattr[0].val.programmaticStreamSerializationAllowed = 1;
    ycfg.attrs = yattr;
    ycfg.numAttrs = 1;
    cudaLaunchKernelEx(&ycfg, tendency_fused_p63_y_kernel<UseTc>, dqdt, D1D, q,
                       v, Escale, Ne);
  }
#else
  tendency_fused_p63_y_kernel<UseTc>
      <<<nblock, P63Y_THREADS, smem_y, dg_cuda_stream>>>(dqdt, D1D, q, v,
                                                         Escale, Ne);
#endif
  check_cuda("tendency_fused_p63 kernels");
}

extern "C" void launch_tendency_fused_p63_dfma(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, double *flux_bnd,
    const double *Escale, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface, int Ne)
{
  launch_tendency_fused_p63_impl<false>(dqdt, D1D, Lift1D, q, u, v, w, flux_bnd,
                                        Escale, VMapM, VMapP, normal_fn, Fscale,
                                        nface, Ne);
}

extern "C" void launch_tendency_fused_p63_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, double *flux_bnd,
    const double *Escale, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface, int Ne)
{
  launch_tendency_fused_p63_impl<true>(dqdt, D1D, Lift1D, q, u, v, w, flux_bnd,
                                       Escale, VMapM, VMapP, normal_fn, Fscale,
                                       nface, Ne);
}

//============================================================================
// p=127 (Nq=128) fused Tensor Core tendency
//============================================================================
//
// Same three-kernel shape as p=63: elembnd_flux_kernel evaluates the six face
// fluxes once, then an xz kernel writes dqdt and a y kernel accumulates onto
// it.  A 128x128 output plane cannot be one block, so both volume kernels
// tile it 128x64 -- the whole contraction index m, half the output index n --
// and run two blocks per plane.  The xz kernel is 512 threads in an 8 by 2
// warp grid with 2x4 mma tiles per warp and a double-buffered chunk loop; the
// y kernel is 512 threads in a 4 by 4 grid with 4x2 tiles and a cp.async
// ping-pong.  The xz kernel was 1024 threads with 2x2 tiles and a single
// buffered loop until section 23; the two changes only pay together.
//
//   x: C[m=k][n=i] = sum_l FU[k][l] * D[i][l]      A = sFU, B = sD
//   z: C[m=k][n=i] = sum_l D[k][l]  * FW[i][l]     A = sD,  B = sFW
//   y: C[m=j][n=i] = sum_l D[j][l] * FV[i][l]      A = sD,  B = sFV
//
// Keeping m over the whole 128 is what makes one sD panel serve both operands
// of the xz kernel, exactly as at p=63; a 64x64 tile splits the operator rows
// (x wants rows i, z wants rows k) and needs two panels.  It also gives the
// warp 2x2 tiles instead of 2x1, which is a third fewer shared loads per unit
// of mma -- and the mma loop is what this kernel spends its time in, so that
// is the ratio that matters.  A 64x64 tile with the same 1024 threads was
// measured at 784.2 us/stage against 757 for this one.
//
// The panels are not staged alike.  q, u, v and w are 134 MB per field and
// miss to DRAM; D1D is 128 KB and every block reads all of it, so it is L2
// resident.  Each kernel therefore keeps its n-indexed flux panel -- sFW or
// sFV, the one it reads exactly once per plane -- in shared at the full
// contraction depth, staged before any mma, and chunks only the panels that
// are re-read anyway (sFU) or come from L2 (sD).  That is what a cp.async
// prologue would buy, and cp.async cannot be used here at all, because the
// panels hold the product q*u rather than a copy of anything.  Chunking the
// resident panel instead of sFU was measured and loses badly (834.3).
//
// Two things that did not work, both measured, both instructive:
//
//   - Staging the block-uniform epilogue data (the four Lift1D slices and the
//     four face planes that are constant along one tile index, 4 KB) into
//     shared before the mma loop, the way the p=7 kernel stages sLift: 844.6
//     against 784.2.  The note on sLift above says why -- at one block per SM
//     the epilogue is bound by L1/TEX, which serves shared and global alike.
//   - Moving the y kernel's read-modify-write of dqdt onto the xz kernel,
//     which has twice the mma to hide it behind: 806.8 against 784.0, the
//     same answer p=63 section 16.6 got.

// Depth of the chunked panels.  The first version chunked every panel
// together, the p=63 arrangement, and deeper was monotonically better (999.0
// / 934.7 / 866.2 us/stage at 16 / 32 / 64 on the 64x64 tile) for the reason
// p=63 section 16 gives: with no prefetch the loads of chunk k+1 are not
// issued until the mma of chunk k has consumed chunk k.  While the loop was
// single buffered that made deep chunks the right answer (794.9 at 32 against
// 757.2 at 64, and 128 does not fit in the 227 KB a Blackwell block holds).
// With the double-buffered loop the sign flips, because the loads of chunk
// k+1 are issued before the mma of chunk k and a shallow chunk buys register
// budget and shared memory instead: 16 is 4.3% faster than 64 was.  See
// section 23 of p127_gap_study.md for the sweep and for the three-point
// measurement that shows the warp shape and the pipelining only pay together.
//
// The register budget is what ties the two together.  Going from 2x2 to 2x4
// mma tiles per warp cuts the operand loads per unit of mma from 1.0 to 0.75
// (ncu: shared load instructions 16.78 M -> 12.58 M), and it needs half the
// warps, so a lane holds 32 accumulator doubles instead of 16.  At 1024
// threads that is impossible -- 1024 x 64 registers is the whole register
// file, and the 2x2 shape already spilled 32 bytes -- so the wider tile is
// only reachable at 512 threads, where the budget is 128 registers a thread
// and the prefetch registers of the double-buffered loop fit alongside the
// accumulators with nothing spilled.

// The panels are 64 outer in the tile index, exactly the p=63 shape, so the
// swizzles carry over unchanged.  swt127 is for outer-fast panels (idx =
// outer + 64*l): the read has the outer index in bits 0-2 and l in bits 6-11,
// so l's low two bits fold into bits 3-4 -- the fold has to land inside the
// low four bits because an 8-byte shared access is serviced in half-warp
// phases of 16 lanes.  sw127 is for the one l-fast panel (idx = l + 64*outer).
__device__ __forceinline__ int swt127(int idx)
{
  return idx ^ (((idx >> 6) & 3) << 2);
}

// Outer-fast panel with 128 rows: idx = outer + 128*l, so l's low two bits
// sit in bits 7-8 and fold into bits 3-4 rather than 2-3.
__device__ __forceinline__ int swt128(int idx)
{
  return idx ^ (((idx >> 7) & 3) << 2);
}


__device__ __forceinline__ int sw127(int idx)
{
  return idx ^ (((idx / NQ127) & 7) << 2);
}

// The xz warp shape was swept three times.  With the first arrangement -- every
// panel chunked together, a 64x64 tile, four blocks per plane -- occupancy
// decided it (Ne=2**3, nstep=400):
//
//   grid        threads  blocking  reg  occupancy  us/stage
//   4x2 warps       256       2x4  128      12.5%    1369.4
//   4x4 warps       512       2x2  128      25.0%    1138.8
//   4x8 warps      1024       2x1   64      50.0%     866.2
//
// After the staging split below the same sweep is nearly flat -- 817.5 /
// 796.6 / 783.6 -- because the load latency the extra warps were hiding is
// no longer there.  What is left is the operand-load ratio, and that is what
// the 128x64 tile buys: 2x2 tiles per warp costs 2+2 operand loads per k-step
// for four mma tiles against the 2+1 for two of a 2x1 shape, so a third fewer
// shared loads per unit of mma, at the same 50% occupancy.
//
// The third sweep (section 23) went the other way, giving up occupancy for a
// wider warp tile once the chunk loop was pipelined.  Interleaved on an
// occupied GPU, twelve rounds each:
//
//   TM x TN  threads  BKD  buffers  regs  spill  us/stage
//     2x2       1024   64        1    64    32 B    708.0  <- was kept
//     2x4        512   64        1   128     8 B    704.1
//     2x2       1024   16        2    64    40 B    705.8
//     2x2       1024   32        2    64   104 B    736.4
//     4x2        512   16        2   128     0      693.0
//     2x4        512    8        2   128     0      683.3
//     2x4        512   16        2   128     0      677.9  <- kept
//
// Neither change alone is worth more than half a percent and one form of the
// pipelining alone is 4% slower; together they are 4.25%.

// The warp shape, the chunk depth and whether the chunk loop is double
// buffered are the three knobs P127_XZ_TM / P127_XZ_TN / BKD127 / P127_XZ_DB
// in fused_kernel_geom.h.  The block tile stays 128x64 in every setting, so
// the warp grid and the thread count follow from TM and TN:
//
//   TM  TN  warps      threads  acc doubles  regs/thread available
//    2   2  8 x 4         1024           16                     64
//    4   2  4 x 4          512           32                    128
//    4   4  4 x 2          256           64                    255
//
// The accumulator count is twice what a one-direction kernel would carry,
// because this kernel holds x and z at once.  See section 23 of
// p127_gap_study.md for the sweep.
template <bool UseTc>
__global__ __launch_bounds__(P127_XZ_THREADS, P127_XZ_MINB) void tendency_fused_p127_xz_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  constexpr int TM = P127_XZ_TM;
  constexpr int TN = P127_XZ_TN;
  constexpr int THREADS = P127_XZ_THREADS;
  constexpr int NBUF = P127_XZ_NBUF;
  constexpr int PANEL = NQ127 * BKD127;
  constexpr int NSTAGE = PANEL / THREADS;

  extern __shared__ __align__(16) double smem127[];
  double *const sFW = smem127;                       //  64 x NQ, full depth
  double *const sFU = smem127 + P127_MT * NQ127;     // NBUF x 128 x BKD
  double *const sD = sFU + NBUF * PANEL;             // NBUF x 128 x BKD

  const int block = (int)blockIdx.x;
  const int ntile = block & 1;
  const int jp = (block >> 1) & (NQ127 - 1);
  const int elem = block >> 8;
  if (elem >= Ne) {
    return;
  }
  const int ibase = ntile * P127_MT;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp % P127_XZ_WM;
  const int wn = warp / P127_XZ_WM;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP127;
  const int efo = elem * NFPTOT127;
  const int npoint = NP127 * Ne;
  const int plane_off = NQ127 * jp;

  double ax[2 * TM * TN], az[2 * TM * TN];
#pragma unroll
  for (int e = 0; e < 2 * TM * TN; ++e) {
    ax[e] = 0.0;
    az[e] = 0.0;
  }

  // The swizzle collapses inside the mma loop.  Every operand address there
  // has the form row_index + stride*l with l = 4*ks + colk, so the field the
  // swizzle folds -- bits 6-7 or 7-8 of the address, which is l's low two
  // bits -- is colk, a per-lane constant.  The fold therefore reduces to one
  // XOR of the row index by colk*4, loop invariant, and what is left is a
  // base plus a compile-time offset once the loop is unrolled.  Written in
  // the general swt127/swt128 form instead, ptxas emits four integer ops per
  // operand load; ncu (Slurm 62173) measured 122.5 M instructions against
  // 16.8 M mma, 66% of them address arithmetic, with the DMMA pipe starved at
  // 53.1%.  See section 11.10 of p127_gap_study.md.
  const int cx = colk << 2;
  int amx[TM], bnx[TN];
#pragma unroll
  for (int a = 0; a < TM; ++a) {
    amx[a] = (8 * (TM * wm + a) + row) ^ cx;
  }
#pragma unroll
  for (int bb = 0; bb < TN; ++bb) {
    bnx[bb] = (8 * (TN * wn + bb) + row) ^ cx;
  }

  // sFW[i][l] = q*w at (ibase+i, jp, l), the panel this tile reads exactly
  // once per plane, so it is the one staged at full depth.
#pragma unroll
  for (int p = 0; p < P127_MT * NQ127 / THREADS; ++p) {
    const int o = tid & (P127_MT - 1);
    const int ll = (tid / P127_MT) + (THREADS / P127_MT) * p;
    const int g = eo + (ibase + o) + plane_off + NQ2_127 * ll;
    sFW[swt127(o + P127_MT * ll)] = q[g] * w[g];
  }
  // Let the y grid start: its mma does not read dqdt.  The wait is in the y
  // epilogue, after this grid's stores.
  asm volatile("griddepcontrol.launch_dependents;");

  // sFU[k][l] = q*u at (kk+l, jp, k), all 128 k.  l is fast in global.
  // sD[r][l] = D1D(r, kk+l), all 128 rows: m runs over the whole range and
  // n sits inside it, so one panel serves both operands as it did at p=63.
#define P127_XZ_FU_IDX(P) \
  const int ll = tid & (BKD127 - 1);                                          \
  const int o = (tid / BKD127) + (THREADS / BKD127) * (P)
#define P127_XZ_D_IDX(P) \
  const int o = tid & (NQ127 - 1);                                            \
  const int ll = (tid / NQ127) + (THREADS / NQ127) * (P)

#if P127_XZ_DB
  // Double buffered: issue(k+1) -> mma(buf) -> store(buf^1) -> barrier.  One
  // barrier per chunk instead of two, and the next chunk's global loads are
  // in flight across the whole mma loop.  The loaded values stay raw in
  // registers and the q*u multiply happens at the store, because multiplying
  // at issue time would make the pipeline wait on the loads exactly where it
  // is trying not to.  Same shape as tendency_p255_kernel.
  double pq[NSTAGE], pu[NSTAGE], pd[NSTAGE];
#define P127_XZ_ISSUE(KK)                                                     \
  do {                                                                        \
    _Pragma("unroll") for (int p = 0; p < NSTAGE; ++p)                        \
    {                                                                         \
      P127_XZ_FU_IDX(p);                                                      \
      const int g = eo + ((KK) + ll) + plane_off + NQ2_127 * o;               \
      pq[p] = q[g];                                                           \
      pu[p] = u[g];                                                           \
    }                                                                         \
    _Pragma("unroll") for (int p = 0; p < NSTAGE; ++p)                        \
    {                                                                         \
      P127_XZ_D_IDX(p);                                                       \
      pd[p] = D1D[o + NQ127 * ((KK) + ll)];                                   \
    }                                                                         \
  } while (0)
#define P127_XZ_STORE(BUF)                                                    \
  do {                                                                        \
    _Pragma("unroll") for (int p = 0; p < NSTAGE; ++p)                        \
    {                                                                         \
      P127_XZ_FU_IDX(p);                                                      \
      sFU[(BUF) * PANEL + swt128(o + NQ127 * ll)] = pq[p] * pu[p];            \
    }                                                                         \
    _Pragma("unroll") for (int p = 0; p < NSTAGE; ++p)                        \
    {                                                                         \
      P127_XZ_D_IDX(p);                                                       \
      sD[(BUF) * PANEL + swt128(o + NQ127 * ll)] = pd[p];                     \
    }                                                                         \
  } while (0)

  P127_XZ_ISSUE(0);
  P127_XZ_STORE(0);
  __syncthreads();
  int buf = 0;
  for (int kk = 0; kk < NQ127; kk += BKD127) {
    const bool more = (kk + BKD127) < NQ127;
    if (more) {
      P127_XZ_ISSUE(kk + BKD127);
    }
#else
  int buf = 0;
  for (int kk = 0; kk < NQ127; kk += BKD127) {
    // The barrier that protects the panels from being overwritten belongs
    // here, not at the end of the body: written at the end it also runs after
    // the last chunk, where nothing follows it.  Three barriers per block
    // instead of four.  ncu (Slurm 62193) put 9.83 of the 32 warps in the
    // barrier stall once the swizzle collapse of section 11.10 removed the
    // integer work that used to cover it.
    if (kk) {
      __syncthreads();
    }
#pragma unroll
    for (int p = 0; p < NSTAGE; ++p) {
      P127_XZ_FU_IDX(p);
      const int g = eo + (kk + ll) + plane_off + NQ2_127 * o;
      sFU[swt128(o + NQ127 * ll)] = q[g] * u[g];
    }
#pragma unroll
    for (int p = 0; p < NSTAGE; ++p) {
      P127_XZ_D_IDX(p);
      sD[swt128(o + NQ127 * ll)] = D1D[o + NQ127 * (kk + ll)];
    }
    __syncthreads();
#endif

#pragma unroll
    for (int ks = 0; ks < BKD127 / 4; ++ks) {
      const int lc = 4 * ks + colk;
      const int lg = kk + lc;
      double av[TM], bv[TN], avz[TM], bvz[TN];
#pragma unroll
      for (int a = 0; a < TM; ++a) {
        av[a] = sFU[buf * PANEL + amx[a] + NQ127 * lc];
        avz[a] = sD[buf * PANEL + amx[a] + NQ127 * lc];
      }
#pragma unroll
      for (int bb = 0; bb < TN; ++bb) {
        bv[bb] = sD[buf * PANEL + ibase + bnx[bb] + NQ127 * lc];
        bvz[bb] = sFW[bnx[bb] + P127_MT * lg];
      }
#pragma unroll
      for (int a = 0; a < TM; ++a) {
#pragma unroll
        for (int bb = 0; bb < TN; ++bb) {
          const int e = 2 * (TN * a + bb);
          mma_m8n8k4_f64<UseTc>(ax[e], ax[e + 1], av[a], bv[bb], ax[e], ax[e + 1]);
          mma_m8n8k4_f64<UseTc>(az[e], az[e + 1], avz[a], bvz[bb], az[e], az[e + 1]);
        }
      }
    }
#if P127_XZ_DB
    if (more) {
      P127_XZ_STORE(buf ^ 1);
      __syncthreads();
      buf ^= 1;
    }
#endif
  }
#undef P127_XZ_FU_IDX
#undef P127_XZ_D_IDX
#if P127_XZ_DB
#undef P127_XZ_ISSUE
#undef P127_XZ_STORE
#endif

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // The epilogue is nested b-outer / a-inner in the sense of section 4.4 of
  // p255_gap_study.md: half of what it reads depends on one tile index only.
  // Faces 2 and 4 and their lift coefficients are indexed by m, faces 5 and 6
  // and the faces 2/4 coefficients by n, and faces 1 and 3 by both.  Written
  // flat as below, ptxas already emits exactly the hoisted count -- 20 LDG.128
  // and 18 LDG.64 per warp against the 56 loads a reload-per-tile schedule
  // would need -- because with TM*TN <= 8 tiles the values stay live.  See
  // section 23.1.
  const double lf1 = Lift1D[jp];
  const double lf3 = Lift1D[jp + 2 * NQ127];

#pragma unroll
  for (int e8 = 0; e8 < TM * TN; ++e8) {
    const int a = e8 / TN;
    const int bb = e8 % TN;
    const int m = 8 * (TM * wm + a) + row;                 // k
    const int n = ibase + 8 * (TN * wn + bb) + 2 * colk;   // i, and i+1
    const int node = eo + n + plane_off + NQ2_127 * m;

    const double2 ex = *reinterpret_cast<const double2 *>(Escale + node);
    const double2 ez =
        *reinterpret_cast<const double2 *>(Escale + node + 2 * npoint);

    const int fp13 = n + NQ127 * m;
    const double2 fb1 = *reinterpret_cast<const double2 *>(flux_bnd + efo + fp13);
    const double2 fb3 =
        *reinterpret_cast<const double2 *>(flux_bnd + efo + 2 * NQ2_127 + fp13);
    const int fp24 = jp + NQ127 * m;
    const double fb2 = flux_bnd[efo + NQ2_127 + fp24];
    const double fb4 = flux_bnd[efo + 3 * NQ2_127 + fp24];
    const int fp56 = n + plane_off;
    const double2 fb5 =
        *reinterpret_cast<const double2 *>(flux_bnd + efo + 4 * NQ2_127 + fp56);
    const double2 fb6 =
        *reinterpret_cast<const double2 *>(flux_bnd + efo + 5 * NQ2_127 + fp56);
    // n is even, so the faces 2 and 4 coefficient pairs could be read as one
    // aligned double2 each, the way p255_epilogue reads them.  Measured: it
    // takes the four LDG.64 to two LDG.128 and is worth -0.76% on the 1024
    // thread 2x2 shape but +0.58% on the shape below, whose ranges do not
    // overlap.  Section 23.4.
    const double lf2a = Lift1D[n + NQ127];
    const double lf2b = Lift1D[n + 1 + NQ127];
    const double lf4a = Lift1D[n + 3 * NQ127];
    const double lf4b = Lift1D[n + 1 + 3 * NQ127];
    const double lf5 = Lift1D[m + 4 * NQ127];
    const double lf6 = Lift1D[m + 5 * NQ127];

    // Same summation order as tendency_fused_p127_xz_kernel.
    *reinterpret_cast<double2 *>(dqdt + node) = make_double2(
        -(ex.x * ax[2 * e8] + ez.x * az[2 * e8] + lf1 * fb1.x + lf2a * fb2 +
          lf3 * fb3.x + lf4a * fb4 + lf5 * fb5.x + lf6 * fb6.x),
        -(ex.y * ax[2 * e8 + 1] + ez.y * az[2 * e8 + 1] + lf1 * fb1.y +
          lf2b * fb2 + lf3 * fb3.y + lf4b * fb4 + lf5 * fb5.y + lf6 * fb6.y));
  }
}

//> p=127 y volume term, accumulated onto what the xz kernel wrote.
//
// One block per (element, k plane, j tile, i tile).
// C[m=j][n=i] = sum_l D[j][l] * FV[i][l], the z contraction of the first
// kernel with (i,j) in place of (i,k), so only the A panel of the two is
// needed and the block holds 64 KB.
template <bool UseTc>
__global__ __launch_bounds__(P127_Y_THREADS, P127_Y_BPSM) void tendency_fused_p127_y_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  extern __shared__ __align__(16) double smem127[];
  double *const sDm = smem127;
  double *const sFV = smem127 + NQ127 * BKDY127;

  const int block = (int)blockIdx.x;
  const int ntile = block & 1;
  const int kp = (block >> 1) & (NQ127 - 1);
  const int elem = block >> 8;
  if (elem >= Ne) {
    return;
  }
  const int ibase = ntile * P127_MT;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // 16 warps in a 4 x 4 grid with 4x2 mma tiles.  512 threads x 64 registers
  // and 96 KB shared both allow 2 blocks/SM, which the 1024-thread 128 KB
  // shape could not.  ncu job 70955: Block Limit Registers = Shared Mem = 2.
  const int wm = warp % P127_Y_WM;
  const int wn = warp / P127_Y_WM;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP127;
  const int npoint = NP127 * Ne;
  const int plane_off = NQ2_127 * kp;

  constexpr int TM = P127_Y_TM;
  constexpr int TN = P127_Y_TN;
  double acc[2 * TM * TN];
#pragma unroll
  for (int e = 0; e < 2 * TM * TN; ++e) {
    acc[e] = 0.0;
  }

  const int cx = colk << 2;
  const int rx = row << 2;
  int amx[TM], boff[TN];
#pragma unroll
  for (int a = 0; a < TM; ++a) {
    amx[a] = (8 * (TM * wm + a) + row) ^ cx;
  }
#pragma unroll
  for (int bb = 0; bb < TN; ++bb) {
    boff[bb] = NQ127 * (8 * (TN * wn + bb) + row);
  }

  // sFV[i][l] = q*v at (ibase+i, l, kp), staged once at the full depth.
#pragma unroll
  for (int p = 0; p < P127_Y_FSTAGE_ITERS; ++p) {
    const int o = tid & (P127_MT - 1);
    const int ll = (tid / P127_MT) + (P127_Y_THREADS / P127_MT) * p;
    const int g = eo + (ibase + o) + NQ127 * ll + plane_off;
    sFV[sw127(ll + NQ127 * o)] = q[g] * v[g];
  }

  // Ping-pong the 32-deep sDm as two 16-deep halves.  D1D is a copy, so
  // cp.async issues the next half and the mma of the current half covers it
  // (the register-path version stored the next half before the mma).
  constexpr int HALF = BKDY127 / 2;
#define P127_Y_STAGE_D(KK, BUF)                                               \
  do {                                                                        \
    const int lbase = (BUF) * HALF;                                           \
    _Pragma("unroll") for (int p = 0;                                         \
                           p < HALF * NQ127 / (2 * P127_Y_THREADS); ++p)      \
    {                                                                         \
      const int o = (tid & 63) * 2;                                           \
      const int ll = (tid / 64) + (P127_Y_THREADS / 64) * p;                  \
      cp_async_16(&sDm[swt128(o + NQ127 * (lbase + ll))],                     \
                  D1D + o + NQ127 * ((KK) + ll));                             \
    }                                                                         \
    asm volatile("cp.async.commit_group;\n" ::);                              \
  } while (0)
  P127_Y_STAGE_D(0, 0);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  int cur = 0;
  for (int kk = 0; kk < NQ127; kk += HALF) {
    const bool more = (kk + HALF) < NQ127;
    if (more) {
      P127_Y_STAGE_D(kk + HALF, cur ^ 1);
    }
    const int lbase = cur * HALF;
#pragma unroll
    for (int ks = 0; ks < HALF / 4; ++ks) {
      const int lc = 4 * ks + colk;
      const int lg = kk + lc;
      double av[TM], bv[TN];
#pragma unroll
      for (int a = 0; a < TM; ++a) {
        av[a] = sDm[amx[a] + NQ127 * (lbase + lc)];
      }
#pragma unroll
      for (int bb = 0; bb < TN; ++bb) {
        bv[bb] = sFV[(lg ^ rx) + boff[bb]];
      }
#pragma unroll
      for (int a = 0; a < TM; ++a) {
#pragma unroll
        for (int bb = 0; bb < TN; ++bb) {
          const int e = 2 * (TN * a + bb);
          mma_m8n8k4_f64<UseTc>(acc[e], acc[e + 1], av[a], bv[bb], acc[e], acc[e + 1]);
        }
      }
    }
    if (more) {
      asm volatile("cp.async.wait_group 0;\n" ::);
      __syncthreads();
      cur ^= 1;
    }
  }
#undef P127_Y_STAGE_D

  asm volatile("griddepcontrol.wait;" ::: "memory");

#pragma unroll
  for (int e8 = 0; e8 < TM * TN; ++e8) {
    const int a = e8 / TN;
    const int bb = e8 % TN;
    const int m = 8 * (TM * wm + a) + row;
    const int n = ibase + 8 * (TN * wn + bb) + 2 * colk;
    const int node = eo + n + NQ127 * m + plane_off;
    const double2 ey =
        *reinterpret_cast<const double2 *>(Escale + node + npoint);
    double2 out = *reinterpret_cast<const double2 *>(dqdt + node);
    out.x -= ey.x * acc[2 * e8];
    out.y -= ey.y * acc[2 * e8 + 1];
    *reinterpret_cast<double2 *>(dqdt + node) = out;
  }
}

template <bool UseTc>
void launch_tendency_fused_p127_impl(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, double *flux_bnd,
    const double *Escale, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface, int Ne)
{
  const int nblock = 2 * NQ127 * Ne;
  const int nblock_y = 2 * NQ127 * Ne;
  constexpr int FACE_THREADS = 256;
  const int nblock_face = (nface + FACE_THREADS - 1) / FACE_THREADS;
  const size_t smem_xz =
      (P127_MT * NQ127 + 2 * P127_XZ_NBUF * NQ127 * BKD127) * sizeof(double);
  const size_t smem_y =
      (NQ127 * BKDY127 + P127_MT * NQ127) * sizeof(double);
  static bool opted_in = false;
  if (!opted_in) {
    cudaFuncSetAttribute(tendency_fused_p127_xz_kernel<UseTc>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_xz);
    cudaFuncSetAttribute(tendency_fused_p127_y_kernel<UseTc>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_y);
    opted_in = true;
  }
  {
    cudaLaunchConfig_t fcfg = {};
    cudaLaunchAttribute fattr[1];
    fcfg.gridDim = dim3(nblock_face);
    fcfg.blockDim = dim3(FACE_THREADS);
    fcfg.dynamicSmemBytes = 0;
    fcfg.stream = dg_cuda_stream;
    fattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    fattr[0].val.programmaticStreamSerializationAllowed = 1;
    fcfg.attrs = fattr;
    fcfg.numAttrs = 1;
    cudaLaunchKernelEx(&fcfg, pdl_elembnd_flux_kernel<true>, flux_bnd, q, u, v, w,
                       VMapM, VMapP, normal_fn, Fscale, nface);
  }
  {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    cfg.gridDim = dim3(nblock);
    cfg.blockDim = dim3(P127_XZ_THREADS);
    cfg.dynamicSmemBytes = (unsigned)smem_xz;
    cfg.stream = dg_cuda_stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, tendency_fused_p127_xz_kernel<UseTc>, dqdt, D1D,
                       Lift1D, q, u, w, flux_bnd, Escale, Ne);
  }
  {
    cudaLaunchConfig_t ycfg = {};
    cudaLaunchAttribute yattr[1];
    ycfg.gridDim = dim3(nblock_y);
    ycfg.blockDim = dim3(P127_Y_THREADS);
    ycfg.dynamicSmemBytes = (unsigned)smem_y;
    ycfg.stream = dg_cuda_stream;
    yattr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    yattr[0].val.programmaticStreamSerializationAllowed = 1;
    ycfg.attrs = yattr;
    ycfg.numAttrs = 1;
    cudaLaunchKernelEx(&ycfg, tendency_fused_p127_y_kernel<UseTc>, dqdt, D1D, q,
                       v, Escale, Ne);
  }
  check_cuda("tendency_fused_p127 kernels");
}

extern "C" void launch_tendency_fused_p127_dfma(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, double *flux_bnd,
    const double *Escale, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface, int Ne)
{
  launch_tendency_fused_p127_impl<false>(
      dqdt, D1D, Lift1D, q, u, v, w, flux_bnd, Escale, VMapM, VMapP, normal_fn,
      Fscale, nface, Ne);
}

extern "C" void launch_tendency_fused_p127_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, double *flux_bnd,
    const double *Escale, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface, int Ne)
{
  launch_tendency_fused_p127_impl<true>(
      dqdt, D1D, Lift1D, q, u, v, w, flux_bnd, Escale, VMapM, VMapP, normal_fn,
      Fscale, nface, Ne);
}
