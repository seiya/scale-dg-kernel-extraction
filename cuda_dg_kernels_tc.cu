#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

// FP64 Tensor Core GEMM helpers: mma.sync.aligned.m8n8k4.f64
// Fragment map (SM80+):
//   lane = thread % 32
//   A(8x4): A[lane/4][lane%4]
//   B(4x8): B[lane%4][lane/4]
//   C(8x8): C[lane/4][(lane%4)*2] and +1

__device__ __forceinline__ void mma_m8n8k4_f64(
    double &d0, double &d1, double a, double b, double c0, double c1)
{
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};"
      : "=d"(d0), "=d"(d1)
      : "d"(a), "d"(b), "d"(c0), "d"(c1));
}

__device__ __forceinline__ void mma_reset(double &c0, double &c1)
{
  c0 = 0.0;
  c1 = 0.0;
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
    double *const m = sM + ((i == 7) ? 0 : XFACE_PLANE) + ((node >> 3) & 7) +
                      ((node >> 6) << 3);
    m[0] = q;
    m[144] = u;
    m[288] = v;
    m[432] = w;
  }
}

__global__ __launch_bounds__(256, 8) void tendency_fused_p7_tc_kernel(
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
  __shared__ __align__(16) double sMface[4 * 144];
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
    sFluxX[sw_xy(node1)] = q1 * u1;
    sFluxY[sw_xy(node1)] = q1 * v1;
    sFluxZ[sw_z(node1)] = q1 * w1;
    stage_xface(sMface, node1, q1, u1, v1, w1);
  }
  {
    const double q2 = q[idx2], u2 = u[idx2], v2 = v[idx2], w2 = w[idx2];
    sFluxX[sw_xy(node2)] = q2 * u2;
    sFluxY[sw_xy(node2)] = q2 * v2;
    sFluxZ[sw_z(node2)] = q2 * w2;
    stage_xface(sMface, node2, q2, u2, v2, w2);
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
    mma_m8n8k4_f64(c0, c1, sDfrag[frag], sFluxZ[fz], c0, c1);
    mma_m8n8k4_f64(c0, c1, sDfrag[frag + 32], sFluxZ[fz ^ 256], c0, c1);
  }
  // sw_z() and sw_dz() permute across warp boundaries, so the z panel needs a
  // block-wide barrier before it overwrites the flux it was read from.
  __syncthreads();
  const int dz_c = sw_dz(((warp << 3) + j0_c) + (row << 6));
  sDz[dz_c] = c0;
  sDz[dz_c ^ 1] = c1;
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
    mma_m8n8k4_f64(c0, c1, sFluxX[fx], sDfrag[frag], c0, c1);
    mma_m8n8k4_f64(c0, c1, sFluxX[fx ^ 4], sDfrag[frag + 32], c0, c1);
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
    mma_m8n8k4_f64(c0, c1, sDfrag[frag], sFluxY[fy], c0, c1);
    mma_m8n8k4_f64(c0, c1, sDfrag[frag + 32], sFluxY[fy ^ 32], c0, c1);
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

#define NQ255 256
#define BM255 64
#define BN255 64
#define BK255 16
#define TM255 4
#define TN255 4
#define TH255 128
// Three blocks per SM is the most that fits without spilling: ptxas lands on
// 168 registers and 3 * 128 * 168 = 64512 of the 65536 register file.  Asking
// for four (128 registers) spills and costs 27%, asking for two wastes a third
// of the machine.
#define MINB255 3

// Shared tiles are stored as l + 16*outer, l in bits 0-3 and outer in bits
// 4-9.  Three different access patterns hit these arrays: the mma read (l from
// lane%4, outer from lane/4), an outer-fast store (D1D and the y/z fluxes,
// whose global source runs fastest in the outer index) and an l-fast store
// (the x flux, whose source runs fastest in l).  Folding bits 4-5 into bits
// 2-3 fixes the read, and folding bits 6-7 into bits 0-1 fixes the outer-fast
// store; each fold is over bits the other pattern holds constant, so one
// function makes all three conflict free at once.
__device__ __forceinline__ int sw255(int idx)
{
  return idx ^ (((idx >> 4) & 3) << 2) ^ (((idx >> 6) & 3) << 0);
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
template <int DIR, int BM, int BN, int TM, int TN>
__device__ __forceinline__ void p255_epilogue(
    double *dqdt, const double *Lift1D, const double *flux_bnd,
    const double *Escale, const double *acc, int m0, int n0, int wm, int wn,
    int row, int colk, int eo, int efo, int npoint, int plane_off, int kplane)
{
  const int NQ = NQ255;
  const int NQ2 = NQ * NQ;

  // Per-row (m) quantities: the two face flux values for x, the two lift
  // coefficients for y and z.
  double ra[TM], rb2[TM];
#pragma unroll
  for (int a = 0; a < TM; ++a) {
    const int m = m0 + 8 * (TM * wm + a) + row;
    if (DIR == 0) {
      const int fp = m + NQ * kplane;
      ra[a] = flux_bnd[efo + NQ2 + fp];
      rb2[a] = flux_bnd[efo + 3 * NQ2 + fp];
    } else if (DIR == 1) {
      ra[a] = Lift1D[m];
      rb2[a] = Lift1D[m + 2 * NQ];
    } else {
      ra[a] = Lift1D[m + 4 * NQ];
      rb2[a] = Lift1D[m + 5 * NQ];
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
        const double2 es = *reinterpret_cast<const double2 *>(Escale + node);
        const double l0 = c0n.x * ra[a] + c1n.x * rb2[a];
        const double l1 = c0n.y * ra[a] + c1n.y * rb2[a];
        *reinterpret_cast<double2 *>(dqdt + node) =
            make_double2(-(es.x * c0 + l0), -(es.y * c1 + l1));
      } else {
        // Faces 1 and 3 vary in (i,k) and faces 5 and 6 in the linear (i,j)
        // point: here the pair shares the coefficient and the two flux values
        // are one aligned double2.
        const double2 es = *reinterpret_cast<const double2 *>(
            Escale + node + (DIR == 1 ? npoint : 2 * npoint));
        double2 out = *reinterpret_cast<const double2 *>(dqdt + node);
        out.x -= es.x * c0 + ra[a] * c0n.x + rb2[a] * c1n.x;
        out.y -= es.y * c1 + ra[a] * c0n.y + rb2[a] * c1n.y;
        *reinterpret_cast<double2 *>(dqdt + node) = out;
      }
    }
  }
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
template <int DIR>
__global__ __launch_bounds__(TH255, MINB255) void tendency_p255_tc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ q,
    const double *__restrict__ velocity, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ flux_bnd,
    const double *__restrict__ Escale, int Ne)
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

  __shared__ __align__(16) double sA[2][BM * BK];
  __shared__ __align__(16) double sB[2][BN * BK];

  const int NQ = NQ255;
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
  constexpr int MTILES = NQ255 / BM;
  constexpr int NTILES = (DIR == 2) ? (NQ255 * NQ255 / BN) : (NQ255 / BN);
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
  //   sw255(l + 16*outer) = 16*outer + (colk ^ c) + ((4*ks) ^ ((row & 3) << 2))
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
    abase[a] = 16 * (8 * t + row) + (colk ^ ((2 * t + (row >> 2)) & 3));
  }
#pragma unroll
  for (int bb = 0; bb < TN; ++bb) {
    const int t = TN * wn + bb;
    bbase[bb] = 16 * (8 * t + row) + (colk ^ ((2 * t + (row >> 2)) & 3));
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
        const int i0 = sw255(ll + 16 * o);                                    \
        const double v0 = raq[2 * p] * rav[2 * p];                            \
        const double v1 = raq[2 * p + 1] * rav[2 * p + 1];                    \
        *reinterpret_cast<double2 *>(&sA[BUF][i0 & ~1]) =                     \
            (i0 & 1) ? make_double2(v1, v0) : make_double2(v0, v1);           \
      } else {                                                                \
        const int o = 2 * (pr % (BM / 2));                                    \
        const int ll = pr / (BM / 2);                                         \
        const int i0 = sw255(ll + 16 * o);                                    \
        const int i1 = sw255(ll + 16 * (o + 1));                              \
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
      const int j0 = sw255(ll + 16 * o);                                      \
      const int j1 = sw255(ll + 16 * (o + 1));                                \
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
          mma_m8n8k4_f64(acc[e], acc[e + 1], av[a], bv[bb], acc[e],
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

  p255_epilogue<DIR, BM, BN, TM, TN>(dqdt, Lift1D, flux_bnd, Escale, acc, m0,
                                     n0, wm, wn, row, colk, eo, efo, npoint,
                                     plane_off, kplane);
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

extern "C" void launch_tendency_fused_p7_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  tendency_fused_p7_tc_kernel<<<Ne, 256, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda("tendency_fused_p7_tc_kernel");
}


extern "C" void launch_tendency_xyz_p255_tc(
    double *dqdt, const double *q, const double *u, const double *v,
    const double *w, const double *D1D, const double *Lift1D,
    const double *flux_bnd, const double *Escale, int Ne)
{
  const int nblock = (NQ255 * NQ255 * NQ255 / (BM255 * BN255)) * Ne;
  tendency_p255_tc_kernel<0><<<nblock, TH255, 0, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_p255_tc_kernel<1><<<nblock, TH255, 0, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_p255_tc_kernel<2><<<nblock, TH255, 0, dg_cuda_stream>>>(
      dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, Ne);
  check_cuda("p255 tensor-core tendency kernels");
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
#define NQ15 16
#define NP15 4096
#define NFPTOT15 1536

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

// Read q, u, v and w at an M-side face node.  The two i-boundary planes of the
// element are in shared memory; every other node still goes to global.  The
// index is the one VMapM gave, so the test is on the map's own answer.
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

__global__ __launch_bounds__(1024, 1) void tendency_fused_p15_tc_kernel(
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
  double *const sMu = sMq + 512;
  double *const sMv = sMu + 512;
  double *const sMw = sMv + 512;

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
    LOAD_M(iM0, qM0, uM0, vM0, wM0);
    const double qP0 = q[iP0];
    const double VelM0 = uM0 * n0.x + vM0 * n1.x + wM0 * n2.x;
    const double VelP0 = u[iP0] * n0.x + v[iP0] * n1.x + w[iP0] * n2.x;
    const double a0 = 0.5 * fabs(VelP0 + VelM0);
    sflux[sw_f15(fp0)] =
        0.5 * fs.x * (qP0 * VelP0 - qM0 * VelM0 - a0 * (qP0 - qM0));

    double qM1, uM1, vM1, wM1;
    LOAD_M(iM1, qM1, uM1, vM1, wM1);
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
      mma_m8n8k4_f64(c0, c1, a, sDfrag[(ks * 8 << 2) + fbase], c0, c1);
      mma_m8n8k4_f64(c2, c3, a, sDfrag[128 + (ks * 8 << 2) + fbase], c2, c3);
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
      mma_m8n8k4_f64(c0, c1, a, sbufY[byA ^ (64 * ks)], c0, c1);
      mma_m8n8k4_f64(c2, c3, a, sbufY[byB ^ (64 * ks)], c2, c3);
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
      mma_m8n8k4_f64(c0, c1, sDfrag[(ks * 8 << 2) + fbase], b, c0, c1);
      mma_m8n8k4_f64(c2, c3, sDfrag[128 + (ks * 8 << 2) + fbase], b, c2, c3);
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

extern "C" void launch_tendency_fused_p15_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  const int p15_smem =
      (3 * NP15 + NFPTOT15 + 256 + 96 + 2048) * (int)sizeof(double);
  static bool p15_optin = false;
  if (!p15_optin) {
    cudaFuncSetAttribute(tendency_fused_p15_tc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, p15_smem);
    p15_optin = true;
  }
  tendency_fused_p15_tc_kernel<<<Ne, 1024, p15_smem, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda("tendency_fused_p15_tc_kernel");
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

#define NQ31 32
#define NP31 32768
#define NFPTOT31 6144
#define JSLAB31 16

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

__device__ __forceinline__ double p31_face_flux_tc(
    int fidx, const double *q, const double *u, const double *v,
    const double *w, const int *VMapM, const int *VMapP,
    const double *normal_fn, const double *Fscale, int nface)
{
  const int iM = VMapM[fidx] - 1;
  const int iP = VMapP[fidx] - 1;
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

__global__ __launch_bounds__(512, 1) void tendency_fused_p31_xz_tc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ v,
    const double *__restrict__ w, const int *__restrict__ VMapM,
    const int *__restrict__ VMapP, const double *__restrict__ normal_fn,
    const double *__restrict__ Fscale, const double *__restrict__ Escale,
    int Ne)
{
  // sFU and sFW hold the current j plane under ONE address map, node index
  // i + 32*k.  For x the contraction index l is the i field and for z it is
  // the k field, which is why one store path and one global load map serve
  // both panels.
  __shared__ __align__(16) double sFU[1024], sFW[1024];
  // Faces 2 and 4 are (j,k) planes and faces 5 and 6 are (i,j) planes, so
  // under the mma output map a consumer thread needs 16 j values of the first
  // pair and 32 of the second.  Faces 1 and 3 are (i,k) planes and stay in
  // registers; see the bijection note below.
  __shared__ __align__(16) double sf2[1024], sf4[1024];
  __shared__ __align__(16) double sf5[512], sf6[512];
  __shared__ __align__(16) double sLift[192];

  const int elem = (int)blockIdx.x >> 1;
  if (elem >= Ne) {
    return;
  }
  const int j0 = ((int)blockIdx.x & 1) * JSLAB31;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ti = warp & 3;
  const int tk = warp >> 2;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int elem_offset = elem * NP31;
  const int face_offset = elem * NFPTOT31;
  const int npoint = NP31 * Ne;
  const int nface = NFPTOT31 * Ne;

  if (tid < 192) {
    sLift[tid] = Lift1D[tid];
  }

  // D1D is Fortran column major, so D(r,l) is at r + 32*l.  x wants the B
  // operand D[i][l] with i = 8*ti + row and z the A operand D[k][l] with
  // k = 8*tk + row; both read column 4*ks + colk, and neither depends on j.
  double Dx[8], Dz[8];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const int col = NQ31 * (4 * ks + colk);
    Dx[ks] = D1D[(ti * 8 + row) + col];
    Dz[ks] = D1D[(tk * 8 + row) + col];
  }

  // Output nodes of this lane, for every j: i = 8*ti + 2*colk and +1 at
  // k = 8*tk + row.  The pair is adjacent in i, so dqdt and Escale move as
  // aligned double2.
  const int i0 = ti * 8 + 2 * colk;
  const int kout = tk * 8 + row;

  // Faces 1 and 3 are (i,k) planes, and (thread, {0,1}) -> i0 + 32*kout and +1
  // is a bijection onto the 1024 face points, so each thread evaluates exactly
  // the two points it will consume.  No redistribution, and the gather is
  // eight 64-byte runs per warp, the same shape as the epilogue.
  const int p13 = i0 + NQ31 * kout;
  const double f1a = p31_face_flux_tc(face_offset + p13, q, u, v, w, VMapM,
                                      VMapP, normal_fn, Fscale, nface);
  const double f1b = p31_face_flux_tc(face_offset + p13 + 1, q, u, v, w, VMapM,
                                      VMapP, normal_fn, Fscale, nface);
  const double f3a = p31_face_flux_tc(face_offset + 2048 + p13, q, u, v, w,
                                      VMapM, VMapP, normal_fn, Fscale, nface);
  const double f3b = p31_face_flux_tc(face_offset + 2048 + p13 + 1, q, u, v, w,
                                      VMapM, VMapP, normal_fn, Fscale, nface);

  // Faces 2 and 4: 16 j by 32 k is exactly 512 points.  The producer takes k
  // from the high nibble so a half warp holds k fixed and walks 16 consecutive
  // j, which is one 128-byte global run and, since the fold source is then
  // constant, a conflict-free shared store.
  {
    const int jl = tid & 15;
    const int kk = tid >> 4;
    const int pl = (j0 + jl) + NQ31 * kk;
    const int sa = sw31(jl + NQ31 * kk);
    sf2[sa] = p31_face_flux_tc(face_offset + 1024 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
    sf4[sa] = p31_face_flux_tc(face_offset + 3072 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
  }
  // Faces 5 and 6: 32 i by 16 j.  A warp walks 32 consecutive i at fixed j.
  {
    const int ii = tid & 31;
    const int jl = tid >> 5;
    const int pl = ii + NQ31 * (j0 + jl);
    const int sa = ii + NQ31 * jl;
    sf5[sa] = p31_face_flux_tc(face_offset + 4096 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
    sf6[sa] = p31_face_flux_tc(face_offset + 5120 + pl, q, u, v, w, VMapM,
                               VMapP, normal_fn, Fscale, nface);
  }
  __syncthreads();

  // Lift1D(Nq,6) varies in j for faces 1 and 3, in i for faces 2 and 4 and in
  // k for faces 5 and 6, so six of the eight coefficients this lane needs are
  // constant over the j loop and are read once here.
  const double lf2a = sLift[32 + i0];
  const double lf2b = sLift[33 + i0];
  const double lf4a = sLift[96 + i0];
  const double lf4b = sLift[97 + i0];
  const double lf5 = sLift[128 + kout];
  const double lf6 = sLift[160 + kout];

  // Operand base addresses.  For sFU the fold source is kout and for sFW it is
  // colk, and neither depends on ks, so the k-step is one XOR by a constant on
  // the x side and one add on the z side.
  const int ax = sw31(colk + NQ31 * kout);
  const int bz = sw31((ti * 8 + row) + NQ31 * colk);

  // The plane load is linear and coalesced, which the mma fragment map is not
  // and does not have to be: a half warp covers 32 consecutive nodes of one k
  // line, one 256-byte run, and each thread moves a double2 of q, u and w.
  const int ldi = 2 * (tid & 15);
  const int ldk = tid >> 4;
  const int ldsh = sw31(ldi + NQ31 * ldk);
  int gidx = elem_offset + ldi + NQ31 * j0 + (NQ31 * NQ31) * ldk;

  // One block per SM leaves no second block to interleave with, so the plane
  // for j+1 is issued before the mma of j consumes the plane for j.
  double2 qp = *reinterpret_cast<const double2 *>(q + gidx);
  double2 up = *reinterpret_cast<const double2 *>(u + gidx);
  double2 wp = *reinterpret_cast<const double2 *>(w + gidx);

  for (int jl = 0; jl < JSLAB31; ++jl) {
    *reinterpret_cast<double2 *>(sFU + ldsh) =
        make_double2(qp.x * up.x, qp.y * up.y);
    *reinterpret_cast<double2 *>(sFW + ldsh) =
        make_double2(qp.x * wp.x, qp.y * wp.y);
    if (jl + 1 < JSLAB31) {
      gidx += NQ31;
      qp = *reinterpret_cast<const double2 *>(q + gidx);
      up = *reinterpret_cast<const double2 *>(u + gidx);
      wp = *reinterpret_cast<const double2 *>(w + gidx);
    }
    __syncthreads();

    // x^T = (D * FU)^T and z^T = (FW * D^T)^T on this j plane.  Same output
    // map, separate accumulators, because Escale differs by direction.
    double cx0, cx1, cz0, cz1;
    mma_reset(cx0, cx1);
    mma_reset(cz0, cz1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      mma_m8n8k4_f64(cx0, cx1, sFU[ax ^ (4 * ks)], Dx[ks], cx0, cx1);
      mma_m8n8k4_f64(cz0, cz1, Dz[ks], sFW[bz + 128 * ks], cz0, cz1);
    }

    const int j = j0 + jl;
    const int nidx = elem_offset + i0 + NQ31 * j + (NQ31 * NQ31) * kout;
    const double2 ex = *reinterpret_cast<const double2 *>(Escale + nidx);
    const double2 ez =
        *reinterpret_cast<const double2 *>(Escale + nidx + 2 * npoint);
    const double lf1 = sLift[j];
    const double lf3 = sLift[64 + j];
    const int a24 = sw31(jl + NQ31 * kout);
    const double fb2 = sf2[a24];
    const double fb4 = sf4[a24];
    const double2 fb5 =
        *reinterpret_cast<const double2 *>(sf5 + i0 + NQ31 * jl);
    const double2 fb6 =
        *reinterpret_cast<const double2 *>(sf6 + i0 + NQ31 * jl);

    // Same summation order as tendency_fused_p31_xz_kernel.
    *reinterpret_cast<double2 *>(dqdt + nidx) = make_double2(
        -(ex.x * cx0 + ez.x * cz0 + lf1 * f1a + lf2a * fb2 + lf3 * f3a +
          lf4a * fb4 + lf5 * fb5.x + lf6 * fb6.x),
        -(ex.y * cx1 + ez.y * cz1 + lf1 * f1b + lf2b * fb2 + lf3 * f3b +
          lf4b * fb4 + lf5 * fb5.y + lf6 * fb6.y));
    __syncthreads();
  }
}

//> p=31 y volume term, accumulated onto what the xz kernel wrote.
//
// Threads are (i,j) with k the inner loop, and the contraction
// C(i,j) = sum_l FV(i,k,l) * D(j,l) is structurally the z contraction of the
// first kernel: transposed it wants D as the A operand from registers and the
// plane as the B operand from shared, under the same address map and the same
// swizzle.  The block owns half an element in k, for the same wave reason.
__global__ __launch_bounds__(512, 1) void tendency_fused_p31_y_tc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  __shared__ __align__(16) double sFV[1024];
  // Two 8 KB stages for the dqdt tile the epilogue reads back.  Each thread
  // asks for the single 16-byte slot it will read, so the buffer needs no
  // barrier of its own; see section 19.4 of p63_gap_study.md.
  __shared__ __align__(16) double sDQ[2 * 1024];

  const int elem = (int)blockIdx.x >> 1;
  if (elem >= Ne) {
    return;
  }
  const int k0 = ((int)blockIdx.x & 1) * JSLAB31;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ti = warp & 3;
  const int tj = warp >> 2;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int elem_offset = elem * NP31;
  const int npoint = NP31 * Ne;

  double Dy[8];
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    Dy[ks] = D1D[(tj * 8 + row) + NQ31 * (4 * ks + colk)];
  }

  const int i0 = ti * 8 + 2 * colk;
  const int jout = tj * 8 + row;
  const int by = sw31((ti * 8 + row) + NQ31 * colk);

  const int ldi = 2 * (tid & 15);
  const int ldj = tid >> 4;
  const int ldsh = sw31(ldi + NQ31 * ldj);
  int gidx = elem_offset + ldi + NQ31 * ldj + (NQ31 * NQ31) * k0;

  double2 qp = *reinterpret_cast<const double2 *>(q + gidx);
  double2 vp = *reinterpret_cast<const double2 *>(v + gidx);

  const int dqslot = 2 * tid;
  const int nidx0 = elem_offset + i0 + NQ31 * jout + (NQ31 * NQ31) * k0;
  cp_async_16(sDQ + dqslot, dqdt + nidx0);
  asm volatile("cp.async.commit_group;\n" ::);

  for (int kl = 0; kl < JSLAB31; ++kl) {
    *reinterpret_cast<double2 *>(sFV + ldsh) =
        make_double2(qp.x * vp.x, qp.y * vp.y);
    if (kl + 1 < JSLAB31) {
      gidx += NQ31 * NQ31;
      qp = *reinterpret_cast<const double2 *>(q + gidx);
      vp = *reinterpret_cast<const double2 *>(v + gidx);
    }
    if (kl + 1 < JSLAB31) {
      cp_async_16(sDQ + 1024 * ((kl + 1) & 1) + dqslot,
                  dqdt + nidx0 + (NQ31 * NQ31) * (kl + 1));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    __syncthreads();

    double c0, c1;
    mma_reset(c0, c1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      mma_m8n8k4_f64(c0, c1, Dy[ks], sFV[by + 128 * ks], c0, c1);
    }

    const int nidx =
        elem_offset + i0 + NQ31 * jout + (NQ31 * NQ31) * (k0 + kl);
    const double2 ey =
        *reinterpret_cast<const double2 *>(Escale + nidx + npoint);
    asm volatile("cp.async.wait_group 1;\n" ::);
    double2 out = *reinterpret_cast<const double2 *>(
        sDQ + 1024 * (kl & 1) + dqslot);
    out.x -= ey.x * c0;
    out.y -= ey.y * c1;
    *reinterpret_cast<double2 *>(dqdt + nidx) = out;
    __syncthreads();
  }
}

extern "C" void launch_tendency_fused_p31_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  tendency_fused_p31_xz_tc_kernel<<<2 * Ne, 512, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  tendency_fused_p31_y_tc_kernel<<<2 * Ne, 512, 0, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda("tendency_fused_p31_tc kernels");
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

#define NQ63 64
#define NP63 262144
#define NQ2_63 4096
#define NFPTOT63 24576
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
#define BK63 64

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
#define P63_WN 4
#define P63_TN (8 / P63_WN)
#define P63_THREADS (32 * 4 * P63_WN)
#define P63_STAGE_ITERS (NQ63 * BK63 / P63_THREADS)
#define P63_BPSM 1

// The y kernel carries one accumulator set where the xz kernel carries two,
// so its shape is set separately.  Two blocks per SM is what that buys: 512
// threads asking for 64 registers is exactly the register file twice over,
// and the y kernel reaches 64 with no spill where the xz kernel spills 128 to
// 176 bytes and loses 8%.  Section 19 of p63_gap_study.md.
#define P63Y_WN 4
#define P63Y_TN (8 / P63Y_WN)
#define P63Y_THREADS (32 * 4 * P63Y_WN)
#define P63Y_STAGE_ITERS (NQ63 * BK63 / P63Y_THREADS)
#define P63Y_BPSM 2

__global__ __launch_bounds__(P63_THREADS, P63_BPSM) void tendency_fused_p63_xz_tc_kernel(
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
  extern __shared__ __align__(16) double smem63[];
  double *const sFU = smem63;
  double *const sD = smem63 + NQ63 * BK63;
  double *const sFW = smem63 + 2 * NQ63 * BK63;

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

  for (int kk = 0; kk < NQ63; kk += BK63) {
    // The barrier that protects the panels from being overwritten belongs at
    // the head of the body, not at its end: written at the end it also runs
    // after the last chunk, where nothing follows it.  At BK63 = NQ63 the
    // loop runs once and that saves one barrier out of two.
    if (kk) {
      __syncthreads();
    }
    // sFU[k][l] = q*u at (l, jp, k).  l is fast in global, so sixteen lanes
    // walk l and cover one 128-byte run.
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      const int ll = tid & (BK63 - 1);
      const int o = (tid / BK63) + (P63_THREADS / BK63) * p;
      const int g = eo + (kk + ll) + plane_off + NQ2_63 * o;
      sFU[swu63(o + BK63 * ll)] = q[g] * u[g];
    }
    // sD[r][l] = D1D(r, l), r fast in the Fortran column-major operator.
    // sFW[i][l] = q*w at (i, jp, l), i fast in global.
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      const int o = tid & 63;
      const int ll = (tid >> 6) + (P63_THREADS / 64) * p;
      sD[swt63(o + BK63 * ll)] = D1D[o + NQ63 * (kk + ll)];
      const int g = eo + o + plane_off + NQ2_63 * (kk + ll);
      sFW[swt63(o + BK63 * ll)] = q[g] * w[g];
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK63 / 4; ++ks) {
      const int l = 4 * ks + colk;
      double av[2], bv[P63_TN], avz[2], bvz[P63_TN];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int m = 8 * (2 * wm + a) + row;
        av[a] = sFU[swu63(m + BK63 * l)];
        avz[a] = sD[swt63(m + BK63 * l)];
      }
#pragma unroll
      for (int bb = 0; bb < P63_TN; ++bb) {
        const int n = 8 * (P63_TN * wn + bb) + row;
        bv[bb] = sD[swt63(n + BK63 * l)];
        bvz[bb] = sFW[swt63(n + BK63 * l)];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < P63_TN; ++bb) {
          const int e = 2 * (P63_TN * a + bb);
          mma_m8n8k4_f64(ax[e], ax[e + 1], av[a], bv[bb], ax[e], ax[e + 1]);
          mma_m8n8k4_f64(az[e], az[e + 1], avz[a], bvz[bb], az[e], az[e + 1]);
        }
      }
    }
  }

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
__global__ __launch_bounds__(P63Y_THREADS, P63Y_BPSM) void tendency_fused_p63_y_tc_kernel(
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
          mma_m8n8k4_f64(acc[e], acc[e + 1], av[a], bv[bb], acc[e], acc[e + 1]);
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

extern "C" void launch_tendency_fused_p63_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = NQ63 * Ne;
  const size_t smem_xz = 3 * NQ63 * BK63 * sizeof(double);
  const size_t smem_y = (2 * NQ63 * BK63 + NQ2_63) * sizeof(double);
  static bool opted_in = false;
  if (!opted_in) {
    cudaFuncSetAttribute(tendency_fused_p63_xz_tc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_xz);
    cudaFuncSetAttribute(tendency_fused_p63_y_tc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_y);
    opted_in = true;
  }
  tendency_fused_p63_xz_tc_kernel<<<nblock, P63_THREADS, smem_xz,
                                    dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p63_y_tc_kernel<<<nblock, P63Y_THREADS, smem_y,
                                   dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda("tendency_fused_p63_tc kernels");
}

//============================================================================
// p=127 (Nq=128) fused Tensor Core tendency
//============================================================================
//
// Same three-kernel shape as p=63: elembnd_flux_kernel evaluates the six face
// fluxes once, then an xz kernel writes dqdt and a y kernel accumulates onto
// it.  A 128x128 output plane cannot be one block, so both volume kernels
// tile it 128x64 -- the whole contraction index m, half the output index n --
// and run two blocks per plane, 1024 threads in an 8 by 4 warp grid with 2x2
// mma tiles per warp.
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

#define NQ127 128
#define NP127 2097152
#define NQ2_127 16384
#define NFPTOT127 98304

// Depth of the chunked panels.  The first version chunked every panel
// together, the p=63 arrangement, and deeper was monotonically better (999.0
// / 934.7 / 866.2 us/stage at 16 / 32 / 64 on the 64x64 tile) for the reason
// p=63 section 16 gives: with no prefetch the loads of chunk k+1 are not
// issued until the mma of chunk k has consumed chunk k.  With the resident
// flux panel that pressure is off the DRAM path, and what is left is an L2
// round trip per chunk:
//
//   BKD127   xz shared   us/stage
//     32       128 KB      794.9
//     64       192 KB      757.2   <- kept
//
// 128 would remove the chunk loop but needs 320 KB against the 227 KB a
// Blackwell block can hold.  The y kernel holds one chunked panel instead of
// two, so 128 does fit there, and it loses anyway (765.6 against 756.1).
//
// The register-blocking ratio is where this design stops.  Going from 2x2 to
// 4x2 mma tiles per warp would cut the operand loads per tile from 1.0 to
// 0.75, and it is reachable only for the y kernel -- the xz kernel carries
// two accumulator sets, so 4x2 is 32 doubles a lane and 1024 threads have 64
// registers.  It was written for the y kernel, one block per whole 128x128
// plane, 192 KB, no spills, and it changes nothing: 759.3 against 757.2.
// Together with the ablation that removes the mma itself for no gain, that
// says the mma loop is bound by neither the mma nor the count of shared
// loads, and nothing further is diagnosable without ncu.
#ifndef BKD127
#define BKD127 64
#endif

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

// Both volume kernels run 1024 threads, 32 warps in an 8 by 4 grid with 2x2
// mma tiles per warp, which covers the 128x64 tile of either kernel.
//
// The xz warp shape was swept twice.  With the first arrangement -- every
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
#define P127_MT 64
#define P127_Y_THREADS 1024
#define P127_Y_FSTAGE_ITERS (P127_MT * NQ127 / P127_Y_THREADS)

__global__ __launch_bounds__(1024, 1) void tendency_fused_p127_xz_tc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ Lift1D, const double *__restrict__ q,
    const double *__restrict__ u, const double *__restrict__ w,
    const double *__restrict__ flux_bnd, const double *__restrict__ Escale,
    int Ne)
{
  extern __shared__ __align__(16) double smem127[];
  double *const sFW = smem127;                       //  64 x NQ, full depth
  double *const sFU = smem127 + P127_MT * NQ127;     // 128 x BKD
  double *const sD = smem127 + P127_MT * NQ127 + NQ127 * BKD127;  // 128 x BKD

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
  const int wm = warp & 7;
  const int wn = warp >> 3;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP127;
  const int efo = elem * NFPTOT127;
  const int npoint = NP127 * Ne;
  const int plane_off = NQ127 * jp;

  double ax[2 * 2 * 2], az[2 * 2 * 2];
#pragma unroll
  for (int e = 0; e < 2 * 2 * 2; ++e) {
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
  int amx[2], bnx[2];
#pragma unroll
  for (int a = 0; a < 2; ++a) {
    amx[a] = (8 * (2 * wm + a) + row) ^ cx;
  }
#pragma unroll
  for (int bb = 0; bb < 2; ++bb) {
    bnx[bb] = (8 * (2 * wn + bb) + row) ^ cx;
  }

  // sFW[i][l] = q*w at (ibase+i, jp, l), the panel this tile reads exactly
  // once per plane, so it is the one staged at full depth.
#pragma unroll
  for (int p = 0; p < P127_MT * NQ127 / 1024; ++p) {
    const int o = tid & (P127_MT - 1);
    const int ll = (tid / P127_MT) + (1024 / P127_MT) * p;
    const int g = eo + (ibase + o) + plane_off + NQ2_127 * ll;
    sFW[swt127(o + P127_MT * ll)] = q[g] * w[g];
  }

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
    // sFU[k][l] = q*u at (kk+l, jp, k), all 128 k.  l is fast in global.
#pragma unroll
    for (int p = 0; p < NQ127 * BKD127 / 1024; ++p) {
      const int ll = tid & (BKD127 - 1);
      const int o = (tid / BKD127) + (1024 / BKD127) * p;
      const int g = eo + (kk + ll) + plane_off + NQ2_127 * o;
      sFU[swt128(o + NQ127 * ll)] = q[g] * u[g];
    }
    // sD[r][l] = D1D(r, kk+l), all 128 rows: m runs over the whole range and
    // n sits inside it, so one panel serves both operands as it did at p=63.
#pragma unroll
    for (int p = 0; p < NQ127 * BKD127 / 1024; ++p) {
      const int o = tid & (NQ127 - 1);
      const int ll = (tid / NQ127) + (1024 / NQ127) * p;
      sD[swt128(o + NQ127 * ll)] = D1D[o + NQ127 * (kk + ll)];
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BKD127 / 4; ++ks) {
      const int lc = 4 * ks + colk;
      const int lg = kk + lc;
      double av[2], bv[2], avz[2], bvz[2];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        av[a] = sFU[amx[a] + NQ127 * lc];
        avz[a] = sD[amx[a] + NQ127 * lc];
      }
#pragma unroll
      for (int bb = 0; bb < 2; ++bb) {
        bv[bb] = sD[ibase + bnx[bb] + NQ127 * lc];
        bvz[bb] = sFW[bnx[bb] + P127_MT * lg];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < 2; ++bb) {
          const int e = 2 * (2 * a + bb);
          mma_m8n8k4_f64(ax[e], ax[e + 1], av[a], bv[bb], ax[e], ax[e + 1]);
          mma_m8n8k4_f64(az[e], az[e + 1], avz[a], bvz[bb], az[e], az[e + 1]);
        }
      }
    }
  }

  const double lf1 = Lift1D[jp];
  const double lf3 = Lift1D[jp + 2 * NQ127];

#pragma unroll
  for (int e8 = 0; e8 < 4; ++e8) {
    const int a = e8 / 2;
    const int bb = e8 % 2;
    const int m = 8 * (2 * wm + a) + row;                 // k
    const int n = ibase + 8 * (2 * wn + bb) + 2 * colk;   // i, and i+1
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
__global__ __launch_bounds__(P127_Y_THREADS, 1) void tendency_fused_p127_y_tc_kernel(
    double *__restrict__ dqdt, const double *__restrict__ D1D,
    const double *__restrict__ q, const double *__restrict__ v,
    const double *__restrict__ Escale, int Ne)
{
  extern __shared__ __align__(16) double smem127[];
  double *const sDm = smem127;
  double *const sFV = smem127 + NQ127 * BKD127;

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
  const int wm = warp & 7;
  const int wn = warp >> 3;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP127;
  const int npoint = NP127 * Ne;
  const int plane_off = NQ2_127 * kp;

  double acc[2 * 2 * 2];
#pragma unroll
  for (int e = 0; e < 2 * 2 * 2; ++e) {
    acc[e] = 0.0;
  }

  // Same collapse as the xz kernel.  sFV is the l-fast panel, so the field
  // its swizzle folds is the row index's low three bits, which is row -- also
  // a per-lane constant -- and the XOR moves onto l instead.
  const int cx = colk << 2;
  const int rx = row << 2;
  int amx[2], boff[2];
#pragma unroll
  for (int a = 0; a < 2; ++a) {
    amx[a] = (8 * (2 * wm + a) + row) ^ cx;
  }
#pragma unroll
  for (int bb = 0; bb < 2; ++bb) {
    boff[bb] = NQ127 * (8 * (2 * wn + bb) + row);
  }

  // sFV[i][l] = q*v at (ibase+i, l, kp), staged once at the full depth.
#pragma unroll
  for (int p = 0; p < P127_Y_FSTAGE_ITERS; ++p) {
    const int o = tid & (P127_MT - 1);
    const int ll = (tid / P127_MT) + (P127_Y_THREADS / P127_MT) * p;
    const int g = eo + (ibase + o) + NQ127 * ll + plane_off;
    sFV[sw127(ll + NQ127 * o)] = q[g] * v[g];
  }

  for (int kk = 0; kk < NQ127; kk += BKD127) {
    if (kk) {
      __syncthreads();
    }
#pragma unroll
    for (int p = 0; p < BKD127 * NQ127 / P127_Y_THREADS; ++p) {
      const int o = tid & (NQ127 - 1);
      const int ll = (tid / NQ127) + (P127_Y_THREADS / NQ127) * p;
      sDm[swt128(o + NQ127 * ll)] = D1D[o + NQ127 * (kk + ll)];
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BKD127 / 4; ++ks) {
      const int lc = 4 * ks + colk;
      const int lg = kk + lc;
      double av[2], bv[2];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        av[a] = sDm[amx[a] + NQ127 * lc];
      }
#pragma unroll
      for (int bb = 0; bb < 2; ++bb) {
        bv[bb] = sFV[(lg ^ rx) + boff[bb]];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < 2; ++bb) {
          const int e = 2 * (2 * a + bb);
          mma_m8n8k4_f64(acc[e], acc[e + 1], av[a], bv[bb], acc[e], acc[e + 1]);
        }
      }
    }
  }

#pragma unroll
  for (int e8 = 0; e8 < 4; ++e8) {
    const int a = e8 / 2;
    const int bb = e8 % 2;
    const int m = 8 * (2 * wm + a) + row;                     // j
    const int n = ibase + 8 * (2 * wn + bb) + 2 * colk;       // i, and i+1
    const int node = eo + n + NQ127 * m + plane_off;
    const double2 ey =
        *reinterpret_cast<const double2 *>(Escale + node + npoint);
    double2 out = *reinterpret_cast<const double2 *>(dqdt + node);
    out.x -= ey.x * acc[2 * e8];
    out.y -= ey.y * acc[2 * e8 + 1];
    *reinterpret_cast<double2 *>(dqdt + node) = out;
  }
}

extern "C" void launch_tendency_fused_p127_tc(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  const int nblock = 2 * NQ127 * Ne;
  const int nblock_y = 2 * NQ127 * Ne;
  const size_t smem_xz =
      (P127_MT * NQ127 + 2 * NQ127 * BKD127) * sizeof(double);
  const size_t smem_y =
      (NQ127 * BKD127 + P127_MT * NQ127) * sizeof(double);
  static bool opted_in = false;
  if (!opted_in) {
    cudaFuncSetAttribute(tendency_fused_p127_xz_tc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_xz);
    cudaFuncSetAttribute(tendency_fused_p127_y_tc_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem_y);
    opted_in = true;
  }
  tendency_fused_p127_xz_tc_kernel<<<nblock, 1024, smem_xz,
                                     dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p127_y_tc_kernel<<<nblock_y, P127_Y_THREADS, smem_y,
                                    dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda("tendency_fused_p127_tc kernels");
}
