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
// The previous kernels here ran one warp per block on a single 8x8 output tile
// and reloaded BOTH mma operands from global for every one of the 64 k-steps,
// so each mma moved 96 doubles for 512 FLOP: 0.67 FLOP/B, about 39 GB/stage
// through L1.  ncu (Slurm job 59500) found all three pinned at 98-99% L1/TEX
// with SM throughput at 18%, which is that arithmetic and nothing else.
//
// This is the same limiter the Nq=32 kernels had, but the cure is different:
// there the operands were already in shared and the fix was to hoist the D1D
// fragment into registers; here they come straight from global and the fix is
// to stage a tile of each in shared and amortize it over a 64x64 output block.
// A 64-wide block reads each operand once for 64 columns instead of once per
// column, which is an 8x cut in operand traffic.
//
// Every direction is written as the transposed product
//
//     C^T[m][n] = sum_l A[m][l] * B[n][l]
//
// which makes both operands the same shape -- [outer][l] with outer taken from
// lane/4 and l from lane%4 -- so one shared layout and one loader serve both.
// The transpose is free with m8n8k4 (it is the operand order) and it is what
// makes the epilogue coalesce: a lane ends up owning two nodes adjacent in the
// fastest index, where the old kernels owned two nodes 256 or 65536 apart.
//
//   x: C^T[j][i], A[j][l] = q*u at (l,j,k)   B[i][l] = D1D(i,l)
//   y: C^T[j][i], A[j][l] = D1D(j,l)         B[i][l] = q*v at (i,l,k)
//   z: C^T[k][p], A[k][l] = D1D(k,l)         B[p][l] = q*w at (p + Nq^2*l)
//
// where p is the linear (i,j) index, so z contracts the whole element at once
// and needs no plane loop.  Faces stay split two per direction exactly as
// before: x lifts faces 2 and 4, y faces 1 and 3, z faces 5 and 6.

#define NQ255 256
#define BM255 64
#define BN255 64
#define BK255 16

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

// DIR: 0 = x, 1 = y, 2 = z.
template <int DIR>
__global__ __launch_bounds__(256, 4) void tendency_p255_tc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  __shared__ __align__(16) double sA[BM255 * BK255];
  __shared__ __align__(16) double sB[BN255 * BK255];

  const int NQ = NQ255;
  const int NP = NQ * NQ * NQ;
  const int NQ2 = NQ * NQ;
  // x and y: (Nq/64) m-tiles * (Nq/64) n-tiles * Nq planes.
  // z:       (Nq/64) m-tiles * (Nq^2/64) n-tiles, no plane loop.  Both are
  // 4096, so the grid is the same shape for all three.
  const int blocks_per_elem = 4096;

  const int elem = (int)blockIdx.x / blocks_per_elem;
  if (elem >= Ne) {
    return;
  }
  const int b = (int)blockIdx.x - elem * blocks_per_elem;
  int tm, tn, kplane;
  if (DIR == 2) {
    tm = b & 3;
    tn = b >> 2;
    kplane = 0;
  } else {
    tm = b & 3;
    tn = (b >> 2) & 3;
    kplane = b >> 4;
  }
  const int m0 = tm * BM255;
  const int n0 = tn * BN255;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row = lane >> 2;
  const int colk = lane & 3;

  const int eo = elem * NP;
  const int efo = elem * 6 * NQ2;
  const int npoint = NP * Ne;
  const int plane_off = kplane * NQ2;

  // Eight 8x8 output tiles per warp, arranged 2 rows by 4 columns rather than
  // 1 by 8.  Both shapes hold eight accumulator pairs, but 2x4 needs 2 + 4 = 6
  // operand loads per k-step where 1x8 needs 1 + 8 = 9, so the same registers
  // buy a third fewer shared loads.  The warp grid is 4 (rows) by 2 (columns).
  const int wm = warp & 3;
  const int wn = warp >> 2;
  double acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    acc[i] = 0.0;
  }

  for (int kk = 0; kk < NQ; kk += BK255) {
    //- stage A --------------------------------------------------------
    if (DIR == 0) {
      // q*u at (l, j, k): l-fast in global, so lanes walk l.
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const int ll = tid & 15;
        const int o = (tid >> 4) + 16 * p;
        const int g = eo + (kk + ll) + NQ * (m0 + o) + plane_off;
        sA[sw255(ll + 16 * o)] = q[g] * velocity[g];
      }
    } else {
      // D1D(m, l): m-fast in global, so lanes walk m.
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const int o = tid & 63;
        const int ll = (tid >> 6) + 4 * p;
        sA[sw255(ll + 16 * o)] = D1D[(m0 + o) + NQ * (kk + ll)];
      }
    }
    //- stage B --------------------------------------------------------
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int o = tid & 63;
      const int ll = (tid >> 6) + 4 * p;
      double val;
      if (DIR == 0) {
        val = D1D[(n0 + o) + NQ * (kk + ll)];
      } else if (DIR == 1) {
        const int g = eo + (n0 + o) + NQ * (kk + ll) + plane_off;
        val = q[g] * velocity[g];
      } else {
        const int g = eo + (n0 + o) + NQ2 * (kk + ll);
        val = q[g] * velocity[g];
      }
      sB[sw255(ll + 16 * o)] = val;
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK255 / 4; ++ks) {
      const int l = 4 * ks + colk;
      double av[2], bv[4];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        av[a] = sA[sw255(l + 16 * (8 * (2 * wm + a) + row))];
      }
#pragma unroll
      for (int bb = 0; bb < 4; ++bb) {
        bv[bb] = sB[sw255(l + 16 * (8 * (4 * wn + bb) + row))];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < 4; ++bb) {
          const int e = 2 * (4 * a + bb);
          mma_m8n8k4_f64(acc[e], acc[e + 1], av[a], bv[bb], acc[e],
                         acc[e + 1]);
        }
      }
    }
    __syncthreads();
  }

  //- epilogue ---------------------------------------------------------
#pragma unroll
  for (int e8 = 0; e8 < 8; ++e8) {
    const int a = e8 >> 2;
    const int bb = e8 & 3;
    const int m = m0 + 8 * (2 * wm + a) + row;
    const int n = n0 + 8 * (4 * wn + bb) + 2 * colk;
    const double c0 = acc[2 * e8];
    const double c1 = acc[2 * e8 + 1];
    if (DIR == 0) {
      // Faces 2 and 4 vary in (j,k), so the node pair shares the flux value
      // and differs only through the Lift1D coefficient, which varies in i.
      const int node = eo + n + NQ * m + plane_off;
      const int fp = m + NQ * kplane;
      const double fb2 = flux_bnd[efo + NQ2 + fp];
      const double fb4 = flux_bnd[efo + 3 * NQ2 + fp];
      const double2 es = *reinterpret_cast<const double2 *>(Escale + node);
      const double l0 = Lift1D[n + NQ] * fb2 + Lift1D[n + 3 * NQ] * fb4;
      const double l1 =
          Lift1D[n + 1 + NQ] * fb2 + Lift1D[n + 1 + 3 * NQ] * fb4;
      *reinterpret_cast<double2 *>(dqdt + node) =
          make_double2(-(es.x * c0 + l0), -(es.y * c1 + l1));
    } else if (DIR == 1) {
      // Faces 1 and 3 vary in (i,k): here the pair shares the coefficient and
      // the two flux values are one aligned double2.
      const int node = eo + n + NQ * m + plane_off;
      const int fp = n + NQ * kplane;
      const double2 fb1 = *reinterpret_cast<const double2 *>(flux_bnd + efo + fp);
      const double2 fb3 =
          *reinterpret_cast<const double2 *>(flux_bnd + efo + 2 * NQ2 + fp);
      const double2 es =
          *reinterpret_cast<const double2 *>(Escale + node + npoint);
      const double lc1 = Lift1D[m];
      const double lc3 = Lift1D[m + 2 * NQ];
      double2 out = *reinterpret_cast<const double2 *>(dqdt + node);
      out.x -= es.x * c0 + lc1 * fb1.x + lc3 * fb3.x;
      out.y -= es.y * c1 + lc1 * fb1.y + lc3 * fb3.y;
      *reinterpret_cast<double2 *>(dqdt + node) = out;
    } else {
      // Faces 5 and 6 are indexed by the linear (i,j) point, which is exactly
      // the n index here.
      const int node = eo + n + NQ2 * m;
      const double2 fb5 =
          *reinterpret_cast<const double2 *>(flux_bnd + efo + 4 * NQ2 + n);
      const double2 fb6 =
          *reinterpret_cast<const double2 *>(flux_bnd + efo + 5 * NQ2 + n);
      const double2 es =
          *reinterpret_cast<const double2 *>(Escale + node + 2 * npoint);
      const double lc5 = Lift1D[m + 4 * NQ];
      const double lc6 = Lift1D[m + 5 * NQ];
      double2 out = *reinterpret_cast<const double2 *>(dqdt + node);
      out.x -= es.x * c0 + lc5 * fb5.x + lc6 * fb6.x;
      out.y -= es.y * c1 + lc5 * fb5.y + lc6 * fb6.y;
      *reinterpret_cast<double2 *>(dqdt + node) = out;
    }
  }
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
  const int nblock = 4096 * Ne;
  tendency_p255_tc_kernel<0><<<nblock, 256, 0, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_p255_tc_kernel<1><<<nblock, 256, 0, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_p255_tc_kernel<2><<<nblock, 256, 0, dg_cuda_stream>>>(
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

__global__ __launch_bounds__(1024, 1) void tendency_fused_p15_tc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sbuf[NP15];
  // A buffer of its own for the six face fluxes, 12288 B on top of the 35584 B
  // the rest of the kernel uses and still inside the 48 KB static limit.
  // Sharing sbuf with the volume panels forced the fluxes to be evaluated
  // after the z phase, which is where their scattered gathers used to be the
  // first thing anyone waited on.  See section 15 of p15_gap_study.md.
  __shared__ __align__(16) double sflux[NFPTOT15];
  __shared__ __align__(16) double sDfrag[256];
  __shared__ __align__(16) double sLift[96];

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

  //- x -----------------------------------------------------------------
  {
    const double2 ua = *reinterpret_cast<const double2 *>(u + iag);
    const double2 ub = *reinterpret_cast<const double2 *>(u + ibg);
    *reinterpret_cast<double2 *>(sbuf + sw_xy15(na)) =
        make_double2(qa.x * ua.x, qa.y * ua.y);
    *reinterpret_cast<double2 *>(sbuf + sw_xy15(nb)) =
        make_double2(qb.x * ub.x, qb.y * ub.y);
  }

  //- numerical flux on the six faces ------------------------------------
  //
  // This runs here, before any of the three contractions is consumed, rather
  // than after the z phase where it used to.  The gathers are unchanged --
  // ncu counts the same 4.63 M requests and 62.36 M sectors either way -- but
  // from here they have the x, y and z phases to complete in.  Section 15 of
  // p15_gap_study.md measured long scoreboard falling 47.1% to 24.5% and the
  // kernel 5.3% shorter, with bit-identical output.  Evaluating them one step
  // earlier still, before the x panel is staged, is 1% worse: it delays the
  // first mma without buying more overlap.
  {
    int fp = tid;
    int fidx = face_offset + fp;
    int iM = VMapM[fidx] - 1;
    int iP = VMapP[fidx] - 1;
    double qM = q[iM], qP = q[iP];
    double VelM = u[iM] * normal_fn[fidx] + v[iM] * normal_fn[fidx + nface] +
                  w[iM] * normal_fn[fidx + 2 * nface];
    double VelP = u[iP] * normal_fn[fidx] + v[iP] * normal_fn[fidx + nface] +
                  w[iP] * normal_fn[fidx + 2 * nface];
    double alpha = 0.5 * fabs(VelP + VelM);
    sflux[sw_f15(fp)] =
        0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
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
      sflux[sw_f15(fp)] =
          0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
    }
  }
  __syncthreads();
  {
    // A = flux panel at (i = colk + 4*ks, j = jout); the k-step moves bits
    // 2-3, which sw_xy15 does not read.
    const int ax = sw_xy15(colk + NQ15 * jout + 256 * k);
    const int fbase = row * 4 + colk;
    mma_reset(c0, c1);
    mma_reset(c2, c3);
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const double a = sbuf[ax ^ (4 * ks)];
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
  __syncthreads();

  //- y -----------------------------------------------------------------
  {
    const double2 va = *reinterpret_cast<const double2 *>(v + iag);
    const double2 vb = *reinterpret_cast<const double2 *>(v + ibg);
    *reinterpret_cast<double2 *>(sbuf + sw_xy15(na)) =
        make_double2(qa.x * va.x, qa.y * va.y);
    *reinterpret_cast<double2 *>(sbuf + sw_xy15(nb)) =
        make_double2(qb.x * vb.x, qb.y * vb.y);
  }
  __syncthreads();
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
      mma_m8n8k4_f64(c0, c1, a, sbuf[byA ^ (64 * ks)], c0, c1);
      mma_m8n8k4_f64(c2, c3, a, sbuf[byB ^ (64 * ks)], c2, c3);
    }
    const double2 ea = *reinterpret_cast<const double2 *>(Escale + gA + npoint);
    const double2 eb = *reinterpret_cast<const double2 *>(Escale + gB + npoint);
    acc0 += ea.x * c0;
    acc1 += ea.y * c1;
    acc2 += eb.x * c2;
    acc3 += eb.y * c3;
  }
  __syncthreads();

  //- z -----------------------------------------------------------------
  {
    const double2 wa = *reinterpret_cast<const double2 *>(w + iag);
    const double2 wb = *reinterpret_cast<const double2 *>(w + ibg);
    *reinterpret_cast<double2 *>(sbuf + sw_z15(na)) =
        make_double2(qa.x * wa.x, qa.y * wa.y);
    *reinterpret_cast<double2 *>(sbuf + sw_z15(nb)) =
        make_double2(qb.x * wb.x, qb.y * wb.y);
  }
  __syncthreads();
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
      const double b = sbuf[bz ^ (1024 * ks)];
      mma_m8n8k4_f64(c0, c1, sDfrag[(ks * 8 << 2) + fbase], b, c0, c1);
      mma_m8n8k4_f64(c2, c3, sDfrag[128 + (ks * 8 << 2) + fbase], b, c2, c3);
    }
    __syncthreads();
    const int dzA = sw_dz15((warp * 8 + 2 * colk) + 256 * row);
    const int dzB = sw_dz15((warp * 8 + 2 * colk) + 256 * (row + 8));
    sbuf[dzA] = c0;
    sbuf[dzA ^ 1] = c1;
    sbuf[dzB] = c2;
    sbuf[dzB ^ 1] = c3;
  }
  __syncthreads();
  {
    const int dA = sw_dz15(outA);
    const int dB = sw_dz15(outB);
    const double2 ea =
        *reinterpret_cast<const double2 *>(Escale + gA + 2 * npoint);
    const double2 eb =
        *reinterpret_cast<const double2 *>(Escale + gB + 2 * npoint);
    acc0 += ea.x * sbuf[dA];
    acc1 += ea.y * sbuf[dA ^ 1];
    acc2 += eb.x * sbuf[dB];
    acc3 += eb.y * sbuf[dB ^ 1];
  }
  __syncthreads();


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
  tendency_fused_p15_tc_kernel<<<Ne, 1024, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, Escale,
      Ne);
  check_cuda("tendency_fused_p15_tc_kernel");
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
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
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
    double *dqdt, const double *D1D, const double *q, const double *v,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sFV[1024];

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

  for (int kl = 0; kl < JSLAB31; ++kl) {
    *reinterpret_cast<double2 *>(sFV + ldsh) =
        make_double2(qp.x * vp.x, qp.y * vp.y);
    if (kl + 1 < JSLAB31) {
      gidx += NQ31 * NQ31;
      qp = *reinterpret_cast<const double2 *>(q + gidx);
      vp = *reinterpret_cast<const double2 *>(v + gidx);
    }
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
    double2 out = *reinterpret_cast<const double2 *>(dqdt + nidx);
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
#define BK63 16

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

__global__ __launch_bounds__(P63_THREADS, 1) void tendency_fused_p63_xz_tc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *w, const double *flux_bnd,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sFU[NQ63 * BK63];
  __shared__ __align__(16) double sD[NQ63 * BK63];
  __shared__ __align__(16) double sFW[NQ63 * BK63];

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
    // sFU[k][l] = q*u at (l, jp, k).  l is fast in global, so sixteen lanes
    // walk l and cover one 128-byte run.
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      const int ll = tid & 15;
      const int o = (tid >> 4) + (P63_THREADS / 16) * p;
      const int g = eo + (kk + ll) + plane_off + NQ2_63 * o;
      sFU[sw255(ll + BK63 * o)] = q[g] * u[g];
    }
    // sD[r][l] = D1D(r, l), r fast in the Fortran column-major operator.
    // sFW[i][l] = q*w at (i, jp, l), i fast in global.
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      const int o = tid & 63;
      const int ll = (tid >> 6) + (P63_THREADS / 64) * p;
      sD[sw255(ll + BK63 * o)] = D1D[o + NQ63 * (kk + ll)];
      const int g = eo + o + plane_off + NQ2_63 * (kk + ll);
      sFW[sw255(ll + BK63 * o)] = q[g] * w[g];
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK63 / 4; ++ks) {
      const int l = 4 * ks + colk;
      double av[2], bv[P63_TN], avz[2], bvz[P63_TN];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int m = 8 * (2 * wm + a) + row;
        av[a] = sFU[sw255(l + BK63 * m)];
        avz[a] = sD[sw255(l + BK63 * m)];
      }
#pragma unroll
      for (int bb = 0; bb < P63_TN; ++bb) {
        const int n = 8 * (P63_TN * wn + bb) + row;
        bv[bb] = sD[sw255(l + BK63 * n)];
        bvz[bb] = sFW[sw255(l + BK63 * n)];
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
    __syncthreads();
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
__global__ __launch_bounds__(P63_THREADS, 1) void tendency_fused_p63_y_tc_kernel(
    double *dqdt, const double *D1D, const double *q, const double *v,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sD[NQ63 * BK63];
  __shared__ __align__(16) double sFV[NQ63 * BK63];

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

  double acc[2 * 2 * P63_TN];
#pragma unroll
  for (int e = 0; e < 2 * 2 * P63_TN; ++e) {
    acc[e] = 0.0;
  }

  for (int kk = 0; kk < NQ63; kk += BK63) {
#pragma unroll
    for (int p = 0; p < P63_STAGE_ITERS; ++p) {
      const int o = tid & 63;
      const int ll = (tid >> 6) + (P63_THREADS / 64) * p;
      sD[sw255(ll + BK63 * o)] = D1D[o + NQ63 * (kk + ll)];
      const int g = eo + o + NQ63 * (kk + ll) + plane_off;
      sFV[sw255(ll + BK63 * o)] = q[g] * v[g];
    }
    __syncthreads();

#pragma unroll
    for (int ks = 0; ks < BK63 / 4; ++ks) {
      const int l = 4 * ks + colk;
      double av[2], bv[P63_TN];
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        av[a] = sD[sw255(l + BK63 * (8 * (2 * wm + a) + row))];
      }
#pragma unroll
      for (int bb = 0; bb < P63_TN; ++bb) {
        bv[bb] = sFV[sw255(l + BK63 * (8 * (P63_TN * wn + bb) + row))];
      }
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int bb = 0; bb < P63_TN; ++bb) {
          const int e = 2 * (P63_TN * a + bb);
          mma_m8n8k4_f64(acc[e], acc[e + 1], av[a], bv[bb], acc[e], acc[e + 1]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int e8 = 0; e8 < 2 * P63_TN; ++e8) {
    const int a = e8 / P63_TN;
    const int bb = e8 % P63_TN;
    const int m = 8 * (2 * wm + a) + row;                 // j
    const int n = 8 * (P63_TN * wn + bb) + 2 * colk;      // i, and i+1
    const int node = eo + n + NQ63 * m + plane_off;
    const double2 ey =
        *reinterpret_cast<const double2 *>(Escale + node + npoint);
    double2 out = *reinterpret_cast<const double2 *>(dqdt + node);
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
  tendency_fused_p63_xz_tc_kernel<<<nblock, P63_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, Lift1D, q, u, w, flux_bnd, Escale, Ne);
  tendency_fused_p63_y_tc_kernel<<<nblock, P63_THREADS, 0, dg_cuda_stream>>>(
      dqdt, D1D, q, v, Escale, Ne);
  check_cuda("tendency_fused_p63_tc kernels");
}
