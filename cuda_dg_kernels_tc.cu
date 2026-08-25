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

__global__ __launch_bounds__(256, 8) void tendency_fused_p7_tc_kernel(
    double *dqdt, const double *D1D, const double *Lift1D, const double *q,
    const double *u, const double *v, const double *w, const int *VMapM,
    const int *VMapP, const double *normal_fn, const double *Fscale,
    const double *Escale, int Ne)
{
  __shared__ __align__(16) double sDfrag[64];
  __shared__ __align__(16) double sLift[48];
  __shared__ __align__(16) double sflux_bnd[384];
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
  sFluxX[sw_xy(node1)] = q[idx1] * u[idx1];
  sFluxY[sw_xy(node1)] = q[idx1] * v[idx1];
  sFluxZ[sw_z(node1)] = q[idx1] * w[idx1];
  sFluxX[sw_xy(node2)] = q[idx2] * u[idx2];
  sFluxY[sw_xy(node2)] = q[idx2] * v[idx2];
  sFluxZ[sw_z(node2)] = q[idx2] * w[idx2];

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
  sflux_bnd[fp] = 0.5 * Fscale[fidx] * (qP * VelP - qM * VelM - alpha * (qP - qM));
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

__global__ void tendency_x_p255_tc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  const int lane = (int)threadIdx.x & 31;
  const int block = (int)blockIdx.x;
  const int tiles_per_elem = 256 * 32 * 32;
  const int elem = block / tiles_per_elem;
  if (elem >= Ne) {
    return;
  }
  int local = block - elem * tiles_per_elem;
  const int k = local / 1024;
  local -= k * 1024;
  const int tile_j = local / 32;
  const int tile_i = local - tile_j * 32;
  const int i0 = tile_i * 8;
  const int j0 = tile_j * 8;
  const int row = lane >> 2;
  const int colk = lane & 3;
  const int coln = lane >> 2;
  const int rowk = lane & 3;
  const int elem_offset = elem * 16777216;

  double c0 = 0.0, c1 = 0.0;
  for (int kk = 0; kk < 256; kk += 4) {
    const double a = D1D[(i0 + row) + (kk + colk) * 256];
    const int l = kk + rowk;
    const int j = j0 + coln;
    const int idx = elem_offset + l + j * 256 + k * 65536;
    const double b = q[idx] * velocity[idx];
    mma_m8n8k4_f64(c0, c1, a, b, c0, c1);
  }

  const int i = i0 + row;
  const int j_c0 = j0 + (lane & 3) * 2;
  const int elem_face_offset = elem * 6 * 65536;
  const int fp0 = j_c0 + k * 256;
  const int fp1 = fp0 + 1;
  const int idx0 = elem_offset + i + j_c0 * 256 + k * 65536;
  const int idx1 = elem_offset + i + (j_c0 + 1) * 256 + k * 65536;
  const double lift0 = Lift1D[i + 256] * flux_bnd[elem_face_offset + 65536 + fp0] +
                       Lift1D[i + 768] * flux_bnd[elem_face_offset + 196608 + fp0];
  const double lift1 = Lift1D[i + 256] * flux_bnd[elem_face_offset + 65536 + fp1] +
                       Lift1D[i + 768] * flux_bnd[elem_face_offset + 196608 + fp1];
  dqdt[idx0] = -(Escale[idx0] * c0 + lift0);
  dqdt[idx1] = -(Escale[idx1] * c1 + lift1);
}

__global__ void tendency_y_p255_tc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  const int lane = (int)threadIdx.x & 31;
  const int block = (int)blockIdx.x;
  const int tiles_per_elem = 256 * 32 * 32;
  const int elem = block / tiles_per_elem;
  if (elem >= Ne) {
    return;
  }
  int local = block - elem * tiles_per_elem;
  const int k = local / 1024;
  local -= k * 1024;
  const int tile_j = local / 32;
  const int tile_i = local - tile_j * 32;
  const int i0 = tile_i * 8;
  const int j0 = tile_j * 8;
  const int row = lane >> 2;
  const int colk = lane & 3;
  const int coln = lane >> 2;
  const int rowk = lane & 3;
  const int elem_offset = elem * 16777216;
  const int npoint = 16777216 * Ne;

  double c0 = 0.0, c1 = 0.0;
  for (int kk = 0; kk < 256; kk += 4) {
    const int l = kk + colk;
    const int idx = elem_offset + (i0 + row) + l * 256 + k * 65536;
    const double a = q[idx] * velocity[idx];
    const double b = D1D[(j0 + coln) + (kk + rowk) * 256];
    mma_m8n8k4_f64(c0, c1, a, b, c0, c1);
  }

  const int i = i0 + row;
  const int j_c0 = j0 + (lane & 3) * 2;
  const int elem_face_offset = elem * 6 * 65536;
  const int fp = i + k * 256;
  const int idx0 = elem_offset + i + j_c0 * 256 + k * 65536;
  const int idx1 = elem_offset + i + (j_c0 + 1) * 256 + k * 65536;
  const double lift0 = Lift1D[j_c0] * flux_bnd[elem_face_offset + fp] +
                       Lift1D[j_c0 + 512] * flux_bnd[elem_face_offset + 131072 + fp];
  const double lift1 = Lift1D[j_c0 + 1] * flux_bnd[elem_face_offset + fp] +
                       Lift1D[j_c0 + 1 + 512] * flux_bnd[elem_face_offset + 131072 + fp];
  dqdt[idx0] -= Escale[idx0 + npoint] * c0 + lift0;
  dqdt[idx1] -= Escale[idx1 + npoint] * c1 + lift1;
}

__global__ void tendency_z_p255_tc_kernel(
    double *dqdt, const double *q, const double *velocity, const double *D1D,
    const double *Lift1D, const double *flux_bnd, const double *Escale, int Ne)
{
  const int lane = (int)threadIdx.x & 31;
  const int block = (int)blockIdx.x;
  const int tiles_per_elem = 8192 * 32;
  const int elem = block / tiles_per_elem;
  if (elem >= Ne) {
    return;
  }
  int local = block - elem * tiles_per_elem;
  const int tile_k = local / 8192;
  const int tile_line = local - tile_k * 8192;
  const int line0 = tile_line * 8;
  const int k0_out = tile_k * 8;
  const int row = lane >> 2;
  const int colk = lane & 3;
  const int coln = lane >> 2;
  const int rowk = lane & 3;
  const int elem_offset = elem * 16777216;
  const int npoint = 16777216 * Ne;

  double c0 = 0.0, c1 = 0.0;
  for (int kk = 0; kk < 256; kk += 4) {
    const int line = line0 + row;
    const int l = kk + colk;
    const int idx = elem_offset + line + l * 65536;
    const double a = q[idx] * velocity[idx];
    const double b = D1D[(k0_out + coln) + (kk + rowk) * 256];
    mma_m8n8k4_f64(c0, c1, a, b, c0, c1);
  }

  const int line = line0 + row;
  const int k_c0 = k0_out + (lane & 3) * 2;
  const int elem_face_offset = elem * 6 * 65536;
  const int fp = line;
  const int idx0 = elem_offset + line + k_c0 * 65536;
  const int idx1 = elem_offset + line + (k_c0 + 1) * 65536;
  const double lift0 = Lift1D[k_c0 + 1024] * flux_bnd[elem_face_offset + 262144 + fp] +
                       Lift1D[k_c0 + 1280] * flux_bnd[elem_face_offset + 327680 + fp];
  const double lift1 = Lift1D[k_c0 + 1 + 1024] * flux_bnd[elem_face_offset + 262144 + fp] +
                       Lift1D[k_c0 + 1 + 1280] * flux_bnd[elem_face_offset + 327680 + fp];
  dqdt[idx0] -= Escale[idx0 + 2 * npoint] * c0 + lift0;
  dqdt[idx1] -= Escale[idx1 + 2 * npoint] * c1 + lift1;
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
  const int nblock_xy = 256 * 32 * 32 * Ne;
  const int nblock_z = 8192 * 32 * Ne;
  tendency_x_p255_tc_kernel<<<nblock_xy, 32, 0, dg_cuda_stream>>>(
      dqdt, q, u, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_y_p255_tc_kernel<<<nblock_xy, 32, 0, dg_cuda_stream>>>(
      dqdt, q, v, D1D, Lift1D, flux_bnd, Escale, Ne);
  tendency_z_p255_tc_kernel<<<nblock_z, 32, 0, dg_cuda_stream>>>(
      dqdt, q, w, D1D, Lift1D, flux_bnd, Escale, Ne);
  check_cuda("p255 tensor-core tendency kernels");
}
