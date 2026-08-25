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
// sDx / sDy: plane-transposed with an XOR on the fast index,
//   idx_dxy(i,j,k) = 64*k + 8*i + (j ^ i).
//   The m8n8k4 accumulator holds C[i][2c] and C[i][2c+1] with i = lane>>2, so
//   the natural i + 8*j order made an accumulator store phase hit only 4
//   distinct banks. This layout makes the epilogue read conflict free and
//   leaves the store 2-way conflicting; see the note below.
// sDz: natural node order permuted by sw_dz(), which folds bit 6 (the low bit
//   of the accumulator row) into bit 3 for the same reason.
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
// sDx / sDy / sDz alias sFluxX / sFluxY / sFluxZ. Warp k reads only plane k of
// sFluxX and sFluxY and writes only plane k of sDx and sDy, and both sw_xy()
// and idx_dxy() stay inside a plane, so a __syncwarp() is enough there. The z
// panel is different: sw_z() and sw_dz() move indices across the 8-column
// range a warp owns, so the z accumulators need a block-wide barrier before
// they overwrite the flux. Halving the 28.16 KB of shared memory is what lets
// __launch_bounds__(256, 8) reach 8 blocks per SM; ncu (Slurm job 43734)
// showed the kernel held at 6 blocks and 72% achieved occupancy, limited by
// both registers and shared memory at once.
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

__device__ __forceinline__ int idx_dxy(int i, int j, int k)
{
  return (k << 6) + (i << 3) + (j ^ i);
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
  // The derivative planes overwrite the flux planes they consume, so the
  // block needs 15.87 KB instead of 28.16 KB. See the aliasing note above.
  double *const sDx = sFluxX;
  double *const sDy = sFluxY;
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
  const int coln = lane >> 2;
  const int rowk = lane & 3;
  const int k = warp;
  const int j0_c = (lane & 3) * 2;

  // Dx = D * Fx  on this k-plane
  double c0, c1;
  mma_reset(c0, c1);
  for (int k0 = 0; k0 < 8; k0 += 4) {
    const double a = sDfrag[(k0 << 3) + (row << 2) + colk];
    const double b = sFluxX[sw_xy((k0 + rowk) + (coln << 3) + (k << 6))];
    mma_m8n8k4_f64(c0, c1, a, b, c0, c1);
  }
  __syncwarp();
  sDx[idx_dxy(row, j0_c, k)] = c0;
  sDx[idx_dxy(row, j0_c + 1, k)] = c1;

  // Dy = Fy * D^T
  mma_reset(c0, c1);
  for (int k0 = 0; k0 < 8; k0 += 4) {
    const double a = sFluxY[sw_xy(row + ((k0 + colk) << 3) + (k << 6))];
    const double b = sDfrag[(k0 << 3) + (coln << 2) + rowk];
    mma_m8n8k4_f64(c0, c1, a, b, c0, c1);
  }
  __syncwarp();
  sDy[idx_dxy(row, j0_c, k)] = c0;
  sDy[idx_dxy(row, j0_c + 1, k)] = c1;

  // Dz = D * Fz_panel; warp owns 8 (i,j) columns, all k
  mma_reset(c0, c1);
  for (int k0 = 0; k0 < 8; k0 += 4) {
    const int ij = (warp << 3) + coln;
    const double a = sDfrag[(k0 << 3) + (row << 2) + colk];
    const double b = sFluxZ[sw_z(ij + ((k0 + rowk) << 6))];
    mma_m8n8k4_f64(c0, c1, a, b, c0, c1);
  }
  // sw_z() and sw_dz() permute across warp boundaries, so unlike the x and y
  // planes the z panel needs a block-wide barrier before it is overwritten.
  __syncthreads();
  sDz[sw_dz(((warp << 3) + j0_c) + (row << 6))] = c0;
  sDz[sw_dz(((warp << 3) + j0_c + 1) + (row << 6))] = c1;
  __syncthreads();

  const int i = node1 & 7;
  const int j = (node1 >> 3) & 7;
  const int k1 = node1 >> 6;
  const int k2 = k1 + 4;
  const int dxy1 = idx_dxy(i, j, k1);
  const int dxy2 = idx_dxy(i, j, k2);
  const int dz1 = sw_dz(node1);
  const int dz2 = sw_dz(node2);
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
  const double lf1 = sLift[j];
  const double lf2 = sLift[i + 8];
  const double lf3 = sLift[j + 16];
  const double lf4 = sLift[i + 24];
  const double lift1 = lf1 * sflux_bnd[face11] + lf2 * sflux_bnd[face12] +
                       lf3 * sflux_bnd[face13] + lf4 * sflux_bnd[face14] +
                       sLift[k1 + 32] * sflux_bnd[face15] +
                       sLift[k1 + 40] * sflux_bnd[face16];
  const double lift2 = lf1 * sflux_bnd[face21] + lf2 * sflux_bnd[face22] +
                       lf3 * sflux_bnd[face23] + lf4 * sflux_bnd[face24] +
                       sLift[k2 + 32] * sflux_bnd[face15] +
                       sLift[k2 + 40] * sflux_bnd[face16];

  dqdt[idx1] = -(Escale[idx1] * sDx[dxy1] + Escale[idx1 + npoint] * sDy[dxy1] +
                 Escale[idx1 + 2 * npoint] * sDz[dz1] + lift1);
  dqdt[idx2] = -(Escale[idx2] * sDx[dxy2] + Escale[idx2 + npoint] * sDy[dxy2] +
                 Escale[idx2 + 2 * npoint] * sDz[dz2] + lift2);
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
