// Does emulating the x, y and z volume GEMMs together help?
//
// Two things "combining" could buy:
//   (A) one INT8 GEMM per modulus over the concatenated N = 3*Nq^2 instead of
//       three GEMMs of N = Nq^2   -> tests whether the INT8 rate improves
//   (B) one reconstruction pass that reads all 3*s INT32 residue sets, applies
//       Escale per direction, adds the lift and writes dqdt once, replacing
//       three separate reconstructions plus dqdt_assembly_kernel
//
// (B) is the only structural saving available: the three derivatives cannot
// share INT32 accumulators, because dqdt_assembly_kernel weights each one by a
// different pointwise field (Escale(:,:,1..3)) before summing.  The weighting
// happens in FP64 after reconstruction, so the residues must stay separate.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);} } while(0)
#define BK(x) do { cublasStatus_t st_=(x); if(st_!=CUBLAS_STATUS_SUCCESS){printf("cublas %d @%d\n",(int)st_,__LINE__);exit(1);} } while(0)

static const int NQ = 256, NQ2 = NQ * NQ;

template <class F> static double med(F body, int it, int warm)
{
  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i=0;i<warm;++i) body();
  CK(cudaDeviceSynchronize());
  std::vector<double> us;
  for (int i=0;i<it;++i){ float ms; CK(cudaEventRecord(a)); body();
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    CK(cudaEventElapsedTime(&ms,a,b)); us.push_back(ms*1000.0); }
  std::sort(us.begin(),us.end());
  cudaEventDestroy(a); cudaEventDestroy(b);
  return us[us.size()/2];
}

// native assembly, as in dqdt_assembly_kernel
__global__ void assembly(double *__restrict__ dqdt, const double *__restrict__ Es,
                         const double *__restrict__ dx, const double *__restrict__ dy,
                         const double *__restrict__ dz, const double *__restrict__ lift,
                         long long n)
{
  for (long long i = blockIdx.x*(long long)blockDim.x+threadIdx.x; i<n;
       i += (long long)gridDim.x*blockDim.x)
    dqdt[i] = -(Es[i]*dx[i] + Es[i+n]*dy[i] + Es[i+2*n]*dz[i] + lift[i]);
}

// one reconstruction per direction: s INT32 residues -> one FP64 derivative
__global__ void recon_one(double *__restrict__ out, const int *const *__restrict__ r,
                          int s, long long n)
{
  for (long long i = blockIdx.x*(long long)blockDim.x+threadIdx.x; i<n;
       i += (long long)gridDim.x*blockDim.x) {
    double a = 0.0;
    for (int j=0;j<s;++j) a += 1.5*(double)r[j][i];
    out[i] = a;
  }
}

// fused: 3*s residue sets -> Escale-weighted sum + lift -> dqdt, one pass
__global__ void recon_fused(double *__restrict__ dqdt, const double *__restrict__ Es,
                            const int *const *__restrict__ r,
                            const double *__restrict__ lift, int s, long long n)
{
  for (long long i = blockIdx.x*(long long)blockDim.x+threadIdx.x; i<n;
       i += (long long)gridDim.x*blockDim.x) {
    double ax=0.0, ay=0.0, az=0.0;
    for (int j=0;j<s;++j)        ax += 1.5*(double)r[j][i];
    for (int j=s;j<2*s;++j)      ay += 1.5*(double)r[j][i];
    for (int j=2*s;j<3*s;++j)    az += 1.5*(double)r[j][i];
    dqdt[i] = -(Es[i]*ax + Es[i+n]*ay + Es[i+2*n]*az + lift[i]);
  }
}

int main(int argc, char **argv)
{
  const int s = (argc>1)?atoi(argv[1]):14;
  cublasHandle_t h; BK(cublasCreate(&h));
  const long long MN = (long long)NQ * NQ2;          // 16.78 M points
  const double gflop = 2.0*NQ*(double)NQ2*NQ/1e9;    // per direction

  double *dA,*dB,*dX,*dY,*dZ,*dEs,*dLift,*dQ;
  CK(cudaMalloc(&dA,(size_t)NQ*NQ*8)); CK(cudaMalloc(&dB,MN*8));
  CK(cudaMalloc(&dX,MN*8)); CK(cudaMalloc(&dY,MN*8)); CK(cudaMalloc(&dZ,MN*8));
  CK(cudaMalloc(&dEs,3*MN*8)); CK(cudaMalloc(&dLift,MN*8)); CK(cudaMalloc(&dQ,MN*8));
  CK(cudaMemset(dA,0x3f,(size_t)NQ*NQ*8)); CK(cudaMemset(dB,0x3f,MN*8));
  CK(cudaMemset(dEs,0x3f,3*MN*8)); CK(cudaMemset(dLift,0x3f,MN*8));

  signed char *iA,*iB3; int *iCcat;
  CK(cudaMalloc(&iA,(size_t)NQ*NQ)); CK(cudaMalloc(&iB3,3*MN));
  CK(cudaMalloc(&iCcat,3*MN*sizeof(int)));
  CK(cudaMemset(iA,1,(size_t)NQ*NQ)); CK(cudaMemset(iB3,1,3*MN));

  std::vector<int*> hp(3*s);
  for (int j=0;j<3*s;++j) CK(cudaMalloc(&hp[j], MN*sizeof(int)));
  int **dp3, **dp1;
  CK(cudaMalloc(&dp3,3*s*sizeof(int*))); CK(cudaMalloc(&dp1,s*sizeof(int*)));
  CK(cudaMemcpy(dp3,hp.data(),3*s*sizeof(int*),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dp1,hp.data(),  s*sizeof(int*),cudaMemcpyHostToDevice));

  const double d1=1.0,d0=0.0; const int i1=1,i0=0;
  const long long sp = (long long)NQ*NQ;

  printf("# combining the x/y/z volume GEMMs, Nq=%d Ne=1, s=%d\n", NQ, s);
  printf("# per-direction work %.2f GFLOP, per-direction output %.1f MB fp64 / %.1f MB int32\n\n",
         gflop, MN*8/1e6, MN*4/1e6);

  // ---------------- native baseline ----------------
  double t_nx = med([&]{ BK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,NQ,(int)NQ2,NQ,
                          &d1,dA,NQ,dB,NQ,&d0,dX,NQ)); },15,5);
  double t_ny = med([&]{ BK(cublasDgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,NQ,NQ,NQ,
                          &d1,dB,NQ,sp,dA,NQ,0,&d0,dY,NQ,sp,NQ)); },15,5);
  double t_nz = med([&]{ BK(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_T,(int)NQ2,NQ,NQ,
                          &d1,dB,(int)NQ2,dA,NQ,&d0,dZ,(int)NQ2)); },15,5);
  double t_as = med([&]{ assembly<<<4096,256>>>(dQ,dEs,dX,dY,dZ,dLift,MN); },15,5);
  double native = t_nx+t_ny+t_nz+t_as;
  printf("native  x  DGEMM          : %8.1f us  (%5.2f TFLOP/s)\n", t_nx, gflop/1e3/(t_nx*1e-6));
  printf("native  y  DGEMM batched  : %8.1f us  (%5.2f TFLOP/s)\n", t_ny, gflop/1e3/(t_ny*1e-6));
  printf("native  z  DGEMM          : %8.1f us  (%5.2f TFLOP/s)\n", t_nz, gflop/1e3/(t_nz*1e-6));
  printf("native  assembly kernel   : %8.1f us  (%5.2f TB/s)\n", t_as, MN*64.0/(t_as*1e-6)/1e12);
  printf("native  x+y+z+assembly    : %8.1f us\n\n", native);

  // ---------------- (A) INT8: 3 separate vs one concatenated ----------------
  double t_i3 = med([&]{
    for (int d=0;d<3;++d)
      for (int j=0;j<s;++j)
        BK(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,NQ,(int)NQ2,NQ,&i1,
                        iA,CUDA_R_8I,NQ, iB3+d*MN,CUDA_R_8I,NQ,&i0,
                        hp[d*s+j],CUDA_R_32I,NQ,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
  },11,4);
  double t_ic = med([&]{
    for (int j=0;j<s;++j)
      BK(cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,NQ,(int)(3*NQ2),NQ,&i1,
                      iA,CUDA_R_8I,NQ, iB3,CUDA_R_8I,NQ,&i0,
                      iCcat,CUDA_R_32I,NQ,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
  },11,4);
  printf("(A) int8 %2dx3 GEMM, N=Nq^2 each : %8.1f us  (%6.1f TOP/s)\n",
         s, t_i3, 3*s*gflop/1e3/(t_i3*1e-6));
  printf("(A) int8 %2dx1 GEMM, N=3*Nq^2    : %8.1f us  (%6.1f TOP/s)\n",
         s, t_ic, 3*s*gflop/1e3/(t_ic*1e-6));
  printf("    -> concatenating N: %+.1f%%\n\n", 100.0*(t_ic-t_i3)/t_i3);

  // ---------------- (B) reconstruction: separate vs fused ----------------
  double t_r3 = med([&]{
    for (int d=0;d<3;++d) {
      CK(cudaMemcpy(dp1,hp.data()+d*s,s*sizeof(int*),cudaMemcpyHostToDevice));
      recon_one<<<4096,256>>>(d==0?dX:(d==1?dY:dZ), dp1, s, MN);
    }
    assembly<<<4096,256>>>(dQ,dEs,dX,dY,dZ,dLift,MN);
  },11,4);
  double t_rf = med([&]{ recon_fused<<<4096,256>>>(dQ,dEs,dp3,dLift,s,MN); },11,4);
  {
    double b3 = 3.0*MN*(4.0*s+8.0) + MN*64.0;
    double bf = MN*(12.0*s + 24.0 + 8.0 + 8.0);
    printf("(B) reconstruct x3 + assembly   : %8.1f us  (%5.2f TB/s, %.2f GB)\n",
           t_r3, b3/(t_r3*1e-6)/1e12, b3/1e9);
    printf("(B) fused reconstruct+assembly  : %8.1f us  (%5.2f TB/s, %.2f GB)\n",
           t_rf, bf/(t_rf*1e-6)/1e12, bf/1e9);
    printf("    -> fusing the reconstruction: %+.1f%%\n\n", 100.0*(t_rf-t_r3)/t_r3);
  }

  printf("totals vs native %.1f us:\n", native);
  printf("  emulated, everything separate : %8.1f us  -> %.2fx\n", t_i3+t_r3, (t_i3+t_r3)/native);
  printf("  emulated, both combinations   : %8.1f us  -> %.2fx\n", t_ic+t_rf, (t_ic+t_rf)/native);
  return 0;
}
