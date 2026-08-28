#ifndef CUDA_OZAKI2_GEMM_H
#define CUDA_OZAKI2_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

int ozaki2_init(int moduli_count, int fixed_mantissa);
int ozaki2_finalize(void);
int ozaki2_alloc_workspace(int Nq, int Ne, int Np);
void ozaki2_free_workspace(void);

int ozaki2_dgemm(int transa, int transb, int m, int n, int k, const double *A,
                 int lda, const double *B, int ldb, double *C, int ldc);

int ozaki2_dgemm_strided_batched(int transa, int transb, int m, int n, int k,
                                 const double *A, int lda, long long strideA,
                                 const double *B, int ldb, long long strideB,
                                 double *C, int ldc, long long strideC,
                                 int batch);

#ifdef __cplusplus
}
#endif

#endif
