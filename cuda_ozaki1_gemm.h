#ifndef CUDA_OZAKI1_GEMM_H
#define CUDA_OZAKI1_GEMM_H

#ifdef __cplusplus
extern "C" {
#endif

int ozaki1_init(int slice_count);
int ozaki1_finalize(void);
int ozaki1_alloc_workspace(int Nq, int Ne, int Np);
void ozaki1_free_workspace(void);

int ozaki1_dgemm(int transa, int transb, int m, int n, int k, const double *A,
                 int lda, const double *B, int ldb, double *C, int ldc);

int ozaki1_dgemm_strided_batched(int transa, int transb, int m, int n, int k,
                                 const double *A, int lda, long long strideA,
                                 const double *B, int ldb, long long strideB,
                                 double *C, int ldc, long long strideC,
                                 int batch);

#ifdef __cplusplus
}
#endif

#endif
