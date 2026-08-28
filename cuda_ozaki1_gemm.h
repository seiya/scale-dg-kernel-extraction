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

void ozaki1_slice_stats_set_enabled(int enabled);
void ozaki1_slice_stats_set_verbose(int verbose);
void ozaki1_slice_stats_begin_step(void);
void ozaki1_slice_stats_end_step(void);
void ozaki1_slice_stats_print(void);

#ifdef __cplusplus
}
#endif

#endif
