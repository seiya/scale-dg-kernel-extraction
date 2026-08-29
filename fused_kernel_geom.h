#ifndef FUSED_KERNEL_GEOM_H
#define FUSED_KERNEL_GEOM_H

// Shared launch geometry for CUDAFORTRAN_FUSED and CUDAFORTRAN_FUSED_TC.
// Both instantiations of cuda_dg_kernels_tc.cu include this file.

#define NQ7 8
#define NP7 512
#define NFPTOT7 384
#define P7_THREADS 256
#define P7_BPSM 8

#define NQ15 16
#define NP15 4096
#define NFPTOT15 1536
#define P15_THREADS 1024

#define NQ31 32
#define NP31 32768
#define NFPTOT31 6144
#define JSLAB31 16
#define P31_THREADS 512

#define NQ63 64
#define NP63 262144
#define NQ2_63 4096
#define NFPTOT63 24576
#define BK63 64
#define P63_WN 4
#define P63_TN (8 / P63_WN)
#define P63_THREADS (32 * 4 * P63_WN)
#define P63_STAGE_ITERS (NQ63 * BK63 / P63_THREADS)
#define P63_BPSM 1
#define P63Y_WN 4
#define P63Y_TN (8 / P63Y_WN)
#define P63Y_THREADS (32 * 4 * P63Y_WN)
#define P63Y_STAGE_ITERS (NQ63 * BK63 / P63Y_THREADS)
#define P63Y_BPSM 2

#define NQ127 128
#define NP127 2097152
#define NQ2_127 16384
#define NFPTOT127 98304
#ifndef BKD127
#define BKD127 64
#endif
#define P127_MT 64
#define P127_Y_THREADS 1024
#define P127_Y_FSTAGE_ITERS (P127_MT * NQ127 / P127_Y_THREADS)
#define P127_XZ_THREADS 1024

#define NQ255 256
#define BM255 64
#define BN255 64
#define BK255 16
#define TM255 4
#define TN255 4
#define TH255 128
#define MINB255 3

#endif
