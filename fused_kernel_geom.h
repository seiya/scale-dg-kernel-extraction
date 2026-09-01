#ifndef FUSED_KERNEL_GEOM_H
#define FUSED_KERNEL_GEOM_H

// Geometry constants for the iso-schedule fused C++ kernels.
// CUDAFORTRAN_FUSED_DFMA and CUDAFORTRAN_FUSED_TC both instantiate
// cuda_dg_kernels_tc.cu. CUDAFORTRAN_FUSED (CUDA-core schedule) lives in
// cuda_dg_kernels_fused.cu / cuda_dg_kernels_fused_highp.cu and must not
// reuse TC thread counts (p=31 CC is 1024 threads, not P31_THREADS).

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
// p=63 xz chunk-loop pipelining, the same knob as P127_XZ_DB.  At the default
// BK63 = NQ63 the loop runs once and there is nothing to pipeline; the switch
// is only meaningful together with a smaller BK63, and measured that way it
// loses: 489.6 us/stage at BK63 = NQ63 against 503.1 / 504.0 / 503.6 at
// BK63 = 8 / 16 / 32 double buffered, ranges disjoint.  The warp shape is
// flat over every P63_WN that fits (485.7 to 489.6, ranges overlapping), so
// the combination that pays 4.25% at p=127 pays nothing here.  Section 52 of
// p63_gap_study.md.
#ifndef P63_XZ_DB
#define P63_XZ_DB 0
#endif
#define P63_XZ_NBUF (P63_XZ_DB + 1)
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
#define BKD127 16
#endif
#define P127_MT 64
#ifndef P127_Y_TM
#define P127_Y_TM 4
#endif
#ifndef P127_Y_TN
#define P127_Y_TN 2
#endif
#ifndef P127_Y_BPSM
#define P127_Y_BPSM 2
#endif
#define P127_Y_WM (NQ127 / (8 * P127_Y_TM))
#define P127_Y_WN (P127_MT / (8 * P127_Y_TN))
#define P127_Y_THREADS (32 * P127_Y_WM * P127_Y_WN)
#define BKDY127 32
#define P127_Y_FSTAGE_ITERS (P127_MT * NQ127 / P127_Y_THREADS)
// p=127 xz warp shape and chunk-loop pipelining.  The block tile is 128x64
// in every setting; TM and TN are the mma tiles a warp holds in m and n, so
// the warp grid and the thread count follow from them.  P127_XZ_DB switches
// the chunk loop to the double-buffered form of tendency_p255_kernel, which
// doubles the two chunked panels in shared memory.  P127_XZ_MINB is the
// second __launch_bounds__ argument, i.e. the register budget.
#ifndef P127_XZ_TM
#define P127_XZ_TM 2
#endif
#ifndef P127_XZ_TN
#define P127_XZ_TN 4
#endif
#ifndef P127_XZ_DB
#define P127_XZ_DB 1
#endif
#ifndef P127_XZ_MINB
#define P127_XZ_MINB 1
#endif
#define P127_XZ_WM (NQ127 / (8 * P127_XZ_TM))
#define P127_XZ_WN (P127_MT / (8 * P127_XZ_TN))
#define P127_XZ_THREADS (32 * P127_XZ_WM * P127_XZ_WN)
#define P127_XZ_NBUF (P127_XZ_DB + 1)

#define NQ255 256
#define BM255 64
#define BN255 64
#define BK255 16
#define TM255 4
#define TN255 4
#define TH255 128
#define MINB255 3

#endif
