#ifndef FUSED_KERNEL_GEOM_H
#define FUSED_KERNEL_GEOM_H

// Geometry constants for the iso-schedule fused C++ kernels.
// CUDAFORTRAN_FUSED_DFMA and CUDAFORTRAN_FUSED_TC both instantiate
// cuda_dg_kernels_tc.cu. CUDAFORTRAN_FUSED (CUDA-core schedule) lives in
// cuda_dg_kernels_fused.cu / cuda_dg_kernels_fused_highp.cu and must not
// reuse TC thread counts (p=31 CC is 1024 threads, not P31_THREADS).

// --------------------------------------------------------------------------
// Out-of-role knob: give the UseTc=false (CUDAFORTRAN_FUSED_DFMA)
// instantiation its own __launch_bounds__ minBlocks, i.e. its own register
// budget.  AGENTS.md defines FUSED_DFMA as the same source as FUSED_TC with
// UseTc=false, which makes the register budget shared as well; several orders
// pay for that with a spill that exists only on the DFMA side.  Measuring the
// price of the sharing needs a build in which it is not shared, so this knob
// exists -- default 0, never in production dispatch.  Thread counts are never
// touched, so the schedule is otherwise identical.  See
// reports/dfma_register_budget.md.
#ifndef DFMA_OWN_BPSM
#define DFMA_OWN_BPSM 0
#endif
#if DFMA_OWN_BPSM
#define TCDFMA_BPSM(tc, dfma) ((UseTc) ? (tc) : (dfma))
#else
#define TCDFMA_BPSM(tc, dfma) (tc)
#endif

#define NQ7 8
#define NP7 512
#define NFPTOT7 384
// Output tile (A) knob for the p=7 TC kernel.  P7_TC_KP is how many k planes
// one warp owns.  KP = 1 is the form this kernel had until section 23 of
// reports/tc_paper_survey_2407.09621.md: 256 threads, 2 outputs per thread,
// 32 registers and 8 blocks per SM, i.e. 100% occupancy with the register
// file exactly full.  KP = 2 halves the thread count to 128 and gives every
// thread 4 outputs; at 80 registers and 6 blocks per SM (37.5% occupancy)
// that is 7.15% of the stage, so it is the production form.  KP = 4 (64
// threads, 8 outputs) is +10.5% and ends the ladder.  P7_BPSM is the register
// budget (B) that has to come with the tile: at the KP = 2 tile, 8 blocks per
// SM is +6.6% and 5 blocks is -0.5%, both against the same 6-block form.
#ifndef P7_TC_KP
#define P7_TC_KP 2
#endif
#define P7_THREADS (256 / P7_TC_KP)
#define P7_NWARP (P7_THREADS / 32)
// Overridable so the out-of-role m16n8k8 ablation below can be built at all:
// ptxas refuses "mma with .f64 type and shape .m16n8k8" under the 32-register
// target that __launch_bounds__(256, 8) implies.
#ifndef P7_BPSM
#if P7_TC_KP == 1
#define P7_BPSM 8
#else
#define P7_BPSM 6
#endif
#endif
#ifndef P7_BPSM_DFMA
#define P7_BPSM_DFMA P7_BPSM
#endif
// Section 23.9: hold the two D1D fragment elements in registers instead of
// reloading them from sDfrag for every mma.  Section 14 rejected this at
// KP = 1 because the 32-register budget forced a spill; the KP = 2 budget is
// 80, so the premise changed and it was measured again.
#ifndef P7_TC_DREG
#define P7_TC_DREG 0
#endif
// Out-of-role ablation knob, default 0 = the production m8n8k4 schedule.
// 1 replaces each pair of mma.sync.m8n8k4 in the p=7 kernel with one
// mma.sync.m16n8k8 (upper 8 A rows zero, upper accumulator half discarded).
// AGENTS.md names m8n8k4 in the definition of CUDAFORTRAN_FUSED_TC, so this
// must never be the default; see reports/tc_paper_survey_2407.09621.md
// section 19 for what it measures and why it cannot pay on sm_100.
#ifndef P7_TC_M16N8K8
#define P7_TC_M16N8K8 0
#endif

#define NQ15 16
#define NP15 4096
#define NFPTOT15 1536
#define P15_THREADS 1024
#ifndef P15_MINB
#define P15_MINB 1
#endif
#ifndef P15_MINB_DFMA
#define P15_MINB_DFMA P15_MINB
#endif

#define NQ31 32
#define NP31 32768
#define NFPTOT31 6144
#define JSLAB31 16
#define P31_THREADS 512
// p=31 fused Tensor Core warp mma tiles and plane-loop pipelining.  The block
// tile is the full 32x32 output plane in both kernels, so the warp tile fixes
// the warp grid and hence the thread count; widening a warp tile is paid for
// with warps and therefore with occupancy.  P31_*_DB switches the plane loop
// to the double-buffered form (one barrier per plane instead of two), which
// needs two shared planes and therefore dynamic shared memory.  P31_*_MINB is
// the second __launch_bounds__ argument, i.e. the register budget.  Section 30
// of p31_gap_study.md has the sweep; every setting other than the default
// lost, so the defaults below are the production form.
#ifndef P31_XZ_TM
#define P31_XZ_TM 1
#endif
#ifndef P31_XZ_TN
#define P31_XZ_TN 1
#endif
#ifndef P31_XZ_DB
#define P31_XZ_DB 0
#endif
#ifndef P31_XZ_MINB
#define P31_XZ_MINB 1
#endif
#ifndef P31_XZ_MINB_DFMA
#define P31_XZ_MINB_DFMA P31_XZ_MINB
#endif
#define P31_XZ_WM (4 / P31_XZ_TM)
#define P31_XZ_WN (4 / P31_XZ_TN)
#define P31_XZ_THREADS (32 * P31_XZ_WM * P31_XZ_WN)
#define P31_XZ_NBUF (P31_XZ_DB + 1)
// Which side of the face barrier the first plane load sits on.  Late (0) is
// 1.40% faster at the 1x1 tile; the wide tiles prefer early.
#ifndef P31_XZ_EARLY
#define P31_XZ_EARLY 0
#endif
// Dynamic shared memory is forced by the double-buffered form (57.5 KB against
// a 48 KB static limit) but is a separate knob, because the conversion alone
// costs 1.70% at the 1x1 tile on an otherwise bit-identical kernel -- and is
// 1.5% cheaper than static at the 2x2 tile.
#ifndef P31_XZ_DYN
#define P31_XZ_DYN P31_XZ_DB
#endif
#ifndef P31_Y_TM
#define P31_Y_TM 1
#endif
#ifndef P31_Y_TN
#define P31_Y_TN 1
#endif
#ifndef P31_Y_DB
#define P31_Y_DB 0
#endif
#ifndef P31_Y_MINB
#define P31_Y_MINB 1
#endif
#ifndef P31_Y_MINB_DFMA
#define P31_Y_MINB_DFMA P31_Y_MINB
#endif
#define P31_Y_WM (4 / P31_Y_TM)
#define P31_Y_WN (4 / P31_Y_TN)
#define P31_Y_THREADS (32 * P31_Y_WM * P31_Y_WN)
#define P31_Y_NBUF (P31_Y_DB + 1)

#define NQ63 64
#define NP63 262144
#define NQ2_63 4096
#define NFPTOT63 24576
#define BK63 64
#define P63_WN 4
#define P63_TN (8 / P63_WN)
#define P63_THREADS (32 * 4 * P63_WN)
#define P63_STAGE_ITERS (NQ63 * BK63 / P63_THREADS)
#ifndef P63_BPSM
#define P63_BPSM 1
#endif
#ifndef P63_BPSM_DFMA
#define P63_BPSM_DFMA P63_BPSM
#endif
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
#ifndef P63Y_BPSM
#define P63Y_BPSM 2
#endif
#ifndef P63Y_BPSM_DFMA
#define P63Y_BPSM_DFMA P63Y_BPSM
#endif

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
#ifndef P127_Y_BPSM_DFMA
#define P127_Y_BPSM_DFMA P127_Y_BPSM
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
#ifndef P127_XZ_MINB_DFMA
#define P127_XZ_MINB_DFMA P127_XZ_MINB
#endif
#define P127_XZ_WM (NQ127 / (8 * P127_XZ_TM))
#define P127_XZ_WN (P127_MT / (8 * P127_XZ_TN))
#define P127_XZ_THREADS (32 * P127_XZ_WM * P127_XZ_WN)
#define P127_XZ_NBUF (P127_XZ_DB + 1)

#define NQ255 256
#ifndef BM255
#define BM255 64
#endif
#ifndef BN255
#define BN255 64
#endif
// p=255 fused Tensor Core chunk width and register budget.  BK255 is the
// reduction chunk the two shared panels hold; MINB255 is the second
// __launch_bounds__ argument.  Section 26.3 of p255_gap_study.md measures
// them; every setting other than the default lost, so the defaults below are
// production.  BK255 = 32 halves the chunk count (and hence the barriers) but
// doubles both the shared panels (64 KB, which forces dynamic shared memory)
// and the prefetch registers (48 -> 96).  It does fit -- 255 registers with no
// spill at MINB255 = 2 -- but buying that budget costs 9.79% of the stage and
// the halved barrier count returns nothing: at MINB255 = 2 the BK = 32 form is
// 1.10% slower than BK = 16, because the barrier was never the bound (barrier
// stall 3370 against math_pipe 21450 and wait 17870 in the production form).
// At MINB255 = 3 it spills 100 bytes and costs 31.35%.
#ifndef BK255
#define BK255 16
#endif
#ifndef TM255
#define TM255 4
#endif
#ifndef TN255
#define TN255 4
#endif
#ifndef TH255
#define TH255 128
#endif
#ifndef MINB255
#define MINB255 3
#endif
#ifndef MINB255_DFMA
#define MINB255_DFMA MINB255
#endif
#define P255_SMEM_BYTES ((size_t)2 * (BM255 + BN255) * BK255 * sizeof(double))
#define P255_DYNSMEM (BK255 > 16)

#endif
