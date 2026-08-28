!> CPU/OpenACC build stub for the optional CUDA Fortran kernels.
module mod_cuda_dg_kernels
  use mod_common, only: RP
  implicit none
  private

  logical, public, parameter :: cuda_dg_kernels_available = .false.

  public :: cuda_cal_volume_flux
  public :: cuda_cal_volume_deriv
  public :: cuda_cal_surface_lift
  public :: cuda_assemble_dqdt
  public :: cuda_cal_dqdt_split
  public :: cuda_cal_dqdt_fused
  public :: cuda_cal_dqdt_fused_p63
  public :: cuda_cal_dqdt_fused_p127
  public :: cuda_cal_dqdt_fused_p255
  public :: cuda_cal_dqdt_fused_tc
  public :: cuda_cal_dqdt_fused_p63_tc
  public :: cuda_cal_dqdt_fused_p127_tc
  public :: cuda_cal_dqdt_fused_p255_tc
  public :: cuda_cal_dqdt_gemm
  public :: cuda_cal_dqdt_gemm_fused
  public :: cuda_cal_dqdt_gemm_cute
  public :: cuda_cal_dqdt_gemm_ozaki2
  public :: cuda_ozaki2_init
  public :: cuda_ozaki2_alloc_workspace
  public :: cuda_ozaki2_finalize
  public :: cuda_cal_dqdt_gemm_ozaki1
  public :: cuda_ozaki1_init
  public :: cuda_ozaki1_alloc_workspace
  public :: cuda_ozaki1_finalize
  public :: cuda_ozaki1_slice_stats_set_enabled
  public :: cuda_ozaki1_slice_stats_set_verbose
  public :: cuda_ozaki1_slice_stats_begin_step
  public :: cuda_ozaki1_slice_stats_end_step
  public :: cuda_ozaki1_slice_stats_print
  public :: cuda_gemm_setup
  public :: cuda_cutlass_set_mma_shape
  public :: cuda_gemm_finalize
  public :: cuda_cal_elembnd_flux
  public :: cuda_dg_bind_acc_stream
  public :: cuda_dg_flush_kernel_time
  public :: cuda_dg_set_event_timing
  public :: cuda_dg_set_side_stream
  public :: cuda_dg_graph_capture_begin
  public :: cuda_dg_graph_capture_end
  public :: cuda_dg_graph_launch
  public :: cuda_dg_graph_is_ready
  public :: cuda_dg_graph_finalize

contains
  !> No-op counterpart of the CUDA Fortran routine: without CUDA kernels the
  !! OpenACC queue does not have to share a stream with anything.
  subroutine cuda_dg_bind_acc_stream(queue)
    implicit none
    integer, intent(in) :: queue

    return
  end subroutine cuda_dg_bind_acc_stream

  !> No CUDA events are recorded in this build, so there is nothing to read.
  subroutine cuda_dg_flush_kernel_time(kernel_time)
    implicit none
    real(RP), intent(out) :: kernel_time(4)

    kernel_time(:) = 0.0_RP

    return
  end subroutine cuda_dg_flush_kernel_time

  !> No CUDA events are recorded in this build, so the switch does nothing.
  subroutine cuda_dg_set_event_timing(on)
    implicit none
    logical, intent(in) :: on

    return
  end subroutine cuda_dg_set_event_timing

  subroutine cuda_dg_set_side_stream(on)
    implicit none
    logical, intent(in) :: on
    !------------------------------------------------------------

    return
  end subroutine cuda_dg_set_side_stream

  !- No-op counterparts of the CUDA graph routines.  Without CUDA Fortran
  !  there is no stream to capture, and cuda_dg_graph_is_ready() stays
  !  .false. so that callers keep launching the step directly.

  subroutine cuda_dg_graph_capture_begin()
    implicit none

    return
  end subroutine cuda_dg_graph_capture_begin

  subroutine cuda_dg_graph_capture_end()
    implicit none

    return
  end subroutine cuda_dg_graph_capture_end

  subroutine cuda_dg_graph_launch()
    implicit none

    return
  end subroutine cuda_dg_graph_launch

  logical function cuda_dg_graph_is_ready()
    implicit none

    cuda_dg_graph_is_ready = .false.

    return
  end function cuda_dg_graph_is_ready

  subroutine cuda_dg_graph_finalize()
    implicit none

    return
  end subroutine cuda_dg_graph_finalize

  subroutine cuda_cal_elembnd_flux( &
    flux, q, u, v, w, VMapM, VMapP, normal_fn, Fscale, &
    Np, NfpTot, Ne, NeA )
    implicit none
    integer, intent(in) :: Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: flux(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_elembnd_flux

  subroutine cuda_cal_dqdt_fused( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne), kernel_time(2)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused

  subroutine cuda_cal_dqdt_fused_p63( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: kernel_time(2)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_p63

  subroutine cuda_cal_dqdt_fused_p127( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: kernel_time(2)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_p127

  subroutine cuda_cal_dqdt_fused_p255( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: kernel_time(2)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_p255

  subroutine cuda_cal_dqdt_fused_tc( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne), kernel_time(2)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_tc

  subroutine cuda_cal_dqdt_fused_p63_tc( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: kernel_time(2)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_p63_tc

  subroutine cuda_cal_dqdt_fused_p127_tc( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: kernel_time(2)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_p127_tc

  subroutine cuda_cal_dqdt_fused_p255_tc( &
    dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: kernel_time(2)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_fused_p255_tc

  subroutine cuda_gemm_setup(emulate)
    implicit none
    logical, intent(in) :: emulate
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_gemm_setup

  subroutine cuda_cutlass_set_mma_shape(shape_id)
    implicit none
    integer, intent(in) :: shape_id
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cutlass_set_mma_shape

  subroutine cuda_gemm_finalize()
    implicit none
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_gemm_finalize

  subroutine cuda_cal_dqdt_gemm( &
    dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    flux_x, flux_y, flux_z, deriv_x, deriv_y, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq), Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne)
    real(RP), intent(out) :: kernel_time(2)
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_gemm

  subroutine cuda_ozaki2_init(moduli_count)
    implicit none
    integer, intent(in) :: moduli_count
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_ozaki2_init

  subroutine cuda_ozaki2_alloc_workspace(Nq, Ne, Np)
    implicit none
    integer, intent(in) :: Nq, Ne, Np
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_ozaki2_alloc_workspace

  subroutine cuda_ozaki2_finalize()
    implicit none
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_ozaki2_finalize

  subroutine cuda_cal_dqdt_gemm_ozaki2( &
    dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    flux_x, flux_y, flux_z, deriv_x, deriv_y, deriv_z, lift_out, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq), Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(out) :: lift_out(Np,Ne)
    real(RP), intent(out) :: kernel_time(2)
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_gemm_ozaki2

  subroutine cuda_ozaki1_init(slice_count)
    implicit none
    integer, intent(in) :: slice_count
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_ozaki1_init

  subroutine cuda_ozaki1_alloc_workspace(Nq, Ne, Np)
    implicit none
    integer, intent(in) :: Nq, Ne, Np
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_ozaki1_alloc_workspace

  subroutine cuda_ozaki1_finalize()
    implicit none
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_ozaki1_finalize

  subroutine cuda_ozaki1_slice_stats_set_enabled(enabled)
    implicit none
    integer, intent(in) :: enabled
    return
  end subroutine cuda_ozaki1_slice_stats_set_enabled

  subroutine cuda_ozaki1_slice_stats_set_verbose(verbose)
    implicit none
    integer, intent(in) :: verbose
    return
  end subroutine cuda_ozaki1_slice_stats_set_verbose

  subroutine cuda_ozaki1_slice_stats_begin_step()
    implicit none
    return
  end subroutine cuda_ozaki1_slice_stats_begin_step

  subroutine cuda_ozaki1_slice_stats_end_step()
    implicit none
    return
  end subroutine cuda_ozaki1_slice_stats_end_step

  subroutine cuda_ozaki1_slice_stats_print()
    implicit none
    return
  end subroutine cuda_ozaki1_slice_stats_print

  subroutine cuda_cal_dqdt_gemm_ozaki1( &
    dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    flux_x, flux_y, flux_z, deriv_x, deriv_y, deriv_z, lift_out, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq), Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(out) :: lift_out(Np,Ne)
    real(RP), intent(out) :: kernel_time(2)
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_gemm_ozaki1

  subroutine cuda_cal_dqdt_gemm_cute( &
    dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    flux_x, flux_y, flux_z, deriv_x, deriv_y, deriv_z, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq), Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(out) :: kernel_time(2)
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_gemm_cute

  subroutine cuda_cal_dqdt_gemm_fused( &
    dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
    normal_fn, Fscale, Escale, flux_bnd, &
    flux_x, flux_y, flux_z, deriv_x, deriv_y, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq), Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne)
    real(RP), intent(out) :: kernel_time(2)
    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_gemm_fused

  subroutine cuda_cal_dqdt_split( &
    flux_x, flux_y, flux_z, deriv_x, deriv_y, deriv_z, lift_out, dqdt, &
    q, u, v, w, D1D, D1D_tr, Lift_mat, flux_bnd, Escale, &
    Nq, Np, NfpTot, Ne, NeA, kernel_time )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(out) :: lift_out(Np,Ne), dqdt(Np,Ne), kernel_time(4)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6), flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_dqdt_split

  subroutine cuda_cal_volume_flux( &
    flux_x, flux_y, flux_z, q, u, v, w, Np, Ne, NeA )
    implicit none
    integer, intent(in) :: Np, Ne, NeA
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_volume_flux

  subroutine cuda_cal_volume_deriv( &
    deriv_x, deriv_y, deriv_z, D1D, D1D_tr, &
    flux_x, flux_y, flux_z, Nq, Np, Ne )
    implicit none
    integer, intent(in) :: Nq, Np, Ne
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_volume_deriv

  subroutine cuda_cal_surface_lift( &
    lift_out, Lift_mat, flux_bnd, Nq, Np, NfpTot, Ne )
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne
    real(RP), intent(out) :: lift_out(Np,Ne)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_cal_surface_lift

  subroutine cuda_assemble_dqdt( &
    dqdt, Escale, deriv_x, deriv_y, deriv_z, lift_in, Np, Ne, NeA )
    implicit none
    integer, intent(in) :: Np, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(in) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(in) :: lift_in(Np,Ne)

    error stop "CUDA Fortran kernels are not available in this build"
  end subroutine cuda_assemble_dqdt
end module mod_cuda_dg_kernels
