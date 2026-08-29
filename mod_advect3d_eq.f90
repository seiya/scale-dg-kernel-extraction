!-------------------------------------------------------------------------------
!> module mesh
!!
!! @par Description
!!      A module to calculate the tendency 3D advection equation
!!
!! @author Yuta Kawai, Xuanzhengbo Ren, Team SCALE
!<
module mod_advect3d_eq
  use mod_common, only: RP, ACC_QUEUE, &
  Timer, Timer_start, Timer_stop, Timer_add, Timer_elapsed, Timer_reset
  use mod_cuda_dg_kernels, only: &
    cuda_dg_kernels_available,  &
    cuda_cal_volume_flux, cuda_cal_volume_deriv, &
    cuda_cal_surface_lift, cuda_assemble_dqdt, cuda_cal_dqdt_split, &
    cuda_cal_dqdt_fused, cuda_cal_dqdt_fused_dfma, &
    cuda_cal_dqdt_fused_p63, cuda_cal_dqdt_fused_p63_dfma, cuda_cal_dqdt_fused_p255, &
    cuda_cal_dqdt_fused_p255_dfma, &
    cuda_cal_dqdt_fused_p127, cuda_cal_dqdt_fused_p127_dfma, cuda_cal_dqdt_fused_p127_tc, &
    cuda_cal_dqdt_fused_tc, cuda_cal_dqdt_fused_p63_tc, &
    cuda_cal_dqdt_fused_p255_tc, &
    cuda_cal_dqdt_gemm, cuda_cal_dqdt_gemm_fused, cuda_cal_dqdt_gemm_cute, &
    cuda_cal_dqdt_gemm_ozaki2, cuda_ozaki2_init, cuda_ozaki2_alloc_workspace, &
    cuda_ozaki2_finalize, cuda_cal_dqdt_gemm_ozaki1, cuda_ozaki1_init, &
    cuda_ozaki1_alloc_workspace, cuda_ozaki1_finalize, &
    cuda_ozaki1_slice_stats_set_enabled, cuda_ozaki1_slice_stats_set_verbose, &
    cuda_ozaki1_slice_stats_begin_step, cuda_ozaki1_slice_stats_end_step, &
    cuda_ozaki1_slice_stats_print, &
    cuda_gemm_setup, cuda_cutlass_set_mma_shape, cuda_gemm_finalize, &
    cuda_cal_elembnd_flux, cuda_dg_bind_acc_stream, &
    cuda_dg_flush_kernel_time, cuda_dg_set_event_timing, &
    cuda_dg_set_side_stream, &
    cuda_dg_graph_capture_begin, cuda_dg_graph_capture_end, &
    cuda_dg_graph_launch, cuda_dg_graph_is_ready, cuda_dg_graph_finalize
  implicit none
  private

  public :: setup_advect3d_eq_setup
  public :: setup_advect3d_eq_finalize
  public :: advect3d_eq_cal_tend
  public :: advect3d_eq_graph_supported
  public :: advect3d_eq_set_time_reporting
  public :: advect3d_eq_reset_timers
  public :: advect3d_eq_ozaki1_slice_stats_begin_step
  public :: advect3d_eq_ozaki1_slice_stats_end_step

  !- Re-exported so that the time-stepping loop does not have to know which
  !  backend module it is built against.
  public :: cuda_dg_set_event_timing
  public :: cuda_dg_set_side_stream
  public :: cuda_dg_graph_capture_begin
  public :: cuda_dg_graph_capture_end
  public :: cuda_dg_graph_launch
  public :: cuda_dg_graph_is_ready

  !> .true. for the paths whose tendency is computed by OpenACC kernels.
  !! Those kernels run on the default OpenACC queue, so the device work that
  !! the time-stepping loop queued on ACC_QUEUE must be complete before they
  !! start.
  logical :: tend_uses_acc_kernels = .true.

  !> .false. when the host-side timers of a tendency call no longer see every
  !! call, which is what a CUDA graph replay does.  The elapsed times are then
  !! reported as not measured instead of as a number that counts only the
  !! steps the host actually walked through.
  logical :: tend_time_is_measured = .true.

  type(Timer) :: timer_ebnd_flux
  type(Timer) :: timer_dqdt
  type(Timer) :: timer_volume_flux
  type(Timer) :: timer_volume_deriv
  type(Timer) :: timer_surface_lift
  type(Timer) :: timer_dqdt_assemble

  integer, parameter :: DQDT_KERNEL_OPENACC_ASIS  = 1
  integer, parameter :: DQDT_KERNEL_OPENACC_SPLIT = 2
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_SPLIT = 3
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_FUSED = 4
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_FUSED_TC = 5
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_GEMM = 6
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED = 7
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE = 8
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 = 9
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1 = 10
  integer, parameter :: DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA = 11
  integer :: dqdt_kernel_typeid
  character(len=24) :: dqdt_kernel_name
  logical :: cublas_emulation_enabled = .false.
  logical :: emulation_mantissa_fixed = .true.
  integer :: emulation_mantissa_bits = 55
  integer :: ozaki_moduli_count = 7
  integer :: ozaki_slice_count = 7
  logical :: ozaki1_slice_stats_enabled = .false.
  !> MMA instruction shape of the CUTLASS volume GEMMs (GEMM_CUTE / GEMM_FUSED).
  !! 0 = SM80 8x8x4, 1 = SM90 16x8x4, 2 = SM90 16x8x8, 3 = SM90 16x8x16.
  integer :: cutlass_mma_shape_id = 0

  real(RP), allocatable :: ebnd_flux(:,:)
  real(RP), allocatable :: volume_flux_x(:,:)
  real(RP), allocatable :: volume_flux_y(:,:)
  real(RP), allocatable :: volume_flux_z(:,:)
  real(RP), allocatable :: volume_deriv_x(:,:)
  real(RP), allocatable :: volume_deriv_y(:,:)
  real(RP), allocatable :: volume_deriv_z(:,:)
  real(RP), allocatable :: surface_lift(:,:)
  real(RP), allocatable :: fused_flux_bnd(:,:)
contains
  !> Setup
!OCL SERIAL
  subroutine setup_advect3d_eq_setup(NfpTot, Np, Ne, dqdt_kernel_type, cublas_emulation, &
    cutlass_mma_shape, ozaki_moduli_count_in, ozaki_slice_count_in, &
    emulation_mantissa_control, emulation_mantissa_bits_in)
    implicit none
    integer, intent(in) :: NfpTot, Np, Ne
    character(len=*), intent(in) :: dqdt_kernel_type
    logical, intent(in), optional :: cublas_emulation
    character(len=*), intent(in), optional :: cutlass_mma_shape
    integer, intent(in), optional :: ozaki_moduli_count_in
    integer, intent(in), optional :: ozaki_slice_count_in
    character(len=*), intent(in), optional :: emulation_mantissa_control
    integer, intent(in), optional :: emulation_mantissa_bits_in
    integer :: Nq
    character(len=16) :: mantissa_ctrl
    !------------------------------------------------------------------------------
    cublas_emulation_enabled = .false.
    if (present(cublas_emulation)) cublas_emulation_enabled = cublas_emulation
    ozaki_moduli_count = 7
    ozaki_slice_count = 7
    if (present(ozaki_moduli_count_in)) ozaki_moduli_count = ozaki_moduli_count_in
    if (present(ozaki_slice_count_in)) ozaki_slice_count = ozaki_slice_count_in

    emulation_mantissa_fixed = .true.
    emulation_mantissa_bits = 55
    if (present(emulation_mantissa_bits_in)) then
      emulation_mantissa_bits = emulation_mantissa_bits_in
    end if
    mantissa_ctrl = "FIXED"
    if (present(emulation_mantissa_control)) mantissa_ctrl = emulation_mantissa_control
    select case (trim(adjustl(mantissa_ctrl)))
    case ("FIXED", "fixed")
      emulation_mantissa_fixed = .true.
    case ("DYNAMIC", "ADP", "dynamic", "adp")
      emulation_mantissa_fixed = .false.
    case default
      write(*,*) "Unsupported EmulationMantissaControl: ", trim(mantissa_ctrl)
      write(*,*) "Choose FIXED or DYNAMIC"
      error stop
    end select
    if (emulation_mantissa_bits < 1 .or. emulation_mantissa_bits > 127) then
      write(*,*) "EmulationMantissaBits must be in [1, 127], got", emulation_mantissa_bits
      error stop
    end if

    cutlass_mma_shape_id = 0
    if (present(cutlass_mma_shape)) then
      select case (trim(cutlass_mma_shape))
      case ("8x8x4", "")
        cutlass_mma_shape_id = 0
      case ("16x8x4")
        cutlass_mma_shape_id = 1
      case ("16x8x8")
        cutlass_mma_shape_id = 2
      case ("16x8x16")
        cutlass_mma_shape_id = 3
      case default
        write(*,*) "Unsupported CutlassMmaShape: ", trim(cutlass_mma_shape)
        write(*,*) "Choose 8x8x4, 16x8x4, 16x8x8 or 16x8x16"
        error stop
      end select
    end if

    !- Bind the OpenACC queue before creating a cuBLAS handle so its stream
    !  and persistent workspace can be configured once during GEMM setup.
    call cuda_dg_bind_acc_stream(ACC_QUEUE)

    select case (trim(dqdt_kernel_type))
    case ("OPENACC_ASIS")
      dqdt_kernel_typeid = DQDT_KERNEL_OPENACC_ASIS
      dqdt_kernel_name = "OPENACC_ASIS"
    case ("OPENACC_SPLIT")
      dqdt_kernel_typeid = DQDT_KERNEL_OPENACC_SPLIT
      dqdt_kernel_name = "OPENACC_SPLIT"
    case ("CUDAFORTRAN_SPLIT")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_SPLIT requires a build with CUDA=1"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_SPLIT
      dqdt_kernel_name = "CUDAFORTRAN_SPLIT"
    case ("CUDAFORTRAN_FUSED")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_FUSED requires a build with CUDA=1"
        error stop
      end if
      if (Np /= 512 .and. Np /= 16**3 .and. Np /= 32**3 .and. &
          Np /= 64**3 .and. Np /= 128**3 .and. Np /= 256**3) then
        write(*,*) "CUDAFORTRAN_FUSED requires PolyOrder=7, 15, 31, 63, 127 or 255"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_FUSED
      dqdt_kernel_name = "CUDAFORTRAN_FUSED"
    case ("CUDAFORTRAN_FUSED_DFMA")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_FUSED_DFMA requires a build with CUDA=1"
        error stop
      end if
      if (Np /= 512 .and. Np /= 16**3 .and. Np /= 32**3 .and. &
          Np /= 64**3 .and. Np /= 128**3 .and. Np /= 256**3) then
        write(*,*) "CUDAFORTRAN_FUSED_DFMA requires PolyOrder=7, 15, 31, 63, 127 or 255"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA
      dqdt_kernel_name = "CUDAFORTRAN_FUSED_DFMA"
    case ("CUDAFORTRAN_FUSED_TC")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_FUSED_TC requires a build with CUDA=1"
        error stop
      end if
      if (Np /= 512 .and. Np /= 16**3 .and. Np /= 32**3 .and. &
          Np /= 64**3 .and. Np /= 128**3 .and. Np /= 256**3) then
        write(*,*) "CUDAFORTRAN_FUSED_TC requires PolyOrder=7, 15, 31, 63, 127 or 255"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_FUSED_TC
      dqdt_kernel_name = "CUDAFORTRAN_FUSED_TC"
    case ("CUDAFORTRAN_GEMM")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_GEMM requires a build with CUDA=1"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_GEMM
      dqdt_kernel_name = "CUDAFORTRAN_GEMM"
      call cuda_gemm_setup(cublas_emulation_enabled, emulation_mantissa_fixed, &
        emulation_mantissa_bits)
    case ("CUDAFORTRAN_GEMM_FUSED")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_GEMM_FUSED requires a build with CUDA=1"
        error stop
      end if
      if (nint(sqrt(real(NfpTot/6)))*Ne > 65535) then
        write(*,*) "CUDAFORTRAN_GEMM_FUSED needs Nq*Ne <= 65535 (CUTLASS batch on grid.z)"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED
      dqdt_kernel_name = "CUDAFORTRAN_GEMM_FUSED"
      call cuda_gemm_setup(cublas_emulation_enabled, emulation_mantissa_fixed, &
        emulation_mantissa_bits)
      call cuda_cutlass_set_mma_shape(cutlass_mma_shape_id)
    case ("CUDAFORTRAN_GEMM_CUTE")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_GEMM_CUTE requires a build with CUDA=1"
        error stop
      end if
      if (nint(sqrt(real(NfpTot/6)))*Ne > 65535) then
        write(*,*) "CUDAFORTRAN_GEMM_CUTE needs Nq*Ne <= 65535 (CUTLASS batch on grid.z)"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE
      dqdt_kernel_name = "CUDAFORTRAN_GEMM_CUTE"
      call cuda_gemm_setup(cublas_emulation_enabled, emulation_mantissa_fixed, &
        emulation_mantissa_bits)
      call cuda_cutlass_set_mma_shape(cutlass_mma_shape_id)
    case ("CUDAFORTRAN_GEMM_OZAKI2")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_GEMM_OZAKI2 requires a build with CUDA=1"
        error stop
      end if
      if (ozaki_moduli_count < 2 .or. ozaki_moduli_count > 20) then
        write(*,*) "OzakiModuliCount must be in [2, 20], got", ozaki_moduli_count
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2
      dqdt_kernel_name = "CUDAFORTRAN_GEMM_OZAKI2"
      call cuda_gemm_setup(.false.)
      call cuda_ozaki2_init(ozaki_moduli_count, emulation_mantissa_fixed)
      Nq = nint(sqrt(real(NfpTot)/6.0_RP))
      call cuda_ozaki2_alloc_workspace(Nq, Ne, Np)
    case ("CUDAFORTRAN_GEMM_OZAKI1")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_GEMM_OZAKI1 requires a build with CUDA=1"
        error stop
      end if
      if (ozaki_slice_count < 2 .or. ozaki_slice_count > 16) then
        write(*,*) "OzakiSliceCount must be in [2, 16], got", ozaki_slice_count
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1
      dqdt_kernel_name = "CUDAFORTRAN_GEMM_OZAKI1"
      call cuda_gemm_setup(.false.)
      call cuda_ozaki1_init(ozaki_slice_count, emulation_mantissa_fixed)
      Nq = nint(sqrt(real(NfpTot)/6.0_RP))
      call cuda_ozaki1_alloc_workspace(Nq, Ne, Np)
      call setup_ozaki1_slice_stats_from_env()
    case default
      write(*,*) "Unsupported dqdt_kernel_type: ", trim(dqdt_kernel_type)
      error stop
    end select

    tend_uses_acc_kernels = &
      dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_ASIS .or. &
      dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_SPLIT .or. &
      dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_SPLIT

    if (cublas_emulation_enabled .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE) then
      write(*,*) "CublasEmulation is ignored unless DqdtKernel_Type uses GEMM"
    end if

    if (cutlass_mma_shape_id /= 0 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE) then
      write(*,*) "CutlassMmaShape is ignored unless DqdtKernel_Type is ", &
        "CUDAFORTRAN_GEMM_FUSED or CUDAFORTRAN_GEMM_CUTE"
    end if

    if (Np == 256**3 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      error stop "PolyOrder=255 currently requires CUDAFORTRAN_FUSED, CUDAFORTRAN_FUSED_TC, CUDAFORTRAN_FUSED_DFMA, CUDAFORTRAN_GEMM, CUDAFORTRAN_GEMM_FUSED, CUDAFORTRAN_GEMM_CUTE, CUDAFORTRAN_GEMM_OZAKI2, or CUDAFORTRAN_GEMM_OZAKI1"
    end if

    if ((Np == 512**3 .or. Np == 576**3 .or. Np == 768**3 .or. &
         Np == 1024**3) .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      error stop "PolyOrder=511, 575, 767, or 1023 requires CUDAFORTRAN_GEMM, CUDAFORTRAN_GEMM_FUSED, CUDAFORTRAN_GEMM_OZAKI2, or CUDAFORTRAN_GEMM_OZAKI1"
    end if

    if (dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA) then
      allocate(ebnd_flux(NfpTot,Ne))
      !$acc enter data create(ebnd_flux)
    end if
    !- The p=63, p=127 and p=255 fused paths evaluate the six face fluxes in a
    !  separate pass instead of inside the volume kernels.  At p=63 and above
    !  Ne is at most 4**3 while the GPU has 152 SMs, so an element has to
    !  be split over many blocks; evaluating the faces in the volume kernel
    !  would then repeat the (i,k) faces once per block.
    if ((dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED .or. &
         dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .or. &
         dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA) .and. &
        (Np == 256**3 .or. Np == 128**3 .or. Np == 64**3)) then
      allocate(fused_flux_bnd(NfpTot,Ne))
      !$acc enter data create(fused_flux_bnd)
    end if

    if (dqdt_kernel_typeid /= DQDT_KERNEL_OPENACC_ASIS .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA) then
      if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM .or. &
          dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .or. &
          dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .or. &
          dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1 .or. &
          dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2) then
        allocate(volume_deriv_x(Np,Ne), volume_deriv_y(Np,Ne))
        !$acc enter data create(volume_deriv_x,volume_deriv_y)
      else
        allocate(volume_deriv_x(Np,Ne), volume_deriv_y(Np,Ne), volume_deriv_z(Np,Ne))
        !$acc enter data create(volume_deriv_x,volume_deriv_y,volume_deriv_z)
      end if
    end if
    if (dqdt_kernel_typeid /= DQDT_KERNEL_OPENACC_ASIS .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2) then
      allocate(surface_lift(Np,Ne))
      !$acc enter data create(surface_lift)
    end if
    if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_SPLIT .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_SPLIT .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      allocate(volume_flux_x(Np,Ne), volume_flux_y(Np,Ne), volume_flux_z(Np,Ne))
      !$acc enter data create(volume_flux_x,volume_flux_y,volume_flux_z)
    end if

    return
  end subroutine setup_advect3d_eq_setup

  !> .true. when every kernel of a tendency call is queued on ACC_QUEUE, so
  !! that a whole time step can be captured from that one stream.  The
  !! OpenACC tendency paths launch on the default queue instead, which a
  !! capture of ACC_QUEUE would not see.
  logical function advect3d_eq_graph_supported()
    implicit none
    !------------------------------------------------------------

    advect3d_eq_graph_supported = &
      cuda_dg_kernels_available .and. (.not. tend_uses_acc_kernels) .and. &
      dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .and. &
      dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1

    return
  end function advect3d_eq_graph_supported

  !> Tell the module whether its host-side timers still see every tendency
  !! call.  See tend_time_is_measured.
  subroutine advect3d_eq_set_time_reporting(measured)
    implicit none
    logical, intent(in) :: measured
    !------------------------------------------------------------

    tend_time_is_measured = measured

    return
  end subroutine advect3d_eq_set_time_reporting

  !> Drop what the tendency timers accumulated so far.
  !!
  !! Called by the time loop once the warm-up steps are over, so that the
  !! report covers the measured steps only.  The CUDA device time of a call is
  !! read back on the following call, so the first device time accumulated
  !! after a reset is still the one of the last warm-up call: the number of
  !! accumulated measurements is right, the window is shifted by one call.
!OCL SERIAL
  subroutine advect3d_eq_reset_timers()
    implicit none
    !------------------------------------------------------------

    call Timer_reset(timer_ebnd_flux)
    call Timer_reset(timer_dqdt)
    call Timer_reset(timer_volume_flux)
    call Timer_reset(timer_volume_deriv)
    call Timer_reset(timer_surface_lift)
    call Timer_reset(timer_dqdt_assemble)

    return
  end subroutine advect3d_eq_reset_timers

  subroutine setup_ozaki1_slice_stats_from_env()
    implicit none
    character(len=256) :: envval
    !------------------------------------------------------------

    ozaki1_slice_stats_enabled = .false.
    envval = ''
    call get_environment_variable('SCALE_DG_OZAKI1_SLICE_STATS', envval)
    if (len_trim(envval) == 0) return

    ozaki1_slice_stats_enabled = .true.
    call cuda_ozaki1_slice_stats_set_enabled(1)

    envval = ''
    call get_environment_variable('SCALE_DG_OZAKI1_SLICE_STATS_VERBOSE', envval)
    if (len_trim(envval) > 0) call cuda_ozaki1_slice_stats_set_verbose(1)

    return
  end subroutine setup_ozaki1_slice_stats_from_env

  subroutine advect3d_eq_ozaki1_slice_stats_begin_step()
    implicit none
    !------------------------------------------------------------

    if (.not. ozaki1_slice_stats_enabled) return
    if (dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) return
    call cuda_ozaki1_slice_stats_begin_step()

    return
  end subroutine advect3d_eq_ozaki1_slice_stats_begin_step

  subroutine advect3d_eq_ozaki1_slice_stats_end_step()
    implicit none
    !------------------------------------------------------------

    if (.not. ozaki1_slice_stats_enabled) return
    if (dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) return
    call cuda_ozaki1_slice_stats_end_step()

    return
  end subroutine advect3d_eq_ozaki1_slice_stats_end_step

  !> Finalize
!OCL SERIAL
  subroutine setup_advect3d_eq_finalize()
    implicit none

    real(RP) :: kernel_time(4)
    !------------------------------------------------------------------------------

    !- The CUDA device time of a tendency call is read back on the next call,
    !  so the measurement of the last call is still pending here.
    call cuda_dg_flush_kernel_time(kernel_time)
    call accumulate_kernel_time(kernel_time)

    call cuda_dg_graph_finalize()

    write(*,'(A30,A24)') "Dqdt kernel type:", trim(dqdt_kernel_name)
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE) then
      if (cublas_emulation_enabled) then
        if (emulation_mantissa_fixed) then
          write(*,'(A30,A24)') "Cublas FP emulation:", "on (FIXED)"
        else
          write(*,'(A30,A24)') "Cublas FP emulation:", "on (DYNAMIC/ADP)"
        end if
        write(*,'(A30,I10)') "  Emulation mantissa bits:", emulation_mantissa_bits
      else
        write(*,'(A30,A24)') "Cublas FP emulation:", "off"
      end if
    end if
    if (.not. tend_time_is_measured) then
      !- A CUDA graph replay runs no Fortran wrapper, so none of the timers
      !  below saw the steps that were replayed.
      write(*,'(A30,A24)') "Tendency breakdown:", "not measured (graph)"
      return
    end if

    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA) then
      write(*,'(A30,1X,A23)') "Element boundary flux:", "included in fused kernel"
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM .or. &
             dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .or. &
             dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .or. &
             dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      write(*,'(A30,1X,A23)') "Element boundary flux:", "included in GEMM path"
    else
      write(*,'(A30,ES24.5)') "Element boundary flux:", Timer_elapsed(timer_ebnd_flux)
    end if
    write(*,'(A30,ES24.5)') "Volume derivate + surface lift:", Timer_elapsed(timer_dqdt)

    if (dqdt_kernel_typeid /= DQDT_KERNEL_OPENACC_ASIS) then
      if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_SPLIT) then
        write(*,'(A30,ES24.5)') "  CUDA device volume flux:", Timer_elapsed(timer_volume_flux)
        write(*,'(A30,ES24.5)') "  CUDA device derivative:", Timer_elapsed(timer_volume_deriv)
        write(*,'(A30,ES24.5)') "  CUDA device surface lift:", Timer_elapsed(timer_surface_lift)
        write(*,'(A30,ES24.5)') "  CUDA device assembly:", Timer_elapsed(timer_dqdt_assemble)
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED .or. &
               dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .or. &
               dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA) then
        write(*,'(A30,ES24.5)') "  CUDA device fused tendency:", Timer_elapsed(timer_volume_deriv)
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
        write(*,'(A30,ES24.5)') "  CUDA device GEMM tendency:", Timer_elapsed(timer_volume_deriv)
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2) then
        write(*,'(A30,ES24.5)') "  CUDA device Ozaki-II GEMM:", Timer_elapsed(timer_volume_deriv)
        write(*,'(A30,I10)') "  Ozaki moduli count:", ozaki_moduli_count
        if (emulation_mantissa_fixed) then
          write(*,'(A30,A24)') "  Ozaki mantissa control:", "FIXED"
        else
          write(*,'(A30,A24)') "  Ozaki mantissa control:", "DYNAMIC"
        end if
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
        write(*,'(A30,ES24.5)') "  CUDA device Ozaki-I GEMM:", Timer_elapsed(timer_volume_deriv)
        write(*,'(A30,I10)') "  Ozaki slice count:", ozaki_slice_count
        if (emulation_mantissa_fixed) then
          write(*,'(A30,A24)') "  Ozaki mantissa control:", "FIXED"
        else
          write(*,'(A30,A24)') "  Ozaki mantissa control:", "DYNAMIC"
        end if
        if (ozaki1_slice_stats_enabled) then
          call cuda_ozaki1_slice_stats_print()
        end if
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED) then
        write(*,'(A30,ES24.5)') "  CUDA device GEMM fused:", Timer_elapsed(timer_volume_deriv)
        write(*,'(A30,ES24.5)') "  FUSED volume GEMM only:", Timer_elapsed(timer_surface_lift)
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE) then
        write(*,'(A30,ES24.5)') "  CUDA device GEMM CUTE:", Timer_elapsed(timer_volume_deriv)
        write(*,'(A30,ES24.5)') "  CUTE volume GEMM only:", Timer_elapsed(timer_volume_flux)
      else
        write(*,'(A30,ES24.5)') "  Volume flux:", Timer_elapsed(timer_volume_flux)
        write(*,'(A30,ES24.5)') "  Tensor-product derivative:", Timer_elapsed(timer_volume_deriv)
        write(*,'(A30,ES24.5)') "  Surface lift:", Timer_elapsed(timer_surface_lift)
        write(*,'(A30,ES24.5)') "  Dqdt assembly:", Timer_elapsed(timer_dqdt_assemble)
      end if
      if (allocated(volume_flux_x)) then
        !$acc exit data delete(volume_flux_x,volume_flux_y,volume_flux_z)
        deallocate(volume_flux_x, volume_flux_y, volume_flux_z)
      end if
      if (allocated(volume_deriv_x)) then
        if (allocated(volume_deriv_z)) then
          !$acc exit data delete(volume_deriv_x,volume_deriv_y,volume_deriv_z)
          deallocate(volume_deriv_x, volume_deriv_y, volume_deriv_z)
        else
          !$acc exit data delete(volume_deriv_x,volume_deriv_y)
          deallocate(volume_deriv_x, volume_deriv_y)
        end if
      end if
      if (allocated(surface_lift)) then
        !$acc exit data delete(surface_lift)
        deallocate(surface_lift)
      end if
    end if

    if (allocated(ebnd_flux)) then
      !$acc exit data delete(ebnd_flux)
      deallocate(ebnd_flux)
    end if
    if (allocated(fused_flux_bnd)) then
      !$acc exit data delete(fused_flux_bnd)
      deallocate(fused_flux_bnd)
    end if
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      call cuda_gemm_finalize()
    end if
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2) then
      call cuda_ozaki2_finalize()
    end if
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      call cuda_ozaki1_finalize()
    end if
    return
  end subroutine setup_advect3d_eq_finalize

  !> Add one device-time measurement to the timers of the active path.
!OCL SERIAL
  subroutine accumulate_kernel_time(kernel_time)
    implicit none
    real(RP), intent(in) :: kernel_time(:)
    !------------------------------------------------------------

    select case (dqdt_kernel_typeid)
    case (DQDT_KERNEL_CUDAFORTRAN_SPLIT)
      call Timer_add(timer_volume_flux,kernel_time(1))
      call Timer_add(timer_volume_deriv,kernel_time(2))
      call Timer_add(timer_surface_lift,kernel_time(3))
      call Timer_add(timer_dqdt_assemble,kernel_time(4))
    case (DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE)
      call Timer_add(timer_volume_deriv,kernel_time(1))
      call Timer_add(timer_volume_flux,kernel_time(2))
    case (DQDT_KERNEL_CUDAFORTRAN_FUSED, DQDT_KERNEL_CUDAFORTRAN_FUSED_TC, &
          DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA, &
          DQDT_KERNEL_CUDAFORTRAN_GEMM, DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED, &
          DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2, &
          DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1)
      call Timer_add(timer_volume_deriv,kernel_time(1))
      call Timer_add(timer_surface_lift,kernel_time(2))
    end select

    return
  end subroutine accumulate_kernel_time

  !> Calculate the tendency of 3D advection equation
!OCL SERIAL
  subroutine advect3d_eq_cal_tend( dqdt, & ! (out)
    q, u, v, w,                              & ! (in)
    D1D, D1D_tr, Lift_mat, Lift1D,           & ! (in)
    VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA )                  ! (in)

     implicit none
    integer, intent(in) :: Nq
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA)
    real(RP), intent(in) :: v(Np,NeA)
    real(RP), intent(in) :: w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(:,:,:,:)
    real(RP), intent(in) :: Lift1D(Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(in) :: Fscale(NfpTot,Ne)
    !------------------------------------------------------------

    if (tend_uses_acc_kernels) then
      !- The tendency kernels of these paths are OpenACC regions on the
      !  default queue, so they are not ordered against ACC_QUEUE by the
      !  stream binding.
      !$acc wait(ACC_QUEUE)
    end if

    call Timer_start(timer_ebnd_flux)
    if (dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      call cal_elembnd_flux( ebnd_flux,   & ! (out)
         q, u, v, w,                      & ! (in)
         VMapM, VMapP, normal_fn, Fscale, & ! (in)
         Np, NfpTot, Ne, NeA )
    end if
    call Timer_stop(timer_ebnd_flux)

    call Timer_start(timer_dqdt)
    if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_ASIS) then
      call cal_dqdt_openacc_asis( dqdt, & ! (out)
         q, u, v, w,  ebnd_flux,        & ! (in)
         D1D, D1D_tr, Lift_mat,         & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_SPLIT) then
      call cal_dqdt_openacc_split( dqdt, & ! (out)
         q, u, v, w,  ebnd_flux,         & ! (in)
         D1D, D1D_tr, Lift_mat,          & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_SPLIT) then
      call cal_dqdt_cudafortran_split( dqdt, & ! (out)
         q, u, v, w,  ebnd_flux,            & ! (in)
         D1D, D1D_tr, Lift_mat,             & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED) then
      call cal_dqdt_cudafortran_fused( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift_mat, Lift1D,     & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_DFMA) then
      call cal_dqdt_cudafortran_fused_dfma( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift_mat, Lift1D,     & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      call cal_dqdt_cudafortran_gemm( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift1D,               & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_FUSED) then
      call cal_dqdt_cudafortran_gemm_fused( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift1D,               & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_CUTE) then
      call cal_dqdt_cudafortran_gemm_cute( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift1D,               & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI2) then
      call cal_dqdt_cudafortran_gemm_ozaki2( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift1D,               & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM_OZAKI1) then
      call cal_dqdt_cudafortran_gemm_ozaki1( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift1D,               & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    else
      call cal_dqdt_cudafortran_fused_tc( dqdt, & ! (out)
         q, u, v, w,                         & ! (in)
         D1D, D1D_tr, Lift_mat, Lift1D,     & ! (in)
         VMapM, VMapP, normal_fn, Fscale,   & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA )    ! (in)
    end if
    call Timer_stop(timer_dqdt)

     return
  end subroutine advect3d_eq_cal_tend

  !> Calculate the element boundary flux
!OCL SERIAL
  subroutine cal_elembnd_flux( flux, & ! (out)
    q_, u_, v_, w_,                  & ! (in)
    VMapM, VMapP, normal_fn, Fscale, & ! (in)
    Np, NfpTot, Ne, NeA              ) ! (in)
    implicit none
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: flux(NfpTot,Ne) 
    real(RP), intent(in) :: q_(Np*NeA)
    real(RP), intent(in) :: u_(Np*NeA)
    real(RP), intent(in) :: v_(Np*NeA)
    real(RP), intent(in) :: w_(Np*NeA)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3)
    real(RP), intent(in) :: Fscale(NfpTot,Ne)

    integer :: ke, fp
    integer :: iM, iP
    real(RP) :: qM, qP
    real(RP) :: VelM, VelP
    real(RP) :: alpha

    !------------------------------------------

    !$omp parallel do private(fp,iM,iP,qM,qP,VelM,VelP,alpha)
    !$acc parallel loop gang vector collapse(2) &
    !$acc& present(flux,q_,u_,v_,w_,VMapM,VMapP,normal_fn,Fscale)
    do ke = 1, Ne
      do fp = 1, NfpTot
        iM = VMapM(fp,ke)
        iP = VMapP(fp,ke)

        qM = q_(iM)
        qP = q_(iP)

        VelM = u_(iM)*normal_fn(fp,ke,1) &
             + v_(iM)*normal_fn(fp,ke,2) &
             + w_(iM)*normal_fn(fp,ke,3)

        VelP = u_(iP)*normal_fn(fp,ke,1) &
             + v_(iP)*normal_fn(fp,ke,2) &
             + w_(iP)*normal_fn(fp,ke,3)

        alpha = 0.5_RP * abs(VelP + VelM)

        flux(fp,ke) = 0.5_RP * Fscale(fp,ke) * ( &
             qP * VelP - qM * VelM - alpha * (qP - qM) )
      end do
    end do
    return
  end subroutine cal_elembnd_flux

  !> Calculate the volume derivative and apply surface lifting
!OCL SERIAL
  subroutine cal_dqdt_openacc_asis( dqdt, & ! (out)
    q, u, v, w, flux_bnd,          & ! (in)
    D1D, D1D_tr, Lift_mat, Escale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA        ) ! (in)

    use mod_dg_optr_kernel, only: &
      tensorprod_divlike_dirXYZ, &
      tensorprod_Lift_hexahedral
    implicit none
    integer, intent(in) :: Nq
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA)
    real(RP), intent(in) :: v(Np,NeA)
    real(RP), intent(in) :: w(Np,NeA)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)

    integer :: ke, p

    real(RP) :: flux_x(Np), flux_y(Np), flux_z(Np)
    real(RP) :: DxFlux(Np), DyFlux(Np), DzFlux(Np)

    real(RP) :: LiftBndFlux(Np)
    !------------------------------------------------------------

    !$omp parallel do private( p, flux_x, flux_y, flux_z, DxFlux, DyFlux, DzFlux, LiftBndFlux )
    !$acc parallel loop gang &
    !$acc& private(flux_x,flux_y,flux_z,DxFlux,DyFlux,DzFlux,LiftBndFlux) &
    !$acc& present(dqdt,q,u,v,w,flux_bnd,D1D,D1D_tr,Lift_mat,Escale)
    do ke = 1, Ne
      !$acc loop vector
      do p = 1, Np
        flux_x(p) = q(p,ke) * u(p,ke)
        flux_y(p) = q(p,ke) * v(p,ke)
        flux_z(p) = q(p,ke) * w(p,ke)
      end do

      call tensorprod_divlike_dirXYZ( &
        DxFlux, DyFlux, DzFlux,       & ! (out)
        D1D, D1D_tr,                  & ! (in)
        flux_x, flux_y, flux_z, Nq    ) ! (in)

      call tensorprod_Lift_hexahedral( &
        LiftBndFlux,                 & ! (out)
        Lift_mat, flux_bnd(:,ke), Nq ) ! (in)

      !$acc loop vector
      do p = 1, Np
        dqdt(p,ke) = -( &
             Escale(p,ke,1)*DxFlux(p) &
           + Escale(p,ke,2)*DyFlux(p) &
           + Escale(p,ke,3)*DzFlux(p) &
           + LiftBndFlux(p) )
      end do
    end do
    return
  end subroutine cal_dqdt_openacc_asis

  !> Calculate the volume derivative and surface lifting in separate kernels
!OCL SERIAL
  subroutine cal_dqdt_openacc_split( dqdt, & ! (out)
    q, u, v, w, flux_bnd,                  & ! (in)
    D1D, D1D_tr, Lift_mat, Escale,         & ! (in)
    Nq, Np, NfpTot, Ne, NeA                ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    !------------------------------------------------------------

    call Timer_start(timer_volume_flux)
    call cal_volume_flux_openacc_split( &
      volume_flux_x, volume_flux_y, volume_flux_z, & ! (out)
      q, u, v, w, Np, Ne, NeA )                     ! (in)
    call Timer_stop(timer_volume_flux)

    call Timer_start(timer_volume_deriv)
    call cal_volume_deriv_openacc_split( &
      volume_deriv_x, volume_deriv_y, volume_deriv_z, & ! (out)
      D1D, D1D_tr,                                      & ! (in)
      volume_flux_x, volume_flux_y, volume_flux_z,       & ! (in)
      Nq, Np, Ne )                                        ! (in)
    call Timer_stop(timer_volume_deriv)

    call Timer_start(timer_surface_lift)
    call cal_surface_lift_openacc_split( &
      surface_lift, Lift_mat, flux_bnd, Nq, Np, NfpTot, Ne )
    call Timer_stop(timer_surface_lift)

    call Timer_start(timer_dqdt_assemble)
    call assemble_dqdt_openacc_split( &
      dqdt, Escale, volume_deriv_x, volume_deriv_y, &
      volume_deriv_z, surface_lift, Np, Ne, NeA )
    call Timer_stop(timer_dqdt_assemble)

    return
  end subroutine cal_dqdt_openacc_split

  !> Calculate the split tendency using CUDA Fortran kernels
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_split( dqdt, & ! (out)
    q, u, v, w, flux_bnd,                     & ! (in)
    D1D, D1D_tr, Lift_mat, Escale,            & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP) :: kernel_time(4)
    !------------------------------------------------------------

    !$acc host_data use_device(volume_flux_x,volume_flux_y,volume_flux_z) &
    !$acc& use_device(volume_deriv_x,volume_deriv_y,volume_deriv_z) &
    !$acc& use_device(surface_lift,dqdt,q,u,v,w,D1D,D1D_tr) &
    !$acc& use_device(Lift_mat,flux_bnd,Escale)
    call cuda_cal_dqdt_split( &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, volume_deriv_z, surface_lift, dqdt, &
      q, u, v, w, D1D, D1D_tr, Lift_mat, flux_bnd, Escale, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_split

  !> Calculate the tendency with fused volume-flux and derivative generation
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_fused( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift_mat, Lift1D,            & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(:,:,:,:)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    if (Nq == 8 .or. Nq == 16 .or. Nq == 32) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale)
      call cuda_cal_dqdt_fused( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 64) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p63( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 128) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p127( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 256) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p255( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else
      error stop "CUDAFORTRAN_FUSED requires Nq=8, 16, 32, 64, 128 or 256"
    end if

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_fused

  !> Iso-schedule DFMA fused tendency (UseTc=false of the Tensor Core kernels).
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_fused_dfma( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift_mat, Lift1D,            & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(:,:,:,:)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    if (Nq == 8 .or. Nq == 16 .or. Nq == 32) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale)
      call cuda_cal_dqdt_fused_dfma( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 64) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p63_dfma( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 128) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p127_dfma( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 256) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p255_dfma( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else
      error stop "CUDAFORTRAN_FUSED_DFMA requires Nq=8, 16, 32, 64, 128 or 256"
    end if

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_fused_dfma

  !> Tensor-core fused tendency. Shares cuda_dg_kernels_tc.cu with FUSED_DFMA.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_fused_tc( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift_mat, Lift1D,            & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(:,:,:,:)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    if (Nq == 8 .or. Nq == 16 .or. Nq == 32) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale)
      call cuda_cal_dqdt_fused_tc( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 64) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p63_tc( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 128) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p127_tc( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else if (Nq == 256) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift1D,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale,fused_flux_bnd)
      call cuda_cal_dqdt_fused_p255_tc( &
        dqdt, q, u, v, w, D1D, Lift1D, VMapM, VMapP, &
        normal_fn, Fscale, Escale, fused_flux_bnd, &
        Nq, Np, NfpTot, Ne, NeA, kernel_time )
      !$acc end host_data
    else
      error stop "CUDAFORTRAN_FUSED_TC requires Nq=8, 16, 32, 64, 128 or 256"
    end if

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_fused_tc

  !> Tendency path using cuBLAS GEMM for tensor-product derivative and lift.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_gemm( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift1D,                      & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    !$acc host_data use_device(dqdt,q,u,v,w,D1D,D1D_tr,Lift1D,VMapM,VMapP) &
    !$acc& use_device(normal_fn,Fscale,Escale,ebnd_flux) &
    !$acc& use_device(volume_flux_x,volume_flux_y,volume_flux_z) &
    !$acc& use_device(volume_deriv_x,volume_deriv_y)
    call cuda_cal_dqdt_gemm( &
      dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
      normal_fn, Fscale, Escale, ebnd_flux, &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_gemm

  !> Tendency path using Ozaki Scheme II INT8 GEMM emulation for volume derivatives.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_gemm_ozaki2( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift1D,                      & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    !$acc host_data use_device(dqdt,q,u,v,w,D1D,D1D_tr,Lift1D,VMapM,VMapP) &
    !$acc& use_device(normal_fn,Fscale,Escale,ebnd_flux) &
    !$acc& use_device(volume_flux_x,volume_flux_y,volume_flux_z) &
    !$acc& use_device(volume_deriv_x,volume_deriv_y)
    call cuda_cal_dqdt_gemm_ozaki2( &
      dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
      normal_fn, Fscale, Escale, ebnd_flux, &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_gemm_ozaki2

  !> Tendency path using Ozaki Scheme I slice-decomposition INT8 GEMM emulation.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_gemm_ozaki1( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift1D,                      & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    !$acc host_data use_device(dqdt,q,u,v,w,D1D,D1D_tr,Lift1D,VMapM,VMapP) &
    !$acc& use_device(normal_fn,Fscale,Escale,ebnd_flux) &
    !$acc& use_device(volume_flux_x,volume_flux_y,volume_flux_z) &
    !$acc& use_device(volume_deriv_x,volume_deriv_y)
    call cuda_cal_dqdt_gemm_ozaki1( &
      dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
      normal_fn, Fscale, Escale, ebnd_flux, &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_gemm_ozaki1

  !> Same pipeline as CUDAFORTRAN_GEMM, but volume derivatives use the tiled GEMM.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_gemm_cute( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift1D,                      & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    !$acc host_data use_device(dqdt,q,u,v,w,D1D,D1D_tr,Lift1D,VMapM,VMapP) &
    !$acc& use_device(normal_fn,Fscale,Escale,ebnd_flux) &
    !$acc& use_device(volume_flux_x,volume_flux_y,volume_flux_z) &
    !$acc& use_device(volume_deriv_x,volume_deriv_y)
    call cuda_cal_dqdt_gemm_cute( &
      dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
      normal_fn, Fscale, Escale, ebnd_flux, &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_gemm_cute

  !> Tendency path with volume-flux GEMM prologue and z-GEMM assembly epilogue.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_gemm_fused( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift1D,                      & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift1D(Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    integer, intent(in) :: VMapM(NfpTot,Ne), VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3), Fscale(NfpTot,Ne)
    real(RP) :: kernel_time(2)
    !------------------------------------------------------------

    !$acc host_data use_device(dqdt,q,u,v,w,D1D,D1D_tr,Lift1D,VMapM,VMapP) &
    !$acc& use_device(normal_fn,Fscale,Escale,ebnd_flux) &
    !$acc& use_device(volume_flux_x,volume_flux_y,volume_flux_z) &
    !$acc& use_device(volume_deriv_x,volume_deriv_y)
    call cuda_cal_dqdt_gemm_fused( &
      dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
      normal_fn, Fscale, Escale, ebnd_flux, &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call accumulate_kernel_time(kernel_time)

    return
  end subroutine cal_dqdt_cudafortran_gemm_fused

  !> Calculate Cartesian volume fluxes
!OCL SERIAL
  subroutine cal_volume_flux_openacc_split( &
    flux_x, flux_y, flux_z, q, u, v, w, Np, Ne, NeA )
    implicit none
    integer, intent(in) :: Np, Ne, NeA
    real(RP), intent(out) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA), v(Np,NeA), w(Np,NeA)

    integer :: ke, p
    !------------------------------------------------------------

    !$omp parallel do private(p)
    !$acc parallel loop gang vector collapse(2) &
    !$acc& present(flux_x,flux_y,flux_z,q,u,v,w)
    do ke = 1, Ne
      do p = 1, Np
        flux_x(p,ke) = q(p,ke) * u(p,ke)
        flux_y(p,ke) = q(p,ke) * v(p,ke)
        flux_z(p,ke) = q(p,ke) * w(p,ke)
      end do
    end do

    return
  end subroutine cal_volume_flux_openacc_split

  !> Apply the tensor-product derivative to all volume fluxes
!OCL SERIAL
  subroutine cal_volume_deriv_openacc_split( &
    deriv_x, deriv_y, deriv_z, D1D, D1D_tr, &
    flux_x, flux_y, flux_z, Nq, Np, Ne )
    use mod_dg_optr_kernel, only: tensorprod_divlike_dirXYZ_all
    implicit none
    integer, intent(in) :: Nq, Np, Ne
    real(RP), intent(out) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: flux_x(Np,Ne), flux_y(Np,Ne), flux_z(Np,Ne)

    !------------------------------------------------------------

    call tensorprod_divlike_dirXYZ_all( &
      deriv_x, deriv_y, deriv_z, & ! (out)
      D1D, D1D_tr,               & ! (in)
      flux_x, flux_y, flux_z, Nq, Ne ) ! (in)

    return
  end subroutine cal_volume_deriv_openacc_split

  !> Apply the surface lifting operator to all elements
!OCL SERIAL
  subroutine cal_surface_lift_openacc_split( &
    lift_out, Lift_mat, flux_bnd, Nq, Np, NfpTot, Ne )
    use mod_dg_optr_kernel, only: tensorprod_Lift_hexahedral_all
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne
    real(RP), intent(out) :: lift_out(Np,Ne)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)

    !------------------------------------------------------------

    call tensorprod_Lift_hexahedral_all( &
      lift_out, Lift_mat, flux_bnd, Nq, Ne )

    return
  end subroutine cal_surface_lift_openacc_split

  !> Assemble the volume derivative and surface lift into dqdt
!OCL SERIAL
  subroutine assemble_dqdt_openacc_split( &
    dqdt, Escale, deriv_x, deriv_y, deriv_z, lift_in, Np, Ne, NeA )
    implicit none
    integer, intent(in) :: Np, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,Ne)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(in) :: deriv_x(Np,Ne), deriv_y(Np,Ne), deriv_z(Np,Ne)
    real(RP), intent(in) :: lift_in(Np,Ne)

    integer :: ke, p
    !------------------------------------------------------------

    !$omp parallel do private(p)
    !$acc parallel loop gang vector collapse(2) &
    !$acc& present(dqdt,Escale,deriv_x,deriv_y,deriv_z,lift_in)
    do ke = 1, Ne
      do p = 1, Np
        dqdt(p,ke) = -( &
             Escale(p,ke,1)*deriv_x(p,ke) &
           + Escale(p,ke,2)*deriv_y(p,ke) &
           + Escale(p,ke,3)*deriv_z(p,ke) &
           + lift_in(p,ke) )
      end do
    end do

    return
  end subroutine assemble_dqdt_openacc_split
end module mod_advect3d_eq
