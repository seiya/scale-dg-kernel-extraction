!> Main program for SCALE-DG kernel extraction
!!
!! @author Yuta Kawai, Team SCALE
!!
program main
  !-----------------------------------------------------------------------------
  !
  !++ used modules
  !  
  use mod_common, only: &
    RP,                                     &
    Timer,                                  &
    Timer_start, Timer_stop, Timer_elapsed, Timer_reset, &
    ACC_QUEUE,                              &
    RK_nstage => RK3s3oSSP_nstage,          &
    rk_a => RK3s3oSSP_rk_a,                 &
    rk_b => RK3s3oSSP_rk_b
  use mod_mesh, only: &
    PolyOrder, Nq, Np, NfpTot, Ne, NeA,  &
    D1D, D1D_tr, Lift_mat, Lift1D, VMapM, VMapP, &
    normal_fn, Escale, Fscale, pos_en, update_halo
  use mod_advect3d_eq, only: &
    advect3d_eq_cal_tend, advect3d_eq_graph_supported, &
    advect3d_eq_set_time_reporting, advect3d_eq_reset_timers, &
    cuda_dg_set_event_timing, cuda_dg_set_side_stream, &
    cuda_dg_graph_capture_begin, &
    cuda_dg_graph_capture_end, cuda_dg_graph_launch, cuda_dg_graph_is_ready
  implicit none

  !-----------------------------------------------------------------------------
  !
  !++ Public parameters & variables
  !

  integer :: nstep, output_interval
  integer :: istep, stage

  !> Number of leading steps that are run but not timed.  The first step of a
  !! run pays for lazy device-side initialization (module load, first-touch of
  !! the device allocations, the cuBLAS/CUTLASS handles and workspaces, and the
  !! clocks still ramping up), which is amortized away at large nstep but
  !! dominates a short run.  Set from the namelist; see the note above the
  !! time loop for the value actually used.
  integer :: WarmupStep = 1
  integer :: nwarmup
  real(RP) :: dt

  !> Replay one captured Runge-Kutta step instead of launching its kernels
  !! one by one.  Set from the namelist; see the note above the time loop.
  logical :: UseCudaGraph = .false.

  !> Bracket the tendency kernels with CUDA events to report their device
  !! time.  Turning it off drops "CUDA device ..." from the report and leaves
  !! only the wall time; a CUDA graph replay cannot report it either, so
  !! UseCudaGraph turns it off as well.
  logical :: MeasureKernelTime = .true.

  real(RP), allocatable :: q(:,:)
  real(RP), allocatable :: q0(:,:)
  real(RP), allocatable :: dqdt(:,:)
  real(RP), allocatable :: u(:,:)
  real(RP), allocatable :: v(:,:)
  real(RP), allocatable :: w(:,:)

  integer :: kelem, pnode
  real(RP) :: q_min, q_max
  real(RP) :: a_rk, b_rk

  type(Timer) :: timer_main
  type(Timer) :: timer_cal_tend

  !- Main program ----------------------------------------------------------

  call init()

  !$acc data copyin(q,u,v,w,D1D,D1D_tr,Lift_mat,Lift1D,VMapM,VMapP) &
  !$acc& copyin(normal_fn,Escale,Fscale) create(q0,dqdt)

  call update_halo(u)
  call update_halo(v)
  call update_halo(w)

  !- A time step is nine kernel launches on one stream, and section 8.2 of
  !  reports/overall_summary_report.md measured 50 us/step of GPU idle left in
  !  their turnaround after the host synchronization was removed.  Capturing
  !  the step once and replaying the graph removes the per-launch host work
  !  without changing which kernels run, in which order, or on what data.
  !
  !  The capture has to happen on a step whose result the host does not read,
  !  because a capture executes nothing: step 1 is therefore always launched
  !  directly, step 2 is captured and replayed, and every later step is a
  !  replay.  A replay does not run the Fortran wrappers, so the per-kernel
  !  CUDA event timing is switched off for the whole run in this mode; only
  !  the wall time is reported.
  !  The second stream that overlaps the element boundary flux with the volume
  !  GEMMs is also switched off in graph mode: on a replay the forked structure
  !  measures as a small loss instead of the gain it is when the kernels are
  !  launched one by one.
  if (UseCudaGraph) then
    MeasureKernelTime = .false.
    call advect3d_eq_set_time_reporting(.false.)
    call cuda_dg_set_side_stream(.false.)
  end if
  if (.not. MeasureKernelTime) then
    call cuda_dg_set_event_timing(.false.)
  end if

  !- The reported time covers the last nstep-nwarmup steps only.  The steps
  !  that are skipped are still run, so the final field is the one of a full
  !  nstep run and stays comparable with a reference; only the clock starts
  !  later.  In graph mode the first two steps are structurally different from
  !  the rest -- step 1 is launched directly and step 2 is the capture -- so at
  !  least two steps are skipped there.  A run that is too short to spare the
  !  steps measures all of them.
  nwarmup = min(max(WarmupStep,0),max(nstep-1,0))
  if (UseCudaGraph) then
    nwarmup = min(max(nwarmup,2),max(nstep-1,0))
  end if

  !- Loop for time integration
  do istep = 1, nstep

    if (istep == nwarmup + 1) then
      !- The warm-up steps are queued asynchronously, so they have to be over
      !  before the clock starts.
      !$acc wait(ACC_QUEUE)
      call Timer_reset(timer_cal_tend)
      call advect3d_eq_reset_timers()
      call Timer_start(timer_main)
    end if

    if (UseCudaGraph .and. istep >= 2) then
      if (.not. cuda_dg_graph_is_ready()) then
        call cuda_dg_graph_capture_begin()
        call advance_step()
        call cuda_dg_graph_capture_end()
      end if
      call cuda_dg_graph_launch()
    else
      call advance_step()
    end if

    if (mod(istep,output_interval) == 0) then
      q_min = huge(q_min)
      q_max = -huge(q_max)
      !- The reduction result is read on the host, so the queued device work
      !  has to be complete first.
      !$acc wait(ACC_QUEUE)
      !$acc parallel loop gang vector collapse(2) present(q) &
      !$acc& reduction(min:q_min) reduction(max:q_max)
      do kelem=1, Ne
        do pnode=1, Np
          q_min = min(q_min,q(pnode,kelem))
          q_max = max(q_max,q(pnode,kelem))
        end do
      end do
      write(*,'(I8,2ES24.15)') istep, q_min, q_max
    end if
  end do

  call dump_q_if_requested()

  !$acc wait(ACC_QUEUE)

  !$acc end data

  call final()
contains
  !> Initialize modules with DG mesh and DG operator kernel
  subroutine init()
    use, intrinsic :: iso_fortran_env, only: error_unit
    use mod_common, only: PI
    use mod_mesh, only: mesh_setup
    use mod_advect3d_eq, only: setup_advect3d_eq_setup
    use mod_dg_optr_kernel, only: dg_optr_kernel_setup    
    implicit none

    character(len=256) :: conf_file
    integer :: NeX = 4
    integer :: NeY = 4
    integer :: NeZ = 4
    integer :: PolyOrder = 3
    real(RP) :: vel_x = 1.0_RP
    real(RP) :: vel_y = 1.0_RP
    real(RP) :: vel_z = 1.0_RP
    character(len=8) :: DGOptrKernel_OptType = "OPT1" ! GENERAL or OPT1
    character(len=24) :: DqdtKernel_Type = "OPENACC_SPLIT"
    logical :: CublasEmulation = .false.
    integer :: OzakiModuliCount = 14
    integer :: OzakiSliceCount = 8
    character(len=16) :: CutlassMmaShape = "8x8x4" ! 8x8x4, 16x8x4, 16x8x8, 16x8x16

    namelist /PARAM_ADVECT3D/ &
      NeX, NeY, NeZ, PolyOrder,   &
      dt, nstep, output_interval, &
      vel_x, vel_y, vel_z,        &
      DGOptrKernel_OptType,        &
      DqdtKernel_Type,             &
      CublasEmulation,             &
      OzakiModuliCount,            &
      OzakiSliceCount,             &
      CutlassMmaShape,             &
      UseCudaGraph,                &
      MeasureKernelTime,           &
      WarmupStep

    integer :: fid
    integer :: ke, p
    character(len=8) :: vary_coeff
    !------------------------------------------------------------

    dt    = 1.0e-3_RP
    nstep = 100
    output_interval = 10

    if (command_argument_count() < 1) then
      write(error_unit,'(A)') 'Error: configuration file argument is required.'
      write(error_unit,'(A)') 'Usage: scale-dg_extraction input.conf'
      flush(error_unit)
      error stop 1
    end if

    call get_command_argument(1,conf_file)

    fid = 10
    open(fid,file=trim(conf_file),status='old',action='read')
    read(fid,nml=PARAM_ADVECT3D)
    close(fid)

    !- Initialize a mesh module
    call mesh_setup( NeX, NeY, NeZ, PolyOrder, &
      1.0_RP, 1.0_RP, 1.0_RP )

    allocate( q(Np,NeA), q0(Np,NeA), dqdt(Np,NeA) )
    allocate( u(Np,NeA), v(Np,NeA), w(Np,NeA) )

    !- Initialize a DG operator module
    call dg_optr_kernel_setup( DGOptrKernel_OptType )

    !- Initialize a advection equation module
    call setup_advect3d_eq_setup(NfpTot, Np, Ne, DqdtKernel_Type, CublasEmulation, &
      CutlassMmaShape, OzakiModuliCount, OzakiSliceCount)

    if (UseCudaGraph .and. .not. advect3d_eq_graph_supported()) then
      write(*,*) "UseCudaGraph is ignored: ", trim(DqdtKernel_Type), &
        " launches kernels outside the captured stream"
      UseCudaGraph = .false.
    end if

    !- Set initial condition

    !$omp parallel do private(p)
    do ke = 1, Ne
    do p = 1, Np
      q(p,ke) = sin( 2.0_RP*PI*pos_en(p,ke,1) ) &
              * sin( 2.0_RP*PI*pos_en(p,ke,2) ) &
              * sin( 2.0_RP*PI*pos_en(p,ke,3) )

      u(p,ke) = vel_x
      v(p,ke) = vel_y
      w(p,ke) = vel_z
    end do
    end do

    vary_coeff = '0'
    call get_environment_variable('SCALE_DG_VARYING_COEFF', vary_coeff)
    if (trim(vary_coeff) == '1') then
      do ke = 1, Ne
        do p = 1, Np
          u(p,ke) = vel_x + 0.15_RP * sin(2.0_RP*PI*pos_en(p,ke,1))
          v(p,ke) = vel_y + 0.11_RP * cos(2.0_RP*PI*pos_en(p,ke,2))
          w(p,ke) = vel_z + 0.07_RP * sin(2.0_RP*PI*pos_en(p,ke,3))
          Escale(p,ke,1) = Escale(p,ke,1) * (1.0_RP + 0.02_RP * sin(2.0_RP*PI*pos_en(p,ke,1)))
          Escale(p,ke,2) = Escale(p,ke,2) * (1.0_RP + 0.03_RP * cos(2.0_RP*PI*pos_en(p,ke,2)))
          Escale(p,ke,3) = Escale(p,ke,3) * (1.0_RP + 0.04_RP * sin(2.0_RP*PI*pos_en(p,ke,3)))
        end do
        do p = 1, NfpTot
          Fscale(p,ke) = Fscale(p,ke) * (1.0_RP + 0.01_RP * sin(dble(p)))
          normal_fn(p,ke,1) = normal_fn(p,ke,1) + 0.01_RP * sin(dble(p+ke))
          normal_fn(p,ke,2) = normal_fn(p,ke,2) + 0.01_RP * cos(dble(p+ke))
          normal_fn(p,ke,3) = normal_fn(p,ke,3) + 0.01_RP * sin(dble(2*p+ke))
        end do
      end do
    end if

    return
  end subroutine init

!OCL SERIAL
  subroutine final()
    use mod_advect3d_eq, only: setup_advect3d_eq_finalize
    use mod_mesh, only: mesh_finalize
    implicit none
    !-----------------------------------------------------------------------------

    call Timer_stop(timer_main)
    write(*,'(A)') "= Report of execution time [sec]"
    write(*,'(A30,ES24.5)') "Main:", Timer_elapsed(timer_main)
    write(*,'(A30,I24)')    "Measured steps:", nstep - nwarmup
    write(*,'(A30,I24)')    "Skipped warm-up steps:", nwarmup
    write(*,'(A30,ES24.5)') "Main per step:", &
      Timer_elapsed(timer_main) / real(max(nstep-nwarmup,1),RP)
    if (UseCudaGraph) then
      write(*,'(A30,A24)') "CUDA graph replay:", "on"
      write(*,'(A30,A24)') "Cal_tend:", "not measured (graph)"
    else
      write(*,'(A30,ES24.5)') "Cal_tend:", Timer_elapsed(timer_cal_tend)
    end if

    call setup_advect3d_eq_finalize()
    call mesh_finalize()
    return
  end subroutine final

  !> One SSP-RK3 step: three stages of halo update, tendency, and update.
  !! Kept as its own procedure so that the whole step can be handed to the
  !! CUDA graph capture unchanged.
  subroutine advance_step()
    implicit none
    !------------------------------------------------------------

    do stage = 1, RK_nstage
      call update_halo(q)

      call Timer_start(timer_cal_tend)
      call advect3d_eq_cal_tend( dqdt,       & ! (out)
        q, u, v, w,                              & ! (in)
        D1D, D1D_tr, Lift_mat, Lift1D,           & ! (in)
        VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
        Nq, Np, NfpTot, Ne, NeA )
      call Timer_stop(timer_cal_tend)

      if (istep == 1 .and. stage == 1) then
        call dump_dqdt_if_requested(dqdt, Np, Ne)
      end if

      !- rk_a and rk_b are indexed by the runtime stage, so left in the loop
      !  body they become one global load each per thread: ncu measured 5
      !  global load instructions per warp where the three fields need 3.
      !  Read once on the host and they are scalars in the kernel.
      a_rk = rk_a(stage)
      b_rk = rk_b(stage)

      call rk_update_stage( q, q0, dqdt, Np*Ne, a_rk, b_rk, dt, stage == 1 )
    end do

    return
  end subroutine advance_step

  !> Write the owned q field after the last step, for regressions that have
  !! to compare a complete field rather than its extrema.  The dqdt dump is
  !! taken at the first step, which a CUDA graph replay never runs.
  !> One SSP-RK stage update over the owned points.
  !!
  !! The owned part of q, q0 and dqdt is the contiguous first Np*Ne elements,
  !! so the update is a flat one-dimensional loop.  Written as a collapse(2)
  !! loop over (Np, Ne) instead, the compiler has to recover both indices from
  !! the flattened thread number by dividing by the runtime value Np.
  !!
  !! At stage 1 q0 is by definition the current q, so the SSP-RK save is fused
  !! into the update.  This removes a separate q0 <- q pass over the field
  !! without changing either the arithmetic or the lifetime of q0.
  subroutine rk_update_stage( qq, qq0, dq, npoint, a, b, dtl, save_q0 )
    implicit none
    integer, intent(in) :: npoint
    real(RP), intent(inout) :: qq(npoint), qq0(npoint)
    real(RP), intent(in) :: dq(npoint)
    real(RP), intent(in) :: a, b, dtl
    logical, intent(in) :: save_q0

    integer :: i
    !-----------------------------------------------------------------------------

    if (save_q0) then
      !$omp parallel do
      !$acc parallel loop gang vector present(qq,qq0,dq) async(ACC_QUEUE)
      do i = 1, npoint
        qq0(i) = qq(i)
        qq(i) = a * qq0(i) + b * ( qq(i) + dtl * dq(i) )
      end do
    else
      !$omp parallel do
      !$acc parallel loop gang vector present(qq,qq0,dq) async(ACC_QUEUE)
      do i = 1, npoint
        qq(i) = a * qq0(i) + b * ( qq(i) + dtl * dq(i) )
      end do
    end if

    return
  end subroutine rk_update_stage

  subroutine dump_q_if_requested()
    implicit none
    character(len=256) :: dump_file
    integer :: ke, p, fid
    !------------------------------------------------------------

    dump_file = ''
    call get_environment_variable('SCALE_DG_DUMP_Q', dump_file)
    if (len_trim(dump_file) == 0) return
    !$acc wait(ACC_QUEUE)
    !$acc update host(q)
    fid = 22
    open(fid, file=trim(dump_file), status='replace', action='write')
    do ke = 1, Ne
      do p = 1, Np
        write(fid,'(ES24.16)') q(p,ke)
      end do
    end do
    close(fid)

    return
  end subroutine dump_q_if_requested

  subroutine dump_dqdt_if_requested(dqdt, Np, Ne)
    implicit none
    integer, intent(in) :: Np, Ne
    real(RP), intent(in) :: dqdt(:,:)
    character(len=256) :: dump_file
    integer :: ke, p, fid
    dump_file = ''
    call get_environment_variable('SCALE_DG_DUMP_DQDT', dump_file)
    if (len_trim(dump_file) == 0) return
    !$acc wait(ACC_QUEUE)
    !$acc update host(dqdt)
    fid = 21
    open(fid, file=trim(dump_file), status='replace', action='write')
    do ke = 1, Ne
      do p = 1, Np
        write(fid,'(ES24.16)') dqdt(p,ke)
      end do
    end do
    close(fid)
  end subroutine dump_dqdt_if_requested
end program main
