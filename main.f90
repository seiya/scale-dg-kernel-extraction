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
    Timer_start, Timer_stop, Timer_elapsed, &
    RK_nstage => RK3s3oSSP_nstage,          &
    rk_a => RK3s3oSSP_rk_a,                 &
    rk_b => RK3s3oSSP_rk_b
  use mod_mesh, only: &
    PolyOrder, Nq, Np, NfpTot, Ne, NeA,  &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, &
    normal_fn, Escale, Fscale, pos_en, update_halo
  use mod_advect3d_eq, only: &
    advect3d_eq_cal_tend
  implicit none

  !-----------------------------------------------------------------------------
  !
  !++ Public parameters & variables
  !

  integer :: nstep, output_interval
  integer :: istep, stage
  real(RP) :: dt

  real(RP), allocatable :: q(:,:)
  real(RP), allocatable :: q0(:,:)
  real(RP), allocatable :: dqdt(:,:)
  real(RP), allocatable :: u(:,:)
  real(RP), allocatable :: v(:,:)
  real(RP), allocatable :: w(:,:)

  integer :: kelem, pnode
  real(RP) :: q_min, q_max

  type(Timer) :: timer_main
  type(Timer) :: timer_cal_tend

  !- Main program ----------------------------------------------------------

  call init()

  !$acc data copyin(q,u,v,w,D1D,D1D_tr,Lift_mat,VMapM,VMapP) &
  !$acc& copyin(normal_fn,Escale,Fscale) create(q0,dqdt)

  call update_halo(u)
  call update_halo(v)
  call update_halo(w)

  call Timer_start(timer_main)

  !- Loop for time integration
  do istep = 1, nstep

    !$omp parallel do private(pnode)
    !$acc parallel loop gang vector collapse(2) present(q0,q)
    do kelem=1, Ne
      do pnode=1, Np
        q0(pnode,kelem) = q(pnode,kelem)
      end do
    end do

    do stage = 1, RK_nstage
      call update_halo(q)

      call Timer_start(timer_cal_tend)
      call advect3d_eq_cal_tend( dqdt,       & ! (out)
        q, u, v, w,                              & ! (in)
        D1D, D1D_tr, Lift_mat,                   & ! (in)
        VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
        Nq, Np, NfpTot, Ne, NeA )
      call Timer_stop(timer_cal_tend)

      !$omp parallel do private(pnode)
      !$acc parallel loop gang vector collapse(2) present(q,q0,dqdt)
      do kelem=1, Ne
        do pnode=1, Np
          q(pnode,kelem) = rk_a(stage) * q0(pnode,kelem) &
                         + rk_b(stage) * ( q(pnode,kelem) + dt * dqdt(pnode,kelem) )
        end do
      end do
    end do

    if (mod(istep,output_interval) == 0) then
      q_min = huge(q_min)
      q_max = -huge(q_max)
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
    character(len=20) :: DqdtKernel_Type = "OPENACC_SPLIT"

    namelist /PARAM_ADVECT3D/ &
      NeX, NeY, NeZ, PolyOrder,   &
      dt, nstep, output_interval, &
      vel_x, vel_y, vel_z,        &
      DGOptrKernel_OptType,        &
      DqdtKernel_Type

    integer :: fid
    integer :: ke, p
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
    call setup_advect3d_eq_setup(NfpTot, Np, Ne, DqdtKernel_Type)

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
    write(*,'(A30,ES24.5)') "Cal_tend:", Timer_elapsed(timer_cal_tend)

    call setup_advect3d_eq_finalize()
    call mesh_finalize()
    return
  end subroutine final
end program main
