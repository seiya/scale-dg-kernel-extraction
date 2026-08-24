!-------------------------------------------------------------------------------
!> module mesh
!!
!! @par Description
!!      A module to calculate the tendency 3D advection equation
!!
!! @author Yuta Kawai, Xuanzhengbo Ren, Team SCALE
!<
module mod_advect3d_eq
  use mod_common, only: RP, &
  Timer, Timer_start, Timer_stop, Timer_add, Timer_elapsed
  use mod_cuda_dg_kernels, only: &
    cuda_dg_kernels_available,  &
    cuda_cal_volume_flux, cuda_cal_volume_deriv, &
    cuda_cal_surface_lift, cuda_assemble_dqdt, cuda_cal_dqdt_split, &
    cuda_cal_dqdt_fused, cuda_cal_dqdt_fused_p255, &
    cuda_cal_dqdt_fused_tc, cuda_cal_dqdt_fused_p255_tc, &
    cuda_cal_dqdt_gemm, cuda_gemm_setup, cuda_gemm_finalize, &
    cuda_cal_elembnd_flux
  implicit none
  private

  public :: setup_advect3d_eq_setup
  public :: setup_advect3d_eq_finalize
  public :: advect3d_eq_cal_tend

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
  integer :: dqdt_kernel_typeid
  character(len=24) :: dqdt_kernel_name
  logical :: cublas_emulation_enabled = .false.

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
  subroutine setup_advect3d_eq_setup(NfpTot, Np, Ne, dqdt_kernel_type, cublas_emulation)
    implicit none
    integer, intent(in) :: NfpTot, Np, Ne
    character(len=*), intent(in) :: dqdt_kernel_type
    logical, intent(in), optional :: cublas_emulation
    !------------------------------------------------------------------------------
    cublas_emulation_enabled = .false.
    if (present(cublas_emulation)) cublas_emulation_enabled = cublas_emulation

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
      if (Np /= 512 .and. Np /= 256**3) then
        write(*,*) "CUDAFORTRAN_FUSED requires PolyOrder=7 or 255"
        error stop
      end if
      dqdt_kernel_typeid = DQDT_KERNEL_CUDAFORTRAN_FUSED
      dqdt_kernel_name = "CUDAFORTRAN_FUSED"
    case ("CUDAFORTRAN_FUSED_TC")
      if (.not. cuda_dg_kernels_available) then
        write(*,*) "CUDAFORTRAN_FUSED_TC requires a build with CUDA=1"
        error stop
      end if
      if (Np /= 512 .and. Np /= 256**3) then
        write(*,*) "CUDAFORTRAN_FUSED_TC requires PolyOrder=7 or 255"
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
      call cuda_gemm_setup(cublas_emulation_enabled)
    case default
      write(*,*) "Unsupported dqdt_kernel_type: ", trim(dqdt_kernel_type)
      error stop
    end select

    if (cublas_emulation_enabled .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      write(*,*) "CublasEmulation is ignored unless DqdtKernel_Type=CUDAFORTRAN_GEMM"
    end if

    if (Np == 256**3 .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      error stop "PolyOrder=255 currently requires CUDAFORTRAN_FUSED, CUDAFORTRAN_FUSED_TC, or CUDAFORTRAN_GEMM"
    end if

    if (dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC) then
      allocate(ebnd_flux(NfpTot,Ne))
      !$acc enter data create(ebnd_flux)
    end if
    if ((dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED .or. &
         dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_TC) .and. &
        Np == 256**3) then
      allocate(fused_flux_bnd(NfpTot,Ne))
      !$acc enter data create(fused_flux_bnd)
    end if

    if (dqdt_kernel_typeid /= DQDT_KERNEL_OPENACC_ASIS .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC) then
      allocate(volume_deriv_x(Np,Ne), volume_deriv_y(Np,Ne), volume_deriv_z(Np,Ne))
      !$acc enter data create(volume_deriv_x,volume_deriv_y,volume_deriv_z)
    end if
    if (dqdt_kernel_typeid /= DQDT_KERNEL_OPENACC_ASIS .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC) then
      allocate(surface_lift(Np,Ne))
      !$acc enter data create(surface_lift)
    end if
    if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_SPLIT .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_SPLIT .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      allocate(volume_flux_x(Np,Ne), volume_flux_y(Np,Ne), volume_flux_z(Np,Ne))
      !$acc enter data create(volume_flux_x,volume_flux_y,volume_flux_z)
    end if

    return
  end subroutine setup_advect3d_eq_setup
  !> Finalize
!OCL SERIAL
  subroutine setup_advect3d_eq_finalize()
    implicit none
    !------------------------------------------------------------------------------
    write(*,'(A30,A24)') "Dqdt kernel type:", trim(dqdt_kernel_name)
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      if (cublas_emulation_enabled) then
        write(*,'(A30,A24)') "Cublas FP emulation:", "on"
      else
        write(*,'(A30,A24)') "Cublas FP emulation:", "off"
      end if
    end if
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED .or. &
        dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_TC) then
      write(*,'(A30,1X,A23)') "Element boundary flux:", "included in fused kernel"
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
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
               dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_FUSED_TC) then
        write(*,'(A30,ES24.5)') "  CUDA device fused tendency:", Timer_elapsed(timer_volume_deriv)
      else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
        write(*,'(A30,ES24.5)') "  CUDA device GEMM tendency:", Timer_elapsed(timer_volume_deriv)
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
        !$acc exit data delete(volume_deriv_x,volume_deriv_y,volume_deriv_z)
        deallocate(volume_deriv_x, volume_deriv_y, volume_deriv_z)
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
    if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      call cuda_gemm_finalize()
    end if
    return
  end subroutine setup_advect3d_eq_finalize

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
    real(RP), intent(out) :: dqdt(Np,NeA)
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

    call Timer_start(timer_ebnd_flux)
    if (dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_FUSED_TC .and. &
        dqdt_kernel_typeid /= DQDT_KERNEL_CUDAFORTRAN_GEMM) then
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
    else if (dqdt_kernel_typeid == DQDT_KERNEL_CUDAFORTRAN_GEMM) then
      call cal_dqdt_cudafortran_gemm( dqdt, & ! (out)
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
    real(RP), intent(out) :: dqdt(Np,NeA)
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
    real(RP), intent(out) :: dqdt(Np,NeA)
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
    real(RP), intent(out) :: dqdt(Np,NeA)
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

    call Timer_add(timer_volume_flux,kernel_time(1))
    call Timer_add(timer_volume_deriv,kernel_time(2))
    call Timer_add(timer_surface_lift,kernel_time(3))
    call Timer_add(timer_dqdt_assemble,kernel_time(4))

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
    real(RP), intent(out) :: dqdt(Np,NeA)
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

    if (Nq == 8) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift_mat,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale)
      call cuda_cal_dqdt_fused( &
        dqdt, q, u, v, w, D1D, Lift_mat, VMapM, VMapP, &
        normal_fn, Fscale, Escale, &
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
      error stop "CUDAFORTRAN_FUSED requires Nq=8 or Nq=256"
    end if

    call Timer_add(timer_volume_deriv,kernel_time(1))
    call Timer_add(timer_surface_lift,kernel_time(2))

    return
  end subroutine cal_dqdt_cudafortran_fused

  !> Tensor-core fused tendency; original CUDAFORTRAN_FUSED kernels stay unchanged.
!OCL SERIAL
  subroutine cal_dqdt_cudafortran_fused_tc( dqdt, & ! (out)
    q, u, v, w,                               & ! (in)
    D1D, D1D_tr, Lift_mat, Lift1D,            & ! (in)
    VMapM, VMapP, normal_fn, Fscale, Escale,  & ! (in)
    Nq, Np, NfpTot, Ne, NeA                   ) ! (in)
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
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

    if (Nq == 8) then
      !$acc host_data use_device(dqdt,q,u,v,w,D1D,Lift_mat,VMapM,VMapP) &
      !$acc& use_device(normal_fn,Fscale,Escale)
      call cuda_cal_dqdt_fused_tc( &
        dqdt, q, u, v, w, D1D, Lift_mat, VMapM, VMapP, &
        normal_fn, Fscale, Escale, &
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
      error stop "CUDAFORTRAN_FUSED_TC requires Nq=8 or Nq=256"
    end if

    call Timer_add(timer_volume_deriv,kernel_time(1))
    call Timer_add(timer_surface_lift,kernel_time(2))

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
    !$acc& use_device(volume_deriv_x,volume_deriv_y,volume_deriv_z,surface_lift)
    call cuda_cal_dqdt_gemm( &
      dqdt, q, u, v, w, D1D, D1D_tr, Lift1D, VMapM, VMapP, &
      normal_fn, Fscale, Escale, ebnd_flux, &
      volume_flux_x, volume_flux_y, volume_flux_z, &
      volume_deriv_x, volume_deriv_y, volume_deriv_z, surface_lift, &
      Nq, Np, NfpTot, Ne, NeA, kernel_time )
    !$acc end host_data

    call Timer_add(timer_volume_deriv,kernel_time(1))
    call Timer_add(timer_surface_lift,kernel_time(2))

    return
  end subroutine cal_dqdt_cudafortran_gemm

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
    real(RP), intent(out) :: dqdt(Np,NeA)
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
