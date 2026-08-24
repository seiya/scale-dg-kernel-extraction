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
  Timer, Timer_start, Timer_stop, Timer_elapsed
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
  integer :: dqdt_kernel_typeid
  character(len=16) :: dqdt_kernel_name

  real(RP), allocatable :: ebnd_flux(:,:)
  real(RP), allocatable :: volume_flux_x(:,:)
  real(RP), allocatable :: volume_flux_y(:,:)
  real(RP), allocatable :: volume_flux_z(:,:)
  real(RP), allocatable :: volume_deriv_x(:,:)
  real(RP), allocatable :: volume_deriv_y(:,:)
  real(RP), allocatable :: volume_deriv_z(:,:)
  real(RP), allocatable :: surface_lift(:,:)
contains
  !> Setup
!OCL SERIAL
  subroutine setup_advect3d_eq_setup(NfpTot, Np, Ne, dqdt_kernel_type)
    implicit none
    integer, intent(in) :: NfpTot, Np, Ne
    character(len=*), intent(in) :: dqdt_kernel_type
    !------------------------------------------------------------------------------

    select case (trim(dqdt_kernel_type))
    case ("OPENACC_ASIS")
      dqdt_kernel_typeid = DQDT_KERNEL_OPENACC_ASIS
      dqdt_kernel_name = "OPENACC_ASIS"
    case ("OPENACC_SPLIT")
      dqdt_kernel_typeid = DQDT_KERNEL_OPENACC_SPLIT
      dqdt_kernel_name = "OPENACC_SPLIT"
    case default
      write(*,*) "Unsupported dqdt_kernel_type: ", trim(dqdt_kernel_type)
      error stop
    end select

    allocate(ebnd_flux(NfpTot,Ne))
    !$acc enter data create(ebnd_flux)

    if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_SPLIT) then
      allocate(volume_flux_x(Np,Ne), volume_flux_y(Np,Ne), volume_flux_z(Np,Ne))
      allocate(volume_deriv_x(Np,Ne), volume_deriv_y(Np,Ne), volume_deriv_z(Np,Ne))
      allocate(surface_lift(Np,Ne))
      !$acc enter data create(volume_flux_x,volume_flux_y,volume_flux_z) &
      !$acc& create(volume_deriv_x,volume_deriv_y,volume_deriv_z,surface_lift)
    end if

    return
  end subroutine setup_advect3d_eq_setup
  !> Finalize
!OCL SERIAL
  subroutine setup_advect3d_eq_finalize()
    implicit none
    !------------------------------------------------------------------------------
    write(*,'(A30,A24)') "Dqdt kernel type:", trim(dqdt_kernel_name)
    write(*,'(A30,ES24.5)') "Element boundary flux:", Timer_elapsed(timer_ebnd_flux)
    write(*,'(A30,ES24.5)') "Volume derivate + surface lift:", Timer_elapsed(timer_dqdt)

    if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_SPLIT) then
      write(*,'(A30,ES24.5)') "  Volume flux:", Timer_elapsed(timer_volume_flux)
      write(*,'(A30,ES24.5)') "  Tensor-product derivative:", Timer_elapsed(timer_volume_deriv)
      write(*,'(A30,ES24.5)') "  Surface lift:", Timer_elapsed(timer_surface_lift)
      write(*,'(A30,ES24.5)') "  Dqdt assembly:", Timer_elapsed(timer_dqdt_assemble)
      !$acc exit data delete(volume_flux_x,volume_flux_y,volume_flux_z) &
      !$acc& delete(volume_deriv_x,volume_deriv_y,volume_deriv_z,surface_lift)
      deallocate(volume_flux_x, volume_flux_y, volume_flux_z)
      deallocate(volume_deriv_x, volume_deriv_y, volume_deriv_z)
      deallocate(surface_lift)
    end if

    !$acc exit data delete(ebnd_flux)
    deallocate(ebnd_flux)
    return
  end subroutine setup_advect3d_eq_finalize

  !> Calculate the tendency of 3D advection equation
!OCL SERIAL
  subroutine advect3d_eq_cal_tend( dqdt, & ! (out)
    q, u, v, w,                              & ! (in)
    D1D, D1D_tr, Lift_mat,                   & ! (in)
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
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(in) :: Fscale(NfpTot,Ne)
    !------------------------------------------------------------

    call Timer_start(timer_ebnd_flux)
    call cal_elembnd_flux( ebnd_flux,   & ! (out)
       q, u, v, w,                      & ! (in)
       VMapM, VMapP, normal_fn, Fscale, & ! (in)
       Np, NfpTot, Ne, NeA )
    call Timer_stop(timer_ebnd_flux)

    call Timer_start(timer_dqdt)
    if (dqdt_kernel_typeid == DQDT_KERNEL_OPENACC_ASIS) then
      call cal_dqdt_openacc_asis( dqdt, & ! (out)
         q, u, v, w,  ebnd_flux,        & ! (in)
         D1D, D1D_tr, Lift_mat,         & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
    else
      call cal_dqdt_openacc_split( dqdt, & ! (out)
         q, u, v, w,  ebnd_flux,         & ! (in)
         D1D, D1D_tr, Lift_mat,          & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
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
