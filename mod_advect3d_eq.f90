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
  real(RP), allocatable :: ebnd_flux(:,:)
contains
  !> Setup
!OCL SERIAL
  subroutine setup_advect3d_eq_setup(NfpTot, Ne)
    implicit none
    integer, intent(in) :: NfpTot, Ne
    !------------------------------------------------------------------------------
    allocate(ebnd_flux(NfpTot,Ne))
    !$acc enter data create(ebnd_flux)
    return
  end subroutine setup_advect3d_eq_setup
  !> Finalize
!OCL SERIAL
  subroutine setup_advect3d_eq_finalize()
    implicit none
    !------------------------------------------------------------------------------
    write(*,'(A30,ES24.5)') "Element boundary flux:", Timer_elapsed(timer_ebnd_flux)
    write(*,'(A30,ES24.5)') "Volume derivate + surface lift:", Timer_elapsed(timer_dqdt)
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
    call cal_dqdt( dqdt,               & ! (out)
       q, u, v, w,  ebnd_flux,         & ! (in)
       D1D, D1D_tr, Lift_mat,          & ! (in)
       Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
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
  subroutine cal_dqdt( dqdt,       & ! (out)
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
  end subroutine cal_dqdt
end module mod_advect3d_eq
