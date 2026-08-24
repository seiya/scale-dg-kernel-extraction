!-------------------------------------------------------------------------------
!> module mesh
!!
!! @par Description
!!      A module to provide element-wise operations with DGM
!!      assuming that the hexahedral element is used and sum-factorization can be applied.
!!      
!!
!! @author Yuta Kawai, Xuanzhengbo Ren, Team SCALE
!<
module mod_dg_optr_kernel
  !-----------------------------------------------------------------------------
  !
  !++ used modules
  !  
  use mod_common, only: RP
  implicit none
  private

  !-----------------------------------------------------------------------------
  !
  !++ Public type & procedure
  !  
  public :: dg_optr_kernel_setup
  public :: tensorprod_divlike_dirXYZ
  public :: tensorprod_Lift_hexahedral
  public :: tensorprod_divlike_dirXYZ_all
  public :: tensorprod_Lift_hexahedral_all

  !-----------------------------------------------------------------------------
  !
  !++ Private procedure
  !
  !-----------------------------------------------------------------------------
  !
  !++ Private parameters & variables
  !
  integer :: eval_typeid !< The evaluation type ID for the tensor-product operations
  !$acc declare create(eval_typeid)

  !> ID of the general tensor-product evaluation type for arbitrary p.
  integer, parameter :: DGOPTR_EVALTYPEID_TENSORPROD3D_GENERAL = 1
  !> ID of the optimized tensor-product evaluation type.
  !! It uses codes explicitly generated for each p (=1~15).
  integer, parameter :: DGOPTR_EVALTYPEID_TENSORPROD3D_OPT1 = 2

contains
  subroutine dg_optr_kernel_setup( dgoptr_evaltype )
    implicit none
    character(len=*), intent(in) :: dgoptr_evaltype
    !---------------------------------------------
    select case( trim(dgoptr_evaltype) )
    case ("GENERAL")
      eval_typeid = DGOPTR_EVALTYPEID_TENSORPROD3D_GENERAL
    case ("OPT1")
      eval_typeid = DGOPTR_EVALTYPEID_TENSORPROD3D_OPT1
    case default
      write(*,*) "Unsupported dgoptr_evaltype: ", dgoptr_evaltype
      error stop
    end select
    !$acc update device(eval_typeid)
    return
  end subroutine dg_optr_kernel_setup

  !> Tensor-product differentiation
!OCL SERIAL
  subroutine tensorprod_divlike_dirXYZ( &
    vec_out_x, vec_out_y, vec_out_z, &
    Mat, Mat_tr,                     &
    vec_in_x, vec_in_y, vec_in_z,    &
    Nq )
    !$acc routine vector
    use mod_dg_optr_kernel_opt1
    implicit none
    integer, intent(in) :: Nq
    real(RP), intent(in) :: Mat(Nq,Nq)
    real(RP), intent(in) :: Mat_tr(Nq,Nq)
    real(RP), intent(in) :: vec_in_x(Nq,Nq**2)
    real(RP), intent(in) :: vec_in_y(Nq,Nq,Nq)
    real(RP), intent(in) :: vec_in_z(Nq,Nq,Nq)
    real(RP), intent(out) :: vec_out_x(Nq,Nq**2)
    real(RP), intent(out) :: vec_out_y(Nq,Nq**2)
    real(RP), intent(out) :: vec_out_z(Nq,Nq**2)
    !--------------------------------------------------

    if ( eval_typeid == DGOPTR_EVALTYPEID_TENSORPROD3D_GENERAL ) then
      call tensorprod_divlike_dirXYZ_general( &
            vec_out_x, vec_out_y, vec_out_z, &
            Mat, Mat_tr,                     &
            vec_in_x, vec_in_y, vec_in_z,    &
            Nq )
    else if ( eval_typeid == DGOPTR_EVALTYPEID_TENSORPROD3D_OPT1 ) then
      select case(Nq-1)
      case(1)
        call element_operation_kernel_matvec_divlike_dirXYZ_P1( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(2)
        call element_operation_kernel_matvec_divlike_dirXYZ_P2( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(3)
        call element_operation_kernel_matvec_divlike_dirXYZ_P3( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(4)
        call element_operation_kernel_matvec_divlike_dirXYZ_P4( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(5)
        call element_operation_kernel_matvec_divlike_dirXYZ_P5( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(6)
        call element_operation_kernel_matvec_divlike_dirXYZ_P6( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(7)
        call element_operation_kernel_matvec_divlike_dirXYZ_P7( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(8)
        call element_operation_kernel_matvec_divlike_dirXYZ_P8( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(9)
        call element_operation_kernel_matvec_divlike_dirXYZ_P9( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(10)
        call element_operation_kernel_matvec_divlike_dirXYZ_P10( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(11)
        call element_operation_kernel_matvec_divlike_dirXYZ_P11( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(12)
        call element_operation_kernel_matvec_divlike_dirXYZ_P12( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(13)
        call element_operation_kernel_matvec_divlike_dirXYZ_P13( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(14)
        call element_operation_kernel_matvec_divlike_dirXYZ_P14( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      case(15)
        call element_operation_kernel_matvec_divlike_dirXYZ_P15( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z )
      end select
    end if
    return
  end subroutine tensorprod_divlike_dirXYZ

!OCL SERIAL
  subroutine tensorprod_Lift_hexahedral( &
    vec_out,         &
    Lift, vec_in, Nq )
    !$acc routine vector
    use mod_dg_optr_kernel_opt1
    implicit none
    integer, intent(in) :: Nq
    real(RP), intent(out) :: vec_out(Nq,Nq,Nq)
    real(RP), intent(in) :: Lift(Nq,Nq,Nq,6)
    real(RP), intent(in) :: vec_in(Nq,Nq,6)
    !----------------------------------------------------------

    if ( eval_typeid == DGOPTR_EVALTYPEID_TENSORPROD3D_GENERAL ) then
      call tensorprod_Lift_hexahedral_general( &
        vec_out,         &
        Lift, vec_in, Nq )
    else if ( eval_typeid == DGOPTR_EVALTYPEID_TENSORPROD3D_OPT1 ) then
      select case(Nq-1)
      case(1)
        call element_operation_kernel_matvec_Lift_hexahedral_P1( Lift, vec_in, vec_out )
      case(2)
        call element_operation_kernel_matvec_Lift_hexahedral_P2( Lift, vec_in, vec_out )
      case(3)
        call element_operation_kernel_matvec_Lift_hexahedral_P3( Lift, vec_in, vec_out )
      case(4)
        call element_operation_kernel_matvec_Lift_hexahedral_P4( Lift, vec_in, vec_out )
      case(5)
        call element_operation_kernel_matvec_Lift_hexahedral_P5( Lift, vec_in, vec_out )
      case(6)
        call element_operation_kernel_matvec_Lift_hexahedral_P6( Lift, vec_in, vec_out )
      case(7)
        call element_operation_kernel_matvec_Lift_hexahedral_P7( Lift, vec_in, vec_out )
      case(8)
        call element_operation_kernel_matvec_Lift_hexahedral_P8( Lift, vec_in, vec_out )
      case(9)
        call element_operation_kernel_matvec_Lift_hexahedral_P9( Lift, vec_in, vec_out )
      case(10)
        call element_operation_kernel_matvec_Lift_hexahedral_P10( Lift, vec_in, vec_out )
      case(11)
        call element_operation_kernel_matvec_Lift_hexahedral_P11( Lift, vec_in, vec_out )
      case(12)
        call element_operation_kernel_matvec_Lift_hexahedral_P12( Lift, vec_in, vec_out )
      case(13)
        call element_operation_kernel_matvec_Lift_hexahedral_P13( Lift, vec_in, vec_out )
      case(14)
        call element_operation_kernel_matvec_Lift_hexahedral_P14( Lift, vec_in, vec_out )
      case(15)
        call element_operation_kernel_matvec_Lift_hexahedral_P15( Lift, vec_in, vec_out )
      end select
    end if

    return
  end subroutine tensorprod_Lift_hexahedral

  !> Tensor-product differentiation over all elements
!OCL SERIAL
  subroutine tensorprod_divlike_dirXYZ_all( &
    vec_out_x, vec_out_y, vec_out_z, &
    Mat, Mat_tr,                     &
    vec_in_x, vec_in_y, vec_in_z,    &
    Nq, Ne )
    use mod_dg_optr_kernel_opt1
    implicit none
    integer, intent(in) :: Nq, Ne
    real(RP), intent(in) :: Mat(Nq,Nq), Mat_tr(Nq,Nq)
    real(RP), intent(in) :: vec_in_x(Nq**3,Ne)
    real(RP), intent(in) :: vec_in_y(Nq**3,Ne)
    real(RP), intent(in) :: vec_in_z(Nq**3,Ne)
    real(RP), intent(out) :: vec_out_x(Nq**3,Ne)
    real(RP), intent(out) :: vec_out_y(Nq**3,Ne)
    real(RP), intent(out) :: vec_out_z(Nq**3,Ne)
    !--------------------------------------------------

    if (eval_typeid == DGOPTR_EVALTYPEID_TENSORPROD3D_GENERAL) then
      call tensorprod_divlike_dirXYZ_all_general( &
        vec_out_x, vec_out_y, vec_out_z, &
        Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, Nq, Ne )
    else
      select case (Nq-1)
      case (1)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P1( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (2)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P2( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (3)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P3( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (4)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P4( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (5)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P5( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (6)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P6( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (7)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P7( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (8)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P8( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (9)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P9( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (10)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P10( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (11)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P11( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (12)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P12( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (13)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P13( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (14)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P14( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      case (15)
        call element_operation_kernel_matvec_divlike_dirXYZ_all_P15( &
          Mat, Mat_tr, vec_in_x, vec_in_y, vec_in_z, vec_out_x, vec_out_y, vec_out_z, Ne )
      end select
    end if

    return
  end subroutine tensorprod_divlike_dirXYZ_all

  !> Tensor-product surface lifting over all elements
!OCL SERIAL
  subroutine tensorprod_Lift_hexahedral_all( &
    vec_out, Lift, vec_in, Nq, Ne )
    use mod_dg_optr_kernel_opt1
    implicit none
    integer, intent(in) :: Nq, Ne
    real(RP), intent(out) :: vec_out(Nq**3,Ne)
    real(RP), intent(in) :: Lift(Nq,Nq,Nq,6)
    real(RP), intent(in) :: vec_in(6*Nq**2,Ne)
    !--------------------------------------------------

    if (eval_typeid == DGOPTR_EVALTYPEID_TENSORPROD3D_GENERAL) then
      call tensorprod_Lift_hexahedral_all_general( &
        vec_out, Lift, vec_in, Nq, Ne )
    else
      select case (Nq-1)
      case (1)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P1( &
          Lift, vec_in, vec_out, Ne )
      case (2)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P2( &
          Lift, vec_in, vec_out, Ne )
      case (3)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P3( &
          Lift, vec_in, vec_out, Ne )
      case (4)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P4( &
          Lift, vec_in, vec_out, Ne )
      case (5)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P5( &
          Lift, vec_in, vec_out, Ne )
      case (6)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P6( &
          Lift, vec_in, vec_out, Ne )
      case (7)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P7( &
          Lift, vec_in, vec_out, Ne )
      case (8)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P8( &
          Lift, vec_in, vec_out, Ne )
      case (9)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P9( &
          Lift, vec_in, vec_out, Ne )
      case (10)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P10( &
          Lift, vec_in, vec_out, Ne )
      case (11)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P11( &
          Lift, vec_in, vec_out, Ne )
      case (12)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P12( &
          Lift, vec_in, vec_out, Ne )
      case (13)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P13( &
          Lift, vec_in, vec_out, Ne )
      case (14)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P14( &
          Lift, vec_in, vec_out, Ne )
      case (15)
        call element_operation_kernel_matvec_Lift_hexahedral_all_P15( &
          Lift, vec_in, vec_out, Ne )
      end select
    end if

    return
  end subroutine tensorprod_Lift_hexahedral_all

  !------------------------------------------------------------------

  !--------------------------------------------------
  !> Tensor-product differentiation
!OCL SERIAL
  subroutine tensorprod_divlike_dirXYZ_general( &
    vec_out_x, vec_out_y, vec_out_z, &
    Mat, Mat_tr,                     &
    vec_in_x, vec_in_y, vec_in_z,    &
    Nq )
    !$acc routine vector
    implicit none
    integer, intent(in) :: Nq
    real(RP), intent(in) :: Mat(Nq,Nq)
    real(RP), intent(in) :: Mat_tr(Nq,Nq)
    real(RP), intent(in) :: vec_in_x(Nq,Nq**2)
    real(RP), intent(in) :: vec_in_y(Nq,Nq,Nq)
    real(RP), intent(in) :: vec_in_z(Nq,Nq,Nq)
    real(RP), intent(out) :: vec_out_x(Nq,Nq**2)
    real(RP), intent(out) :: vec_out_y(Nq,Nq**2)
    real(RP), intent(out) :: vec_out_z(Nq,Nq**2)

    integer :: i,j,k,l,jk
    !----------------------------------------------------------

    !- x-direction
    !$acc loop vector collapse(2)
    do jk = 1, Nq**2
      do i = 1, Nq
        vec_out_x(i,jk) = 0.0_RP
        do l = 1, Nq
          vec_out_x(i,jk) = &
               vec_out_x(i,jk) &
             + Mat(i,l)*vec_in_x(l,jk)
        end do
      end do
    end do

    !- y-direction
    !$acc loop vector collapse(3)
    do k = 1, Nq
      do j = 1, Nq
        do i = 1, Nq
          jk = j + (k-1)*Nq
          vec_out_y(i,jk) = 0.0_RP
          do l = 1, Nq
            vec_out_y(i,jk) = vec_out_y(i,jk) &
               + vec_in_y(i,l,k)*Mat_tr(l,j)
          end do
        end do
      end do
    end do

    !- z-direction
    !$acc loop vector collapse(3)
    do k = 1, Nq
      do j = 1, Nq
        do i = 1, Nq
          jk = j + (k-1)*Nq
          vec_out_z(i,jk) = 0.0_RP
          do l = 1, Nq
            vec_out_z(i,jk) = vec_out_z(i,jk) &
               + vec_in_z(i,j,l)*Mat_tr(l,k)

          end do
        end do
      end do
    end do
    return
  end subroutine tensorprod_divlike_dirXYZ_general

  !> Tensor-product lifting for a hexahedral element
!OCL SERIAL
  subroutine tensorprod_Lift_hexahedral_general( &
    vec_out,         &
    Lift, vec_in, Nq )
    !$acc routine vector
    implicit none
    integer, intent(in) :: Nq
    real(RP), intent(out) :: vec_out(Nq,Nq,Nq)
    real(RP), intent(in) :: Lift(Nq,Nq,Nq,6)
    real(RP), intent(in) :: vec_in(Nq,Nq,6)

    integer :: i,j,k
    !----------------------------------------------------------

    !$acc loop vector collapse(3)
    do k = 1, Nq
    do j = 1, Nq
    do i = 1, Nq
      vec_out(i,j,k) = &
            Lift(i,j,k,1)*vec_in(i,k,1) &
          + Lift(i,j,k,2)*vec_in(j,k,2) &
          + Lift(i,j,k,3)*vec_in(i,k,3) &
          + Lift(i,j,k,4)*vec_in(j,k,4) &
          + Lift(i,j,k,5)*vec_in(i,j,5) &
          + Lift(i,j,k,6)*vec_in(i,j,6)
    end do
    end do
    end do
    return
  end subroutine tensorprod_Lift_hexahedral_general

  !> General tensor-product differentiation over all elements
!OCL SERIAL
  subroutine tensorprod_divlike_dirXYZ_all_general( &
    vec_out_x, vec_out_y, vec_out_z, &
    Mat, Mat_tr,                     &
    vec_in_x, vec_in_y, vec_in_z,    &
    Nq, Ne )
    implicit none
    integer, intent(in) :: Nq, Ne
    real(RP), intent(in) :: Mat(Nq,Nq), Mat_tr(Nq,Nq)
    real(RP), intent(in) :: vec_in_x(Nq**3,Ne)
    real(RP), intent(in) :: vec_in_y(Nq**3,Ne)
    real(RP), intent(in) :: vec_in_z(Nq**3,Ne)
    real(RP), intent(out) :: vec_out_x(Nq**3,Ne)
    real(RP), intent(out) :: vec_out_y(Nq**3,Ne)
    real(RP), intent(out) :: vec_out_z(Nq**3,Ne)

    integer :: ke
    !--------------------------------------------------

    !$omp parallel do
    !$acc parallel loop gang &
    !$acc& present(Mat,Mat_tr,vec_in_x,vec_in_y,vec_in_z) &
    !$acc& present(vec_out_x,vec_out_y,vec_out_z)
    do ke = 1, Ne
      call tensorprod_divlike_dirXYZ_general( &
        vec_out_x(:,ke), vec_out_y(:,ke), vec_out_z(:,ke), &
        Mat, Mat_tr, vec_in_x(:,ke), vec_in_y(:,ke), vec_in_z(:,ke), Nq )
    end do

    return
  end subroutine tensorprod_divlike_dirXYZ_all_general

  !> General tensor-product surface lifting over all elements
!OCL SERIAL
  subroutine tensorprod_Lift_hexahedral_all_general( &
    vec_out, Lift, vec_in, Nq, Ne )
    implicit none
    integer, intent(in) :: Nq, Ne
    real(RP), intent(out) :: vec_out(Nq**3,Ne)
    real(RP), intent(in) :: Lift(Nq,Nq,Nq,6)
    real(RP), intent(in) :: vec_in(6*Nq**2,Ne)

    integer :: ke
    !--------------------------------------------------

    !$omp parallel do
    !$acc parallel loop gang present(vec_out,Lift,vec_in)
    do ke = 1, Ne
      call tensorprod_Lift_hexahedral_general( &
        vec_out(:,ke), Lift, vec_in(:,ke), Nq )
    end do

    return
  end subroutine tensorprod_Lift_hexahedral_all_general
end module mod_dg_optr_kernel
