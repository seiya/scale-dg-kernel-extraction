!-------------------------------------------------------------------------------
!> module common
!!
!! @par Description
!!      A common module for SCALE-DG kernel extraction.
!!      
!!
!! @author Yuta Kawai, Team SCALE
!<
module mod_common
  !-----------------------------------------------------------------------------
  !
  !++ used modules
  !  
  use, intrinsic :: iso_fortran_env, only: &
    int64, real64
  implicit none
  private
  !-----------------------------------------------------------------------------
  !
  !++ Public parameters & variables
  !
  integer, public, parameter :: RP = real64
  integer, public, parameter :: IP = int64
  real(RP), public, parameter :: PI = 4.0_RP*atan(1.0_RP)

  ! OpenACC queue used by every device region of the time-stepping loop.
  ! In a CUDA Fortran build this queue is bound to the stream of the CUDA
  ! kernels (see cuda_dg_bind_acc_stream), so that the launches keep their
  ! order without the host synchronizing after each of them.
  integer, public, parameter :: ACC_QUEUE = 1

  ! 3-stage third-order SSP Runge-Kutta method
  integer, public, parameter :: RK3s3oSSP_nstage = 3
  real(RP), public, parameter :: RK3s3oSSP_rk_a(RK3s3oSSP_nstage) = [ 0.0_RP, 0.75_RP, 1.0_RP/3.0_RP ]
  real(RP), public, parameter :: RK3s3oSSP_rk_b(RK3s3oSSP_nstage) = [ 1.0_RP, 0.25_RP, 2.0_RP/3.0_RP ]

  ! Timer
  type, public :: Timer
    integer(IP) :: count_start = 0_int64
    integer(IP) :: count_accum = 0_int64
    integer(IP) :: count_rate  = 0_int64
  end type Timer  
  public :: Timer_start, Timer_stop, Timer_add, Timer_elapsed, Timer_reset

contains
!OCL SERIAL
  subroutine Timer_start(this)
    implicit none
    type(Timer), intent(inout) :: this
    !------------------------------------------------------------------------------
    call system_clock(this%count_start, this%count_rate)
    return
  end subroutine Timer_start
!OCL SERIAL
  subroutine Timer_stop(this)
    implicit none
    type(Timer), intent(inout) :: this
    integer(IP) :: count_now
    !------------------------------------------------------------------------------
    call system_clock(count_now)
    this%count_accum = this%count_accum &
                    + (count_now - this%count_start)
    return
  end subroutine Timer_stop  
!OCL SERIAL
  subroutine Timer_add(this, time_sec)
    implicit none
    type(Timer), intent(inout) :: this
    real(RP), intent(in) :: time_sec
    integer(IP) :: count_now
    !------------------------------------------------------------------------------
    if (this%count_rate == 0_int64) then
      call system_clock(count_now, this%count_rate)
    end if
    this%count_accum = this%count_accum &
                    + nint(time_sec*real(this%count_rate,RP),IP)
    return
  end subroutine Timer_add
!OCL SERIAL
  !> Discard everything accumulated so far.  Used to drop the warm-up steps
  !! of a run from the reported time.
  subroutine Timer_reset(this)
    implicit none
    type(Timer), intent(inout) :: this
    !------------------------------------------------------------------------------
    this%count_accum = 0_int64
    return
  end subroutine Timer_reset
!OCL SERIAL
  function Timer_elapsed(this) result(time_sec)
    implicit none
    type(Timer), intent(in) :: this
    real(RP)   :: time_sec

    integer(IP) :: count_now
    !------------------------------------------------------------------------------
    !- count_rate is set by the first Timer_start, so a timer that was never
    !  started has no scale to report against.
    if (this%count_rate == 0_int64) then
      time_sec = 0.0_RP
      return
    end if
    time_sec = real(this%count_accum, RP) &
            / real(this%count_rate, RP)    
    return
  end function Timer_elapsed
end module mod_common
