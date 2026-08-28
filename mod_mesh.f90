!-------------------------------------------------------------------------------
!> module mesh
!!
!! @par Description
!!      A module to manage DG element and mesh
!!
!! @author Yuta Kawai, Team SCALE
!<
module mod_mesh
  !-----------------------------------------------------------------------------
  !
  !++ Used modules
  !
  use mod_common, only: RP, PI, ACC_QUEUE
  implicit none
  private

  !-----------------------------------------------------------------------------
  !
  !++ Public procedure
  !

  public :: mesh_setup
  public :: mesh_finalize
  public :: update_halo
  
  !-----------------------------------------------------------------------------
  !
  !++ Public parameters & variables
  !

  !- DG element

  integer, public :: PolyOrder  !< Polynomial order
  integer, public :: Nq         !< Number of nodes in 1D (Nq = PolyOrder + 1)
  integer, public :: Np         !< Number of nodes per element
  integer, public :: Nfp        !< Number of nodes per face
  integer, public :: NfpTot     !< Total number of face nodes

  integer, public, allocatable :: Fmask(:,:)         !< Face node indices
  real(RP), public, allocatable :: x1D(:)            !< 1D coordinates in reference element
  real(RP), public, allocatable :: D1D(:,:)          !< 1D differentiation matrix
  real(RP), public, allocatable :: D1D_tr(:,:)       !< Transpose of 1D differentiation matrix
  real(RP), public, allocatable :: Lift_mat(:,:,:,:) !< Lift matrix
  real(RP), public, allocatable :: Lift1D(:,:)        !< Separable face lift coefficients

  !- Mesh

  integer, public :: Ne    !< Number of elements
  integer, public :: NeA   !< Number of elements including halo buffer
  integer, public, allocatable :: VMapM(:,:)        !< Mapping of face nodes to element nodes (minus side)
  integer, public, allocatable :: VMapP(:,:)        !< Mapping of face nodes to element nodes (plus side)
  real(RP), public, allocatable :: normal_fn(:,:,:) !< Face normal vectors
  real(RP), public, allocatable :: Escale(:,:,:)    !< Element scaling factors
  real(RP), public, allocatable :: Fscale(:,:)      !< Face scaling factors
  real(RP), public, allocatable :: pos_en(:,:,:)    !< Element node coordinates


  !-----------------------------------------------------------------------------
  !
  !++ Private procedure
  !
  !-----------------------------------------------------------------------------
  !
  !++ Private parameters & variables
  !

  integer :: Nhalo                         !< Halo-buffer size in element units
  integer :: NhaloNode                     !< Number of face nodes stored in halo buffer
  integer, allocatable :: halo_src_map(:)  !< Source DOF index for each halo-buffer node

  character(len=*), parameter :: operator_data_dir = "operator_data"
  
contains
!OCL SERIAL
  subroutine mesh_setup( NeX, NeY, NeZ, p, &
    domsize_x, domsize_y, domsize_z )
    implicit none
    integer, intent(in) :: NeX, NeY, NeZ                    !< Number of elements in x, y, z directions
    integer, intent(in) :: p                                !< Polynomial order
    real(RP), intent(in) :: domsize_x, domsize_y, domsize_z !< Domain size in x, y, z directions
    !----------------------------------------------------------

    integer :: i, j, k
    integer :: ke
    integer :: f, fp
    integer :: keP         !< Neighboring element index
    integer :: oppface     ! Opposite face index
    logical :: is_boundary ! Flag indicating if the face is on the boundary
    integer :: ibuf        ! Index for halo buffer

    real(RP) :: dx, dy, dz
    real(RP) :: xc, yc, zc
    !----------------------------------------------------------

    call element_init( p )

    Ne = NeX * NeY * NeZ
    Nhalo = 2 * ( NeY * NeZ + NeX * NeZ + NeX * NeY )
    NhaloNode = Nfp * Nhalo

    ! The halo contains only face points and is packed directly after the
    ! owned Np*Ne values.  NeA is an allocation extent, not a count of full
    ! halo elements, so reserve only the number of Np-sized columns needed to
    ! hold those points.  This matters at high order: for p=511, Ne=1 the old
    ! Ne+Nhalo extent reserved six unused 512^3-point elements per field even
    ! though the complete halo has only 6*512^2 points.
    NeA = Ne + (NhaloNode + Np - 1) / Np

    allocate(halo_src_map(NhaloNode))

    allocate(VMapM(NfpTot,Ne), VMapP(NfpTot,Ne))

    allocate(normal_fn(NfpTot,Ne,3))
    allocate(Escale(Np,Ne,3))
    allocate(Fscale(NfpTot,Ne))

    allocate(pos_en(Np,Ne,3))

    dx = domsize_x / real(NeX,RP)
    dy = domsize_y / real(NeY,RP)
    dz = domsize_z / real(NeZ,RP)

    !----------------------------------------------------------
    ! Geometry and VMapM / VMapP
    !
    ! Interior face:
    !   VMapP -> DOF of neighboring owned element
    !
    ! Boundary face:
    !   VMapP -> packed halo buffer
    !     Np*Ne + 1
    !     Np*Ne + 2
    !     ...
    !
    ! halo_src_map stores the corresponding source DOF on the
    ! opposite periodic boundary.
    !----------------------------------------------------------

    ibuf = 0

    !$omp parallel do private(ke,xc,yc,zc,f,fp,keP,oppface,is_boundary)
    do k = 1, NeZ
    do j = 1, NeY
    do i = 1, NeX

      ke = elem_id(i,j,k,NeX,NeY)

      xc = ( real(i,RP) - 0.5_RP ) * dx
      yc = ( real(j,RP) - 0.5_RP ) * dy
      zc = ( real(k,RP) - 0.5_RP ) * dz

      call set_element_position(ke, xc,yc,zc, dx,dy,dz)

      Escale(:,ke,1) = 2.0_RP / dx
      Escale(:,ke,2) = 2.0_RP / dy
      Escale(:,ke,3) = 2.0_RP / dz

      call set_face_geometry(ke, dx, dy, dz)

      do f = 1, 6
        call eval_neighbor_element( &
              i,j,k,f, NeX,NeY,NeZ,     & ! (in)
              keP, oppface, is_boundary ) ! (out)

        do fp = 1, Nfp
          VMapM((f-1)*Nfp+fp,ke) = Fmask(fp,f) + (ke-1)*Np
        end do
        if ( .not. is_boundary ) then
          do fp = 1, Nfp
            VMapP((f-1)*Nfp+fp,ke) = Fmask(fp,oppface) + (keP-1)*Np
          end do
        end if
      end do
    end do
    end do
    end do

    ! This loop should be serialized
    do k = 1, NeZ
    do j = 1, NeY
    do i = 1, NeX
      ke = elem_id(i,j,k,NeX,NeY)
      do f = 1, 6
        call eval_neighbor_element( i,j,k,f, NeX,NeY,NeZ, & ! (in)
              keP, oppface, is_boundary )                   ! (out)

        do fp = 1, Nfp
          if ( is_boundary ) then
            ibuf = ibuf + 1
            VMapP((f-1)*Nfp+fp,ke) = Np*Ne + ibuf
            halo_src_map(ibuf) = Fmask(fp,oppface) + (keP-1)*Np
          end if
        end do
      end do
    end do
    end do
    end do

    !$acc enter data copyin(halo_src_map)

    return
  end subroutine mesh_setup


  !> Release device data owned by the mesh module
!OCL SERIAL
  subroutine mesh_finalize()
    implicit none
    !------------------------------------------------------------

    !$acc exit data delete(halo_src_map)
    deallocate(halo_src_map)

    return
  end subroutine mesh_finalize


  !> Update the halo values of a field variable
  !
  !  The caller stores the field as
  !
  !    f(Np,NeA)
  !
  !  while this routine views the same memory as a flattened
  !  one-dimensional array.  The halo region starts immediately
  !  after the owned-element DOFs:
  !
  !    f(Np*Ne+1 : Np*Ne+NhaloNode)
  !
!OCL SERIAL
  subroutine update_halo(f)
    implicit none
    ! The caller owns the allocation extent.  Keep this dummy assumed-size so
    ! merely forming Np*NeA cannot overflow the default integer kind at very
    ! high order; the actual halo indices below remain within their validated
    ! 32-bit range.
    real(RP), intent(inout) :: f(*)

    integer :: ibuf
    !------------------------------------------------------------

    !$acc parallel loop gang vector present(f,halo_src_map) async(ACC_QUEUE)
    do ibuf = 1, NhaloNode
      f(Np*Ne+ibuf) = f(halo_src_map(ibuf))
    end do

    return
  end subroutine update_halo
  
  
  !- Private subroutines -----------------------------------------

  subroutine element_init( p )
    implicit none
    integer, intent(in) :: p

    integer :: fid
    integer :: i, j, k, fp

    logical :: have_file, dense_lift
    character(len=128) :: fname
    !----------------------------------------------------------

    PolyOrder = p

    Nq     = p + 1
    Np     = Nq**3
    Nfp    = Nq**2
    NfpTot = 6*Nfp

    allocate(x1D(Nq))
    allocate(Fmask(Nfp,6))

    allocate(D1D(Nq,Nq), D1D_tr(Nq,Nq))
    allocate(Lift1D(Nq,6))

    !----------------------------------------------------------
    ! Operator data
    !----------------------------------------------------------

    write(fname,'(a,"/p",I0,".dat")') operator_data_dir, p
    inquire(file=trim(fname), exist=have_file)

    ! The dense Lift_mat is a repackaging of Lift1D that costs 6*Nq**3 doubles.
    ! Keep it where the OpenACC / CPU paths can afford it (100 MB at Nq=128);
    ! at Nq=256 it would be 805 MB, so that order gets a dummy allocation and
    ! only the separable Lift1D.
    dense_lift = (Nq <= 128)

    if (dense_lift) then
      allocate(Lift_mat(Nq,Nq,Nq,6))
    else
      allocate(Lift_mat(1,1,1,1))
    end if

    if (have_file) then
      fid = 20
      open(fid,file=trim(fname),status='old',action='read')

      read(fid,*) x1D
      read(fid,*) D1D
      read(fid,*) Lift_mat
      close(fid)

      Lift1D(:,1) = Lift_mat(1,:,1,1)
      Lift1D(:,2) = Lift_mat(:,1,1,2)
      Lift1D(:,3) = Lift_mat(1,:,1,3)
      Lift1D(:,4) = Lift_mat(:,1,1,4)
      Lift1D(:,5) = Lift_mat(1,1,:,5)
      Lift1D(:,6) = Lift_mat(1,1,:,6)
    else
      call generate_lgl_operators(p,x1D,D1D,Lift1D)
      if (dense_lift) call expand_lift1d(Lift1D,Lift_mat,Nq)
    end if

    D1D_tr = transpose(D1D)

    !----------------------------------------------------------
    ! Face node indices
    ! Element node ordering: n = i + (j-1)*Nq + (k-1)*Nq**2
    !
    ! Faces:
    ! 1 : -y
    ! 2 : +x
    ! 3 : +y
    ! 4 : -x
    ! 5 : -z
    ! 6 : +z
    !----------------------------------------------------------

    fp = 0
    do k = 1, Nq
      do i = 1, Nq
        fp = fp + 1
        Fmask(fp,1) = i + (k-1)*Nq**2
      end do
    end do

    fp = 0
    do k = 1, Nq
      do j = 1, Nq
        fp = fp + 1
        Fmask(fp,2) = Nq + (j-1)*Nq + (k-1)*Nq**2
      end do
    end do

    fp = 0
    do k = 1, Nq
      do i = 1, Nq
        fp = fp + 1
        Fmask(fp,3) = i + (Nq-1)*Nq + (k-1)*Nq**2
      end do
    end do

    fp = 0
    do k = 1, Nq
      do j = 1, Nq
        fp = fp + 1
        Fmask(fp,4) = 1 + (j-1)*Nq + (k-1)*Nq**2
      end do
    end do

    fp = 0
    do j = 1, Nq
      do i = 1, Nq
        fp = fp + 1
        Fmask(fp,5) = i + (j-1)*Nq
      end do
    end do

    fp = 0
    do j = 1, Nq
      do i = 1, Nq
        fp = fp + 1
        Fmask(fp,6) = i + (j-1)*Nq + (Nq-1)*Nq**2
      end do
    end do

    return
  end subroutine element_init


  !> Generate the Legendre-Gauss-Lobatto operators of degree p at startup.
  !! Used for the orders that have no operator_data/p<p>.dat file.  Keeping
  !! them generated avoids text data files that grow as 6*Nq**3 values.
  subroutine generate_lgl_operators(p,nodes,deriv,lift1d)
    implicit none
    integer, intent(in) :: p
    real(RP), intent(out) :: nodes(p+1)
    real(RP), intent(out) :: deriv(p+1,p+1)
    real(RP), intent(out) :: lift1d(p+1,6)

    integer :: n
    integer :: i, j, iter
    real(RP) :: x, xnew, pn_i, pnm1_i, pn_j, pnm1_j
    real(RP), allocatable :: left_lift(:), right_lift(:)
    real(RP), parameter :: tolerance = 32.0_RP*epsilon(1.0_RP)
    !------------------------------------------------------------

    n = p + 1
    allocate(left_lift(n), right_lift(n))

    nodes(1) = -1.0_RP
    nodes(n) = 1.0_RP
    do i = 2, n-1
      x = -cos(PI*real(i-1,RP)/real(p,RP))
      do iter = 1, 100
        call eval_legendre_pair(p,x,pn_i,pnm1_i)
        xnew = x - (x*pn_i-pnm1_i)/(real(p+1,RP)*pn_i)
        if (abs(xnew-x) <= tolerance*max(1.0_RP,abs(xnew))) exit
        x = xnew
      end do
      if (iter > 100) error stop "LGL node iteration did not converge"
      nodes(i) = xnew
    end do

    ! Enforce the exact symmetry of the LGL grid.
    do i = 1, n/2
      x = 0.5_RP*(nodes(n+1-i)-nodes(i))
      nodes(i) = -x
      nodes(n+1-i) = x
    end do
    if (mod(n,2) == 1) nodes((n+1)/2) = 0.0_RP

    do i = 1, n
      call eval_legendre_pair(p,nodes(i),pn_i,pnm1_i)
      do j = 1, n
        if (i == j) then
          if (i == 1) then
            deriv(i,j) = -0.25_RP*real(p*(p+1),RP)
          else if (i == n) then
            deriv(i,j) = 0.25_RP*real(p*(p+1),RP)
          else
            deriv(i,j) = 0.0_RP
          end if
        else
          call eval_legendre_pair(p,nodes(j),pn_j,pnm1_j)
          deriv(i,j) = pn_i/(pn_j*(nodes(i)-nodes(j)))
        end if
      end do

      left_lift(i) = 0.5_RP*real(p+1,RP)*pn_i*(-1.0_RP)
      right_lift(i) = 0.5_RP*real(p+1,RP)*pn_i
    end do
    left_lift(1) = left_lift(1) &
      + 0.5_RP*real(p*(p+1),RP)
    right_lift(n) = right_lift(n) &
      + 0.5_RP*real(p*(p+1),RP)

    lift1d(:,1) = left_lift
    lift1d(:,2) = right_lift
    lift1d(:,3) = right_lift
    lift1d(:,4) = left_lift
    lift1d(:,5) = left_lift
    lift1d(:,6) = right_lift

    deallocate(left_lift, right_lift)
  end subroutine generate_lgl_operators


  !> Expand the separable 1D lift coefficients into the dense Lift_mat that
  !! the general tensor-product lift kernel indexes.  Faces 1 and 3 vary with
  !! j, faces 2 and 4 with i, faces 5 and 6 with k; this is the same structure
  !! the Lift1D slices are pulled back out of when the operators come from a
  !! data file.
  subroutine expand_lift1d(lift1d,lift_mat,nq_in)
    implicit none
    integer, intent(in) :: nq_in
    real(RP), intent(in) :: lift1d(nq_in,6)
    real(RP), intent(out) :: lift_mat(nq_in,nq_in,nq_in,6)

    integer :: i, j, k
    !------------------------------------------------------------

    do k = 1, nq_in
    do j = 1, nq_in
    do i = 1, nq_in
      lift_mat(i,j,k,1) = lift1d(j,1)
      lift_mat(i,j,k,2) = lift1d(i,2)
      lift_mat(i,j,k,3) = lift1d(j,3)
      lift_mat(i,j,k,4) = lift1d(i,4)
      lift_mat(i,j,k,5) = lift1d(k,5)
      lift_mat(i,j,k,6) = lift1d(k,6)
    end do
    end do
    end do

    return
  end subroutine expand_lift1d


  !> Evaluate P_n(x) and P_{n-1}(x) by the stable three-term recurrence.
  subroutine eval_legendre_pair(n,x,pn,pnm1)
    implicit none
    integer, intent(in) :: n
    real(RP), intent(in) :: x
    real(RP), intent(out) :: pn, pnm1

    integer :: m
    real(RP) :: p0, p1, p2
    !------------------------------------------------------------

    if (n == 0) then
      pn = 1.0_RP
      pnm1 = 1.0_RP
      return
    end if
    p0 = 1.0_RP
    p1 = x
    do m = 2, n
      p2 = (real(2*m-1,RP)*x*p1-real(m-1,RP)*p0)/real(m,RP)
      p0 = p1
      p1 = p2
    end do
    pn = p1
    pnm1 = p0
  end subroutine eval_legendre_pair


  !> Setup element coordinates
!OCL SERIAL
  subroutine set_element_position(ke, xc, yc, zc, dx, dy, dz)
    implicit none
    integer, intent(in) :: ke
    real(RP), intent(in) :: xc, yc, zc
    real(RP), intent(in) :: dx, dy, dz

    integer :: i, j, k, n
    !----------------------------------------------------------

    do k = 1, Nq
    do j = 1, Nq
    do i = 1, Nq
      n = i + (j-1)*Nq + (k-1)*Nq**2
      pos_en(n,ke,1) = xc + 0.5_RP*dx*x1D(i)
      pos_en(n,ke,2) = yc + 0.5_RP*dy*x1D(j)
      pos_en(n,ke,3) = zc + 0.5_RP*dz*x1D(k)
    end do
    end do
    end do

    return
  end subroutine set_element_position

  !> Setup face geometry
  subroutine set_face_geometry(ke, dx, dy, dz)
    implicit none
    integer, intent(in) :: ke
    real(RP), intent(in) :: dx, dy, dz

    integer :: is, ie
    !----------------------------------------------------------

    normal_fn(:,ke,:) = 0.0_RP

    ! -y
    is = 1; ie = Nfp
    normal_fn(is:ie,ke,2) = -1.0_RP
    Fscale(is:ie,ke) = 2.0_RP/dy

    ! +x
    is = Nfp + 1; ie = 2*Nfp
    normal_fn(is:ie,ke,1) = 1.0_RP
    Fscale(is:ie,ke) = 2.0_RP/dx

    ! +y
    is = 2*Nfp + 1; ie = 3*Nfp
    normal_fn(is:ie,ke,2) = 1.0_RP
    Fscale(is:ie,ke) = 2.0_RP/dy

    ! -x
    is = 3*Nfp + 1; ie = 4*Nfp
    normal_fn(is:ie,ke,1) = -1.0_RP
    Fscale(is:ie,ke) = 2.0_RP/dx

    ! -z
    is = 4*Nfp + 1; ie = 5*Nfp
    normal_fn(is:ie,ke,3) = -1.0_RP
    Fscale(is:ie,ke) = 2.0_RP/dz

    ! +z
    is = 5*Nfp + 1; ie = 6*Nfp
    normal_fn(is:ie,ke,3) = 1.0_RP
    Fscale(is:ie,ke) = 2.0_RP/dz

    return
  end subroutine set_face_geometry

  !> Get neighboring element information
  !
  !  For an interior face, keP is the neighboring owned element.
  !
  !  For a boundary face, keP is the physical element on the
  !  opposite side of the periodic domain.  Its face-node DOFs
  !  are copied to the packed halo buffer by update_halo().
  !
  subroutine eval_neighbor_element( &
       i,j,k,f, NeX,NeY,NeZ,        & ! (in)
       keP, oppface, is_boundary    ) ! (out)
    implicit none
    integer, intent(in) :: i, j, k, f
    integer, intent(in) :: NeX, NeY, NeZ
    integer, intent(out) :: keP
    integer, intent(out) :: oppface
    logical, intent(out) :: is_boundary
    !----------------------------------------------------------

    is_boundary = .false.

    select case(f)
    case(1)
      oppface = 3

      if (j > 1) then
        keP = elem_id(i,j-1,k,NeX,NeY)
      else
        keP = elem_id(i,NeY,k,NeX,NeY)
        is_boundary = .true.
      end if

    case(2)
      oppface = 4

      if (i < NeX) then
        keP = elem_id(i+1,j,k,NeX,NeY)
      else
        keP = elem_id(1,j,k,NeX,NeY)
        is_boundary = .true.
      end if

    case(3)
      oppface = 1

      if (j < NeY) then
        keP = elem_id(i,j+1,k,NeX,NeY)
      else
        keP = elem_id(i,1,k,NeX,NeY)
        is_boundary = .true.
      end if

    case(4)
      oppface = 2

      if (i > 1) then
        keP = elem_id(i-1,j,k,NeX,NeY)
      else
        keP = elem_id(NeX,j,k,NeX,NeY)
        is_boundary = .true.
      end if
    
    case(5)
      oppface = 6

      if (k > 1) then
        keP = elem_id(i,j,k-1,NeX,NeY)
      else
        keP = elem_id(i,j,NeZ,NeX,NeY)
        is_boundary = .true.
      end if

    case(6)
      oppface = 5

      if (k < NeZ) then
        keP = elem_id(i,j,k+1,NeX,NeY)
      else
        keP = elem_id(i,j,1,NeX,NeY)
        is_boundary = .true.
      end if
    end select

    return
  end subroutine eval_neighbor_element

!OCL SERIAL
  integer function elem_id(i, j, k, NeX, NeY)
    implicit none
    integer, intent(in) :: i, j, k
    integer, intent(in) :: NeX, NeY
    !----------------------------------------------------------
    elem_id = i + (j-1)*NeX + (k-1)*NeX*NeY
    return
  end function elem_id

end module mod_mesh
