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
  use mod_common, only: RP
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
    NeA = Ne + Nhalo

    NhaloNode = Nfp * Nhalo

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
    real(RP), intent(inout) :: f(Np*NeA)

    integer :: ibuf
    !------------------------------------------------------------

    !$acc parallel loop gang vector present(f,halo_src_map)
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
    allocate(Lift_mat(Nq,Nq,Nq,6))

    !----------------------------------------------------------
    ! Operator data
    !----------------------------------------------------------

    write(fname,'(a,"/p",I0,".dat")') operator_data_dir, p

    fid = 20
    open(fid,file=trim(fname),status='old',action='read')

    read(fid,*) x1D
    read(fid,*) D1D
    read(fid,*) Lift_mat
    close(fid)

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
