ACC ?= 0
CUDA ?= 0

ifeq ($(CUDA),1)
ifeq ($(origin FC), default)
FC       = nvfortran
endif
GPUFLAGS ?= -gpu=ccnative
FFLAGS  ?= -O3 -acc=gpu -cuda $(GPUFLAGS) -cudalib=cublas -Minfo=accel
NVCC ?= nvcc
GPUNVCCFLAGS ?= -arch=native
NVCCFLAGS ?= -O3 -std=c++17 $(GPUNVCCFLAGS)
CUTLASS_HOME ?= third_party/cutlass
NVCCFLAGS += -I$(CUTLASS_HOME)/include --expt-relaxed-constexpr
CUDA_KERNEL_OBJ = mod_cuda_dg_kernels.o cuda_dg_kernels_tc.o cuda_dg_kernels_fused.o cuda_dg_kernels_fused_highp.o cuda_cublas_gemm.o cuda_cutlass_gemm_fused.o cuda_ozaki2_gemm.o cuda_ozaki1_gemm.o
LDLIBS ?= -c++libs -lcublas
else ifeq ($(ACC),1)
ifeq ($(origin FC), default)
FC       = nvfortran
endif
GPUFLAGS ?= -gpu=ccnative
FFLAGS  ?= -O3 -acc=gpu $(GPUFLAGS) -Minfo=accel
CUDA_KERNEL_OBJ = mod_cuda_dg_kernels_stub.o
else
ifeq ($(origin FC), default)
FC       = gfortran
endif
FFLAGS  ?= -O3 -fopenmp
CUDA_KERNEL_OBJ = mod_cuda_dg_kernels_stub.o
endif

TARGET = scale-dg_extraction
OBJS   = mod_common.o         \
         mod_mesh.o                \
		 mod_dg_optr_kernel_opt1.o \
		 mod_dg_optr_kernel.o      \
		 $(CUDA_KERNEL_OBJ)         \
		 mod_advect3d_eq.o         \
		 main.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^ $(LDLIBS)

%.f90: %.F90.erb
	erb $< > $@

.SUFFIXES:
.SUFFIXES: .o .f90 .c .erb .mod

mod_dg_optr_kernel_opt1.f90: mod_dg_optr_kernel_opt1.F90.erb


%.o: %.f90
	$(FC) $(FFLAGS) -c $<

%.o: %.cuf
	$(FC) $(FFLAGS) -c $<

cuda_dg_kernels_tc.o: cuda_dg_kernels_tc.cu fused_kernel_geom.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_dg_kernels_fused.o: cuda_dg_kernels_fused.cu fused_kernel_geom.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_dg_kernels_fused_highp.o: cuda_dg_kernels_fused_highp.cu fused_kernel_geom.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_cublas_gemm.o: cuda_cublas_gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_cutlass_gemm_fused.o: cuda_cutlass_gemm_fused.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_ozaki2_gemm.o: cuda_ozaki2_gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

cuda_ozaki1_gemm.o: cuda_ozaki1_gemm.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


# Dependency
cuda_cutlass_gemm_fused.o: cutlass_z_gemm_assembly.h cutlass_f64_kdeep_mma.h
mod_mesh.o: mod_common.o
mod_dg_optr_kernel_opt1.o: mod_common.o
mod_dg_optr_kernel.o: mod_common.o mod_dg_optr_kernel_opt1.o
mod_cuda_dg_kernels.o: mod_common.o cuda_dg_kernels_tc.o cuda_dg_kernels_fused.o cuda_dg_kernels_fused_highp.o cuda_cublas_gemm.o cuda_cutlass_gemm_fused.o cuda_ozaki2_gemm.o cuda_ozaki1_gemm.o
mod_cuda_dg_kernels_stub.o: mod_common.o
mod_advect3d_eq.o: mod_common.o mod_dg_optr_kernel.o $(CUDA_KERNEL_OBJ)
main.o: mod_mesh.o mod_advect3d_eq.o

clean:
	rm -f $(TARGET) *.o *.mod
