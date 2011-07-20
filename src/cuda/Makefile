include ../../common.mk

NVCC=nvcc -arch sm_20 -Xcompiler -fPIC

DEFS=
#DEFS+=-DDEBUG
OBJS=gs_cuda_diameter.o

EINA_DIR=/home/gsavin/local
CUDA_DIR=/usr/local/cuda

CFLAGS += -I.. -I$(CUDA_DIR)/include
CLIBS  += -L$(CUDA_DIR)/lib64 -lcublas

libgs_cuda: $(OBJS)
	$(CC) -shared -Wl,-soname,libgs_cuda.so.1.0 -o ../../libgs_cuda.so.1.0 $(OBJS) $(CLIBS)

gs_cuda_diameter.o :
	$(NVCC) -c $(DEFS) $(CFLAGS) gs_cuda_diameter.cu

clean:
	rm -f *.o *.d *~ ../../libgs_cuda.so*
