include ../../common.mk

NVCC=nvcc -arch sm_20 -Xcompiler -fPIC

OBJS    := $(patsubst %.c,%.o,$(wildcard *.c))
KERNELS := $(patsubst %.cu,%.o,$(wildcard *_kernel.cu))

CUDA_DIR := /usr/local/cuda

CFLAGS += -I.. -I$(CUDA_DIR)/include
CLIBS  += -L$(CUDA_DIR)/lib64 -lcublas

libgs_cuda: $(KERNELS) $(OBJS)
	$(CC) -shared -Wl,-soname,libgs_cuda.so.1.0 -o ../../libgs_cuda.so.1.0 $(OBJS) $(KERNELS) $(CLIBS)

$(OBJS): %.o: %.c
	$(CC) -c $(DEFS) $(CFLAGS) -o $@ $<

$(KERNELS): %.o: %.cu
	$(NVCC) -c $(DEFS) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o *.d *~ ../../libgs_cuda.so*
