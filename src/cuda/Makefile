include ../../common.mk

NVCC=nvcc -arch sm_20 -Xcompiler "-fPIC"

OBJS    := $(patsubst %.cc,%.o,$(wildcard *.cc))
CUOBJS  := $(patsubst %.cu,%.o,$(wildcard *.cu))

CUDA_DIR := /usr/local/cuda

CFLAGS += -I.. -I$(CUDA_DIR)/include
CLIBS  += -L$(CUDA_DIR)/lib64 -lcublas

libgs_cuda: $(CUOBJS) $(OBJS)
	$(CC) -shared -Wl,-soname,libgs_cuda.so.1.0 -o ../../libgs_cuda.so.1.0 $(OBJS) $(CUOBJS) $(CLIBS)

$(OBJS): %.o: %.cc
	$(NVCC) -c $(DEFS) $(CFLAGS) -o $@ $<

$(CUOBJS): %.o: %.cu
	$(NVCC) -c $(DEFS) $(CFLAGS) -o $@ $<

clean:
	rm -f *.o *.d *~ ../../libgs_cuda.so*
