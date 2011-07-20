CC=gcc -fPIC -O3
NVCC=nvcc -arch sm_20 -Xcompiler -fPIC
# -g -pg
DEFS=
#DEFS+=-DDEBUG
OBJS=gs_cuda_diameter.o

EINA_DIR=/home/gsavin/local
CUDA_DIR=/usr/local/cuda

CFLAGS=-I$(EINA_DIR)/include/eina-1 -I$(EINA_DIR)/include/eina-1/eina -I.. -I$(CUDA_DIR)/include
CLIBS=-L$(EINA_DIR)/lib -leina -L$(CUDA_DIR)/lib64 -lcublas

libgs_cuda: $(OBJS)
	$(CC) -shared -Wl,-soname,libgs_cuda.so.1.0 -o ../../libgs_cuda.so.1.0 $(OBJS) $(CLIBS)

gs_cuda_diameter.o :
	$(NVCC) -c $(DEFS) $(CFLAGS) gs_cuda_diameter.cu

clean:
	rm -f *.o *.d *~ ../../libgs_cuda.so*
