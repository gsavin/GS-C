include common.mk

BIN := $(patsubst %.c,%,$(wildcard *.c)) 

# src/gs_element.o src/gs_node.o src/gs_edge.o src/gs_graph.o src/gs_common.o src/gs_stream.o src/gs_stream_dgs.o src/gs_id.o src/gs_graph_bfs.o src/gs_algorithm_unweighted_eccentricity.o src/gs_algorithm_diameter.o src/gs_matrix.o src/gs_matrix_bfs.o

CFLAGS += -Isrc -Isrc/cuda
CLIBS  += -L. -l:libgs.so.1.0 -l:libgs_cuda.so.1.0

all: compil $(BIN)

compil:
	make -C src all

$(BIN): %.c: %
	$(CC) $(DEFS) $(CFLAGS) $(CLIBS) $< -o $@

clean:
	make -C src clean
	rm -f *.o *.d *~ test
