CC=gcc -O3
# -g -pg
DEFS=
#DEFS+=-DDEBUG
OBJS=main.o # src/gs_element.o src/gs_node.o src/gs_edge.o src/gs_graph.o src/gs_common.o src/gs_stream.o src/gs_stream_dgs.o src/gs_id.o src/gs_graph_bfs.o src/gs_algorithm_unweighted_eccentricity.o src/gs_algorithm_diameter.o src/gs_matrix.o src/gs_matrix_bfs.o
EINA_DIR=/usr/local
CFLAGS=-Isrc/ -I$(EINA_DIR)/include/eina-1 -I$(EINA_DIR)/include/eina-1/eina
CLIBS=-L$(EINA_DIR)/lib -leina -L. -l:libgs.so.1.0 -l:libgs_cuda.so.1.0

test: libgs $(OBJS)
	$(CC) -o test $(OBJS) $(CLIBS) 

libgs:
	make -C src libgs

.c.o :
	$(CC) -c $(DEFS) $(CFLAGS) $<

clean:
	make -C src clean
	rm -f *.o *.d *~ test
