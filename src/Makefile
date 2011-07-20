include ../common.mk

OBJS=gs_element.o gs_node.o gs_edge.o gs_graph.o gs_common.o gs_stream.o gs_stream_dgs.o gs_id.o gs_graph_bfs.o gs_algorithm_unweighted_eccentricity.o gs_algorithm_diameter.o gs_matrix.o gs_matrix_bfs.o

all: libgs libgs_cuda

libgs: $(OBJS)
	$(CC) -shared -Wl,-soname,libgs.so.1.0 -o ../libgs.so.1.0 $(OBJS) $(CLIBS)

libgs_cuda:
	make -C cuda libgs_cuda

.c.o :
	$(CC) -c $(DEFS) $(CFLAGS) $<

clean:
	rm -f *.o *.d *~ ../libgs.so*
