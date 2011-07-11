#DEFS=-DDEBUG
CC=gcc
DEFS=
OBJS=gs_element.o gs_node.o gs_edge.o gs_graph.o gs_common.o gs_stream.o gs_stream_dgs.o
CFLAGS=-I/usr/local/include/eina-1 -I/usr/local/include/eina-1/eina
CLIBS=-L/usr/local/lib -leina

libgs: $(OBJS)
	$(CC) -shared -Wl,-soname,libgs.so.1 -o ../libgs.so -g $(OBJS) $(CLIBS)

.c.o :
	$(CC) -fpic -O2 -g -c $(DEFS) $(CFLAGS) $<

clean:
	rm -f *.o *.d *~ ../libgs.so*
