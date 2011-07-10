#DEFS=-DDEBUG
DEFS=
OBJS=main.o gs_element.o gs_node.o gs_edge.o gs_graph.o gs_common.o gs_stream.o
CFLAGS=-I/usr/local/include/eina-1 -I/usr/local/include/eina-1/eina
CLIBS=-L/usr/local/lib -leina

test: $(OBJS)
	gcc -o test -g $(OBJS) $(CLIBS) 

.c.o :
	gcc -g -c $(DEFS) $(CFLAGS) $<

clean:
	rm -f *.o *.d *~ test
