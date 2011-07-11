#DEFS=-DDEBUG
DEFS=
OBJS=main.o
CFLAGS=-Isrc/ -I/usr/local/include/eina-1 -I/usr/local/include/eina-1/eina
CLIBS=-L/usr/local/lib -leina -L. -lgs
GS_LIB=libgs.so.1.0.0

test: libgs $(OBJS)
	gcc -o test -g $(OBJS) $(CLIBS) 

libgs:
	make -C src libgs

.c.o :
	gcc -fpic -g -c $(DEFS) $(CFLAGS) $<

clean:
	make -C src clean
	rm -f *.o *.d *~ test
