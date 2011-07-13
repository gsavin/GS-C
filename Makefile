CC=gcc -fpic -g
DEFS=
DEFS+=-DDEBUG
OBJS=main.o
CFLAGS=-Isrc/ -I/usr/local/include/eina-1 -I/usr/local/include/eina-1/eina
CLIBS=-L/usr/local/lib -leina -L. -l:libgs.so.1.0
GS_LIB=libgs.so.1.0.0

test: libgs $(OBJS)
	$(CC) -o test $(OBJS) $(CLIBS) 

libgs:
	make -C src libgs

.c.o :
	$(CC) -c $(DEFS) $(CFLAGS) $<

clean:
	make -C src clean
	rm -f *.o *.d *~ test
