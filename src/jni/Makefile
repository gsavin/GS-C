include ../../common.mk


OBJS := $(patsubst %.c,%.o,$(wildcard *.c))

JAVA_HOME=/home/opt/jdk1.6.0_25

CFLAGS += -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I..
CLIBS  += -L../.. -l:libgs.so.1.0 -ldl -lpthread

CLASSES = org/graphstream/graph/implementations/NativeElement.java \
	org/graphstream/util/NativePointer.java

all: libgs_native java

libgs_native: $(OBJS)
	$(CC) -shared -Wl,-soname,libgs_native.so.1.0 -o libgs_native.so.1.0 $(OBJS) $(CLIBS)

java: $(CLASSES)
	javac -cp .:/home/raziel/workspace/gs-core/bin/ $(CLASSES)

.cc.o :
	$(CC) -c $(DEFS) $(CFLAGS) $<

clean:
	rm -f *.o *.d *~ libgs_native.so*
