CC=gcc -fPIC -g
GLIB_DIR=/usr/

DEFS=

CFLAGS=-I$(GLIB_DIR)/include/glib-2.0 -I$(GLIB_DIR)/lib/glib-2.0/include
CLIBS=-lglib-2.0
