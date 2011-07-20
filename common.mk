CC=gcc -fPIC -O3
EINA_DIR=/usr/local

DEFS=

CFLAGS=-I$(EINA_DIR)/include/eina-1 -I$(EINA_DIR)/include/eina-1/eina
CLIBS =-L$(EINA_DIR)/lib -leina