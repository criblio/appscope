
CC=gcc
CFLAGS=-fPIC -shared -g -Wall
LD_FLAGS=-Wl, -ldl

.PHONY: all
all: libwrap.so

libwrap.so: src/wrap.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o $@ $^ -e,prog_version $(LD_FLAGS)

clean:
	rm -f libwrap.so
