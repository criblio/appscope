
CC=gcc
CFLAGS=-fPIC -shared -g -Wall
LD_FLAGS=-Wl, -ldl

.PHONY: all test
all: libwrap.so

libwrap.so: src/wrap.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/macOS/$@ $^ -e,prog_version $(LD_FLAGS)

clean:
	rm -f ./lib/macOS/libwrap.so
	rm -f test/macOS/test

test: test/test.c
	@echo "Building Tests"	
	$(CC) -Wall -g -o test/macOS/test $^
