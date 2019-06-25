
CC=gcc
CFLAGS=-fPIC -shared -g -Wall
LD_FLAGS=-ldl

.PHONY: all test
all: libwrap.so

libwrap.so: src/wrap.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/linux/$@ $^ $(LD_FLAGS)

clean:
	rm -f ./lib/linux/libwrap.so
	rm -f test/linux/test

test: test/test.c
	@echo "Building Tests"
	$(CC) -Wall -g -o test/linux/test $^
