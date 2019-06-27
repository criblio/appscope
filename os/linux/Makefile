
CC=gcc
CFLAGS=-fPIC -shared -g -Wall -D__LINUX__
LD_FLAGS=-ldl

.PHONY: all test
all: libwrap.so

OBJS: wrap.o os.o

libwrap.so: src/wrap.c os/linux/os.c 
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/linux/$@ $^ $(LD_FLAGS)

clean:
	rm -f ./lib/linux/libwrap.so
	rm -f test/linux/test

test: test/test.c
	@echo "Building Tests"
	$(CC) -Wall -g -o test/linux/test $^
