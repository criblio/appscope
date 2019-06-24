
CC=gcc
CFLAGS=-fPIC -shared -ldl -g -Wall
LD_FLAGS=-Wl, $(STATIC_CPP) -ldl

.PHONY: all
all: CXXFLAGS += -O2 -s
all: build

debug: CXXFLAGS += -ggdb
debug: clean build tester 


libwrap.so: src/wrap.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o $@ $^ -Wl,-e,prog_version $(LD_FLAGS)

clean:
	rm -f libwrap.so
