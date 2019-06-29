
CC=gcc
CFLAGS=-fPIC -shared -g -Wall -D__MACOS__
LD_FLAGS=-Wl, -ldl

.PHONY: all test
all: libwrap.so

libwrap.so: src/wrap.c os/macOS/os.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/macOS/$@ $^ -e,prog_version $(LD_FLAGS)

clean:
	rm -f ./lib/macOS/libwrap.so
	rm -f test/macOS/test

test: test/test.c contrib/libyaml/src/.libs/libyaml.a test/cfgtest.c src/cfg.c
	@echo "Building Tests"	
	$(CC) -Wall -g -o test/macOS/test test/test.c 
	$(CC) -Wall -g -o test/macOS/cfgtest -I./src -I./contrib/libyaml/include contrib/libyaml/src/.libs/libyaml.a test/cfgtest.c src/cfg.c
	test/macOS/cfgtest

contrib/libyaml/src/.libs/libyaml.a:
	@echo "Building libyaml"
	cd ./contrib/libyaml && ./bootstrap
	cd ./contrib/libyaml && ./configure
	cd ./contrib/libyaml && make
