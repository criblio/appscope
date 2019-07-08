
CC=gcc
CFLAGS=-fPIC -shared -g -Wall -D__LINUX__
LD_FLAGS=-ldl -lpthread

.PHONY: all test
all: libwrap.so

OBJS: wrap.o os.o

libwrap.so: src/wrap.c os/linux/os.c 
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/linux/$@ $^ $(LD_FLAGS)

clean:
	rm -f ./lib/linux/libwrap.so
	rm -f test/linux/test

test: contrib/libyaml/src/.libs/libyaml.a contrib/cmocka/build/src/libcmocka.dylib test/test.c test/cfgtest.c src/cfg.c
	@echo "Building Tests"
	$(CC) -Wall -g -o test/linux/test test/test.c
	@echo "Building cfgtest without YAML only to inspect size"
	$(CC) -Wall -g -DNO_YAML -o test/linux/cfgtest -I./src -I./contrib/libyaml/include -Icontrib/cmocka/include test/cfgtest.c src/cfg.c contrib/libyaml/src/.libs/libyaml.a -Lcontrib/cmocka/build/src -lcmocka && ls -al test/linux/cfgtest
	@echo "Building cfgtest with YAML to inspect size and for real testing"
	$(CC) -Wall -g -O0 -coverage -o test/linux/cfgtest -I./src -I./contrib/libyaml/include -Icontrib/cmocka/include test/cfgtest.c src/cfg.c contrib/libyaml/src/.libs/libyaml.a -Lcontrib/cmocka/build/src -lcmocka && ls -al test/linux/cfgtest
	@echo "Running Tests"
	LD_LIBRARY_PATH=contrib/cmocka/build/src/ test/linux/cfgtest
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory out
	# see file:///Users/cribl/scope/out/index.html

contrib/libyaml/src/.libs/libyaml.a:
	@echo "Building libyaml"
	cd ./contrib/libyaml && ./bootstrap
	cd ./contrib/libyaml && ./configure
	cd ./contrib/libyaml && make
	# don't install on the host.  We only need the static .a file

contrib/cmocka/build/src/libcmocka.dylib:
	@echo "Building cmocka"
	cd contrib/cmocka && test -d ./build || mkdir ./build
	cd contrib/cmocka/build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ..
	cd contrib/cmocka/build && make
	# don't install on the host.
	# We can tell the linker where to find the .dylib file
