
CC=gcc
CFLAGS=-fPIC -shared -g -Wall -D__MACOS__
LD_FLAGS=-Wl, -ldl -lpthread

.PHONY: all test
all: libwrap.so

libwrap.so: src/wrap.c os/macOS/os.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/macOS/$@ $^ -e,prog_version $(LD_FLAGS)

clean:
	rm -f ./lib/macOS/libwrap.so
	rm -f test/macOS/test

test: contrib/libyaml/src/.libs/libyaml.a contrib/cmocka/build/src/libcmocka.dylib test/test.c test/cfgtest.c src/cfg.c
	@echo "Building Tests"	
	$(CC) -Wall -g -o test/macOS/test test/test.c 
	@echo "Building cfgtest without YAML only to inspect size"
	$(CC) -Wall -g -DNO_YAML -o test/macOS/cfgtest -I./src -I./contrib/libyaml/include -Icontrib/cmocka/include test/cfgtest.c src/cfg.c contrib/libyaml/src/.libs/libyaml.a -Lcontrib/cmocka/build/src -lcmocka && ls -al test/macOS/cfgtest
	@echo "Building cfgtest with YAML to inspect size and for real testing"
	$(CC) -Wall -g -O0 -coverage -o test/macOS/cfgtest -I./src -I./contrib/libyaml/include -Icontrib/cmocka/include test/cfgtest.c src/cfg.c contrib/libyaml/src/.libs/libyaml.a -Lcontrib/cmocka/build/src -lcmocka && ls -al test/macOS/cfgtest
	@echo "Running Tests"
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/macOS/cfgtest
	@echo "Testing Coverage"
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
