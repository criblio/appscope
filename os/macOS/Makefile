CC=gcc
CFLAGS=-fPIC -shared -g -Wall -D__MACOS__
LD_FLAGS=-Wl, -ldl -lpthread
INCLUDES=-I./contrib/libyaml/include
STATIC_LIBS=contrib/libyaml/src/.libs/libyaml.a
TEST_INCLUDES=-I./src -I./contrib/cmocka/include
TEST_LD_FLAGS=-Lcontrib/cmocka/build/src -lcmocka

.PHONY: all test
all: libwrap.so

libwrap.so: src/wrap.c src/cfg.c os/macOS/os.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/macOS/$@ $(INCLUDES) $^ -e,prog_version $(LD_FLAGS) $(STATIC_LIBS)

clean:
	rm -f ./lib/macOS/libwrap.so
	rm -f test/macOS/test

test: $(STATIC_LIBS) contrib/cmocka/build/src/libcmocka.dylib test/test.c test/cfgtest.c src/cfg.c
	@echo "Building Tests"	
	$(CC) -Wall -g -o test/macOS/test test/test.c
	@echo "Building cfgtest without YAML only to inspect size"
	$(CC) -Wall -g -DNO_YAML -o test/macOS/cfgtest $(INCLUDES) $(TEST_INCLUDES) test/cfgtest.c src/cfg.c $(STATIC_LIBS) $(TEST_LD_FLAGS) && ls -al test/macOS/cfgtest
	@echo "Building cfgtest with YAML to inspect size and for real testing"
	$(CC) -Wall -g -O0 -coverage -o test/macOS/cfgtest $(INCLUDES) $(TEST_INCLUDES) test/cfgtest.c src/cfg.c $(STATIC_LIBS) $(TEST_LD_FLAGS) && ls -al test/macOS/cfgtest
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
