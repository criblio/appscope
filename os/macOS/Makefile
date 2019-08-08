OS=macOS
CC=gcc
CFLAGS=-fPIC -shared -g -Wall -Wno-nonnull -D__MACOS__
TEST_CFLAGS=-g -Wall -Wno-nonnull -O0 -coverage -D__MACOS__
LD_FLAGS=-Wl, -ldl -lpthread
INCLUDES=-I./contrib/libyaml/include
YAML_AR=contrib/libyaml/src/.libs/libyaml.a
TEST_LIB=contrib/cmocka/build/src/libcmocka.dylib
TEST_INCLUDES=-I./src -I./contrib/cmocka/include
TEST_LD_FLAGS=-Lcontrib/cmocka/build/src -lcmocka -ldl

.PHONY: all clean test
all: $(YAML_AR) libwrap.so

libwrap.so: src/wrap.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/out.c src/format.c
	@echo "Building libwrap.so ..."
	$(CC) $(CFLAGS) -o ./lib/$(OS)/$@ $(INCLUDES) $^ -e,prog_version $(LD_FLAGS) $(YAML_AR)

clean:
	rm -f ./lib/$(OS)/libwrap.so
	rm -f test/$(OS)/test

test: $(YAML_AR) $(TEST_LIB) test/test.c src/cfgutils.c test/cfgutilstest.c src/cfg.c test/cfgtest.c src/transport.c test/transporttest.c src/log.c test/logtest.c src/out.c test/outtest.c src/format.c test/formattest.c test/wraptest.c
	@echo "Building Tests"
	$(CC) -Wall -g -o test/$(OS)/test test/test.c
	@echo "Building cfgtest without YAML only to inspect size"
	$(CC) $(TEST_CFLAGS) -DNO_YAML -o test/$(OS)/cfgtest $(INCLUDES) $(TEST_INCLUDES) test/cfgtest.c src/cfg.c $(YAML_AR) $(TEST_LD_FLAGS) && ls -al test/$(OS)/cfgtest
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgutilstest $(INCLUDES) $(TEST_INCLUDES) test/cfgutilstest.c src/cfgutils.c src/cfg.c src/out.c src/log.c src/transport.c src/format.c $(YAML_AR) $(TEST_LD_FLAGS)
	@echo "Building cfgtest with YAML to inspect size and for real testing"
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgtest $(INCLUDES) $(TEST_INCLUDES) test/cfgtest.c src/cfg.c $(YAML_AR) $(TEST_LD_FLAGS) && ls -al test/$(OS)/cfgtest
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/transporttest $(INCLUDES) $(TEST_INCLUDES) test/transporttest.c src/transport.c $(YAML_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/logtest $(INCLUDES) $(TEST_INCLUDES) test/logtest.c src/log.c src/transport.c $(YAML_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/outtest $(INCLUDES) $(TEST_INCLUDES) test/outtest.c src/out.c src/log.c src/transport.c src/format.c $(YAML_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/formattest $(INCLUDES) $(TEST_INCLUDES) test/formattest.c src/format.c $(YAML_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/wraptest $(INCLUDES) $(TEST_INCLUDES) test/wraptest.c $(YAML_AR) $(TEST_LD_FLAGS) -Llib/$(OS)/ -lwrap
	@echo "Running Tests"
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/$(OS)/cfgutilstest
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/$(OS)/cfgtest
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/$(OS)/transporttest
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/$(OS)/logtest
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/$(OS)/outtest
	DYLD_LIBRARY_PATH=contrib/cmocka/build/src/ test/$(OS)/formattest
	DYLD_LIBRARY_PATH=lib/$(OS)/:contrib/cmocka/build/src/ test/$(OS)/wraptest
	@echo "Testing Coverage"
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory out
	# see file:///Users/cribl/scope/out/index.html

$(YAML_AR):
	@echo "Building libyaml"
	cd ./contrib/libyaml && ./bootstrap
	cd ./contrib/libyaml && ./configure
	cd ./contrib/libyaml && make
	# don't install on the host.  We only need the static .a file

$(TEST_LIB):
	@echo "Building cmocka"
	cd contrib/cmocka && test -d ./build || mkdir ./build
	cd contrib/cmocka/build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ..
	cd contrib/cmocka/build && make
	# don't install on the host.
	# We can tell the linker where to find the .dylib file
