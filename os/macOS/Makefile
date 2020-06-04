OS=macOS
CC=gcc
CFLAGS=-fPIC -g -Wall -Wno-nonnull -D__MACOS__
SCOPE_VER:="$(shell git --no-pager describe --abbrev=12 --dirty=+ --always --tags)"
TEST_CFLAGS=-g -Wall -Wno-nonnull -O0 -coverage -D__MACOS__
TEST_CFLAGS+=-DSCOPE_VER=\"$(SCOPE_VER)\"
TEST_CFLAGS +=-D__MACOS__
LD_FLAGS=$(PCRE2_AR) -Wl, -ldl -lpthread
INCLUDES=-I./contrib/libyaml/include -I./contrib/cJSON -I./os/$(OS) -I./contrib/pcre2/src -I./contrib/pcre2/build
YAML_DEFINES=-DYAML_VERSION_MAJOR="0" -DYAML_VERSION_MINOR="2" -DYAML_VERSION_PATCH="2" -DYAML_VERSION_STRING="\"0.2.2\""
YAML_SRC=$(wildcard contrib/libyaml/src/*.c)
YAML_AR=contrib/libyaml/src/.libs/libyaml.a
JSON_AR=contrib/cJSON/libcjson.a
PCRE2_AR=contrib/pcre2/build/libpcre2-posix.a contrib/pcre2/build/libpcre2-8.a
TEST_AR=$(YAML_AR) $(JSON_AR) $(PCRE2_AR)
TEST_LIB=contrib/cmocka/build/src/libcmocka.dylib
TEST_INCLUDES=-I./src -I./contrib/cmocka/include
TEST_LD_FLAGS=-Lcontrib/cmocka/build/src -lcmocka -ldl

.PHONY: all clean test
all: libscope.so

$(PCRE2_AR):
	@echo "Building pcre2"
	cd contrib/pcre2 && mkdir -p build
	cd contrib/pcre2/build && cmake ..
	cd contrib/pcre2/build && make

libscope.so: src/wrap.c src/state.c src/report.c src/plattime.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c $(YAML_SRC) contrib/cJSON/cJSON.c
	@echo "Building libscope.so ..."
	make $(PCRE2_AR)
	$(CC) $(CFLAGS) -shared -fvisibility=hidden -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) -o ./lib/$(OS)/$@ $(INCLUDES) $^ -e,prog_version $(LD_FLAGS)
	$(CC) -c $(CFLAGS) -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) $(INCLUDES) $^
	rm -rf ./test/selfinterpose && mkdir ./test/selfinterpose && mv *.o ./test/selfinterpose/

clean:
	rm -f ./lib/$(OS)/libscope.so

SRC_C_FILES:=$(wildcard src/*.c)
SRC_C_FILES:=$(filter-out src/wrap.c, $(SRC_C_FILES))
TEST_C_FILES:=$(wildcard test/*.c)
TEST_C_FILES:=$(filter-out test/glibcvertest.c test/wraptest.c, $(TEST_C_FILES))
C_FILES:=$(SRC_C_FILES) $(TEST_C_FILES) os/$(OS)/os.c
O_FILES:=$(C_FILES:.c=.o)

test: $(C_FILES)
	@echo "Building Tests"
	$(CC) -c $(TEST_CFLAGS) $^ $(INCLUDES) $(TEST_INCLUDES)
	make $(YAML_AR)
	make $(JSON_AR)
	make $(TEST_LIB)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgutilstest cfgutilstest.o cfgutils.o cfg.o mtc.o log.o evtformat.o ctl.o transport.o mtcformat.o dbg.o circbuf.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgtest cfgtest.o cfg.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/transporttest transporttest.o transport.o dbg.o log.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/logtest logtest.o log.o transport.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtctest mtctest.o mtc.o log.o transport.o mtcformat.o dbg.o circbuf.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/evtformattest evtformattest.o evtformat.o log.o transport.o mtcformat.o dbg.o cfg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/ctltest ctltest.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	#$(CC) $(TEST_CFLAGS) -o test/$(OS)/reporttest reporttest.o report.o state.o com.o plattime.o os.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o search.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtcformattest mtcformattest.o mtcformat.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/circbuftest circbuftest.o circbuf.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/linklisttest linklisttest.o linklist.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/comtest comtest.o com.o ctl.o log.o transport.o evtformat.o circbuf.o mtcformat.o cfgutils.o cfg.o mtc.o dbg.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dbgtest dbgtest.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/selfinterposetest selfinterposetest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dnstest dnstest.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/searchtest searchtest.o search.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	@echo "Running Tests and Generating Test Coverage"
	test/execute.sh
# see file:///Users/cribl/scope/coverage/index.html

$(YAML_AR): $(YAML_SRC)
	@echo "Building libyaml"
	cd ./contrib/libyaml && ./bootstrap
	cd ./contrib/libyaml && ./configure LIBS=-ldl
	cd ./contrib/libyaml && make
# don't install on the host.  We only need the static .a file

$(JSON_AR): contrib/cJSON/cJSON.c
	@echo "Building cJSON"
	cd ./contrib/cJSON && make clean
	cd ./contrib/cJSON && make all

$(TEST_LIB):
	@echo "Building cmocka"
	cd contrib/cmocka && test -d ./build || mkdir ./build
	cd contrib/cmocka/build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ..
	cd contrib/cmocka/build && make
# don't install on the host.
# We can tell the linker where to find the .dylib file
