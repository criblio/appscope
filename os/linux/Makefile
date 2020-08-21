CWD = $(shell pwd)
OS=linux
CC=gcc
CFLAGS=-fPIC -g -Wall -Wno-nonnull -Wno-deprecated-declarations -D__LINUX__ -D__GO__
SCOPE_VER:="$(shell git --no-pager describe --abbrev=12 --dirty=+ --always --tags)"
TEST_CFLAGS=-g -Wall -Wno-nonnull -O0 -coverage -D__LINUX__
TEST_CFLAGS+=-DSCOPE_VER=\"$(SCOPE_VER)\"
LD_FLAGS=$(PCRE2_AR) -ldl -lpthread -lrt -Lcontrib/funchook/build -lfunchook -ldistorm -e __scope_main
INCLUDES=-I./contrib/libyaml/include -I./contrib/cJSON -I./os/$(OS) -I./contrib/pcre2/src -I./contrib/pcre2/build -I./contrib/jni -I./contrib/jni/linux/
YAML_DEFINES=-DYAML_VERSION_MAJOR="0" -DYAML_VERSION_MINOR="2" -DYAML_VERSION_PATCH="2" -DYAML_VERSION_STRING="\"0.2.2\""
YAML_SRC=$(wildcard contrib/libyaml/src/*.c)
YAML_AR=contrib/libyaml/src/.libs/libyaml.a
JSON_AR=contrib/cJSON/libcjson.a
FUNCHOOK_AR=contrib/funchook/build/libfunchook.a contrib/funchook/build/libdistorm.a
PCRE2_AR=contrib/pcre2/build/libpcre2-posix.a contrib/pcre2/build/libpcre2-8.a
TEST_AR=$(YAML_AR) $(JSON_AR) $(PCRE2_AR)
TEST_LIB=contrib/cmocka/build/src/libcmocka.dylib
TEST_INCLUDES=-I./src -I./contrib/cmocka/include
TEST_LD_FLAGS=-Lcontrib/cmocka/build/src -lcmocka -ldl

.PHONY: all clean test
all: libscope.so scope

$(PCRE2_AR):
	@echo "Building pcre2"
	cd contrib/pcre2 && mkdir -p build
	cd contrib/pcre2/build && cmake ..
	cd contrib/pcre2/build && make

$(FUNCHOOK_AR):
	@echo "Building funchook and distorm"
	cd contrib/funchook && mkdir -p build
	cd contrib/funchook/build && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd contrib/funchook/build && make distorm funchook-static

libscope.so: src/wrap.c src/state.c src/httpstate.c src/report.c src/plattime.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c src/sysexec.c src/gocontext.S src/wrap_go.c $(YAML_SRC) contrib/cJSON/cJSON.c src/javabci.c src/javaagent.c
	@echo "Building libscope.so ..."
	make $(FUNCHOOK_AR)
	make $(PCRE2_AR)
	$(CC) $(CFLAGS) -shared -fvisibility=hidden -fno-stack-protector -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) -o ./lib/$(OS)/$@ $(INCLUDES) $^ $(LD_FLAGS)
	$(CC) -c $(CFLAGS) -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) $(INCLUDES) $^
	rm -rf ./test/selfinterpose && mkdir ./test/selfinterpose && mv *.o ./test/selfinterpose/ && rm ./test/selfinterpose/wrap_go.o

scope: src/scope.c libscope.so
	objcopy -I binary -O elf64-x86-64 -B i386 ./lib/$(OS)/libscope.so ./lib/$(OS)/libscope.o
	gcc -Wall  -g src/scope.c -ldl -lrt -o ./bin/$(OS)/scope ./lib/$(OS)/libscope.o
	rm -f ./lib/$(OS)/libscope.o

clean:
	rm -f ./lib/$(OS)/libscope.so contrib/funchook/build/*.a bin/$(OS)/scope

SRC_C_FILES:=$(wildcard src/*.c)
SRC_C_FILES:=$(filter-out src/wrap.c, $(SRC_C_FILES))
SRC_C_FILES:=$(filter-out src/sysexec.c, $(SRC_C_FILES))
TEST_C_FILES:=$(wildcard test/*.c)
TEST_C_FILES:=$(filter-out test/wraptest.c, $(TEST_C_FILES))
C_FILES:=$(SRC_C_FILES) $(TEST_C_FILES) os/$(OS)/os.c
O_FILES:=$(C_FILES:.c=.o)

test: $(C_FILES)
	@echo "Building Tests"
	$(CC) -c $(TEST_CFLAGS) $^ $(INCLUDES) $(TEST_INCLUDES)
	make $(YAML_AR)
	make $(JSON_AR)
	make $(TEST_LIB)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgutilstest cfgutilstest.o cfgutils.o cfg.o mtc.o log.o evtformat.o ctl.o transport.o mtcformat.o com.o dbg.o circbuf.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgtest cfgtest.o cfg.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/transporttest transporttest.o transport.o dbg.o log.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/logtest logtest.o log.o transport.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtctest mtctest.o mtc.o log.o transport.o mtcformat.o com.o ctl.o evtformat.o cfg.o cfgutils.o dbg.o circbuf.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/evtformattest evtformattest.o evtformat.o log.o transport.o mtcformat.o dbg.o cfg.o com.o ctl.o mtc.o circbuf.o cfgutils.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/ctltest ctltest.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o com.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpstatetest httpstatetest.o httpstate.o plattime.o search.o os.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -lrt
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/reporttest reporttest.o report.o state.o httpstate.o com.o plattime.o os.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o search.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -lrt -Wl,--wrap=cmdSendEvent -Wl,--wrap=cmdSendMetric
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtcformattest mtcformattest.o mtcformat.o dbg.o log.o transport.o com.o ctl.o mtc.o evtformat.o cfg.o cfgutils.o linklist.o circbuf.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/circbuftest circbuftest.o circbuf.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/linklisttest linklisttest.o linklist.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/comtest comtest.o com.o ctl.o log.o transport.o evtformat.o circbuf.o mtcformat.o cfgutils.o cfg.o mtc.o dbg.o linklist.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dbgtest dbgtest.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/glibcvertest glibcvertest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/selfinterposetest selfinterposetest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dnstest dnstest.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/javabcitest javabcitest.o javabci.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/searchtest searchtest.o search.o dbg.o log.o transport.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) test/manual/passfd.c -lpthread -o test/$(OS)/passfd
	$(CC) $(TEST_CFLAGS) test/manual/unixpeer.c -lpthread -o test/$(OS)/unixpeer
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
