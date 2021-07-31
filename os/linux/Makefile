# Note: Please see `gcc -dM -E - < /dev/null` for what the compiler already defines automatically before adding things.

CWD = $(shell pwd)
OS=linux
CC=gcc
CFLAGS=-fPIC -g -Wall -Wno-nonnull -Wno-deprecated-declarations -D__LINUX__
SCOPE_VER:="$(shell git --no-pager describe --abbrev=12 --dirty=+ --always --tags)"
TEST_CFLAGS=-g -Wall -Wno-nonnull -O0 -coverage -D__LINUX__
TEST_CFLAGS+=-DSCOPE_VER=\"$(SCOPE_VER)\"
YAML_DEFINES=-DYAML_VERSION_MAJOR="0" -DYAML_VERSION_MINOR="2" -DYAML_VERSION_PATCH="2" -DYAML_VERSION_STRING="\"0.2.2\""
CJSON_DEFINES=-DENABLE_LOCALES
YAML_SRC=$(wildcard contrib/libyaml/src/*.c)
YAML_AR=contrib/libyaml/src/.libs/libyaml.a
JSON_AR=contrib/cJSON/libcjson.a
PCRE2_AR=contrib/pcre2/build/libpcre2-posix.a contrib/pcre2/build/libpcre2-8.a
OPENSSL_AR=contrib/openssl/libssl.a contrib/openssl/libcrypto.a
TEST_AR=$(YAML_AR) $(JSON_AR) $(PCRE2_AR) ${OPENSSL_AR}
#TEST_LIB=contrib/cmocka/build/src/libcmocka.dylib
TEST_LIB=contrib/cmocka/build/src/libcmocka.so
TEST_INCLUDES=-I./src -I./contrib/cmocka/include
TEST_LD_FLAGS=-Lcontrib/cmocka/build/src -lcmocka -ldl -lresolv -lrt -lpthread

ARCH=$(shell uname -m)

LIBSCOPE=lib/$(OS)/$(ARCH)/libscope.so
LDSCOPE=bin/$(OS)/$(ARCH)/ldscope
LDSCOPEDYN=bin/$(OS)/$(ARCH)/ldscopedyn

coreall: $(LIBSCOPE) $(LDSCOPE) $(LDSCOPEDYN)

corerebuild:
	@$(RM) $(LIBSCOPE) $(LDSCOPE) $(LDSCOPEDYN)
	@$(MAKE) coreall

include os/linux/$(ARCH).mk

$(OPENSSL_AR):
	@echo "$${CI:+::group::}Building OpenSSL"
	cd contrib/openssl && ./Configure no-async
	cd contrib/openssl && make -j8
	# Yuck.  Avoids naming conflict between our wrap.c and libssl.a
	objcopy --redefine-sym SSL_read=SCOPE_SSL_read contrib/openssl/libssl.a
	objcopy --redefine-sym SSL_write=SCOPE_SSL_write contrib/openssl/libssl.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

$(PCRE2_AR):
	@echo "$${CI:+::group::}Building PCRE2"
	cd contrib/pcre2 && mkdir -p build
	cd contrib/pcre2/build && cmake ..
	cd contrib/pcre2/build && make
	@[ -z "$(CI)" ] || echo "::endgroup::"

$(LDSCOPEDYN): src/fn.c os/$(OS)/os.c src/dbg.c src/scopeelf.c src/scope.c src/utils.c src/inject.c
	@echo "$${CI:+::group::}Building $@"
	gcc -Wall -g -D__LINUX__ $(ARCH_CFLAGS) $(DISTRO_FLAGS) \
		-DSCOPE_VER=\"$(SCOPE_VER)\" \
		src/fn.c os/$(OS)/os.c src/dbg.c src/scopeelf.c src/scope.c src/utils.c src/inject.c \
		-ldl -lrt -o $@ -I./os/$(OS) $(INCLUDES)
	@[ -z "$(CI)" ] || echo "::endgroup::"

$(LDSCOPE): src/scope_static.c src/libdir.c $(LDSCOPEDYN) $(LIBSCOPE)
	@echo "$${CI:+::group::}Building $@"
	cd $(dir $(LDSCOPEDYN)) && objcopy \
		-I binary -O $(ARCH_BINARY) -B $(ARCH_OBJ) \
		$(notdir $(LDSCOPEDYN)) $(notdir $(LDSCOPEDYN)).o
	cd $(dir $(LIBSCOPE)) && objcopy \
		-I binary -O $(ARCH_BINARY) -B $(ARCH_OBJ) \
		$(notdir $(LIBSCOPE)) $(notdir $(LIBSCOPE:.so=.o))
	gcc -Wall -static -g -D__LINUX__ $(ARCH_CFLAGS) $(DISTRO_FLAGS) \
		-DSCOPE_VER=\"$(SCOPE_VER)\" \
		src/scope_static.c src/libdir.c \
		$(INCLUDES) \
		$(LDSCOPEDYN).o $(LIBSCOPE:.so=.o) \
		-lrt -lpthread -o $@
	$(RM) $(LDSCOPEDYN).o $(LIBSCOPE:.so=.o)
	@[ -z "$(CI)" ] || echo "::endgroup::"

coreclean:
	$(RM) $(LIBSCOPE) $(LDSCOPE) $(LDSCOPEDYN) contrib/funchook/build/*.a

SRC_C_FILES:=$(wildcard src/*.c)
SRC_C_FILES:=$(filter-out src/wrap.c, $(SRC_C_FILES))
SRC_C_FILES:=$(filter-out src/sysexec.c, $(SRC_C_FILES))
SRC_C_FILES:=$(filter-out src/wrap_go.c, $(SRC_C_FILES))
SRC_C_FILES:=$(filter-out src/scope_static.c, $(SRC_C_FILES))
ifeq (aarch64,$(ARCH))
SRC_C_FILES:=$(filter-out src/bashmem.c, $(SRC_C_FILES))
endif
TEST_C_FILES:=$(wildcard test/*.c)
TEST_C_FILES:=$(filter-out test/wraptest.c, $(TEST_C_FILES))
C_FILES:=$(SRC_C_FILES) $(TEST_C_FILES) os/$(OS)/os.c
O_FILES:=$(C_FILES:.c=.o)
CACHE_FILES:=$(shell find . -name CMakeCache.txt)

coretest: $(C_FILES) $(YAML_AR) $(JSON_AR) $(TEST_LIB)
	@echo "$${CI:+::group::}Building Tests"
	$(CC) -c $(TEST_CFLAGS) $(C_FILES) $(INCLUDES) $(TEST_INCLUDES)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgutilstest cfgutilstest.o cfgutils.o cfg.o mtc.o log.o evtformat.o ctl.o transport.o mtcformat.o com.o dbg.o circbuf.o linklist.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgtest cfgtest.o cfg.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/transporttest transporttest.o transport.o dbg.o log.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/logtest logtest.o log.o transport.o dbg.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtctest mtctest.o mtc.o log.o transport.o mtcformat.o com.o ctl.o evtformat.o cfg.o cfgutils.o dbg.o circbuf.o linklist.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/evtformattest evtformattest.o evtformat.o log.o transport.o mtcformat.o dbg.o cfg.o com.o ctl.o mtc.o circbuf.o cfgutils.o linklist.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/ctltest ctltest.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o com.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpstatetest httpstatetest.o httpstate.o plattime.o search.o fn.o utils.o os.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -lrt
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpheadertest httpheadertest.o report.o httpagg.o state.o com.o httpstate.o plattime.o fn.o utils.o os.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o search.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cmdSendHttp -Wl,--wrap=cmdPostEvent
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpaggtest httpaggtest.o httpagg.o fn.o utils.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/reporttest reporttest.o report.o httpagg.o state.o httpstate.o com.o plattime.o fn.o utils.o os.o ctl.o log.o transport.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o circbuf.o linklist.o search.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cmdSendEvent -Wl,--wrap=cmdSendMetric
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtcformattest mtcformattest.o mtcformat.o dbg.o log.o transport.o com.o ctl.o mtc.o evtformat.o cfg.o cfgutils.o linklist.o fn.o utils.o circbuf.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/circbuftest circbuftest.o circbuf.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/linklisttest linklisttest.o linklist.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/comtest comtest.o com.o ctl.o log.o transport.o evtformat.o circbuf.o mtcformat.o cfgutils.o cfg.o mtc.o dbg.o linklist.o fn.o utils.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dbgtest dbgtest.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/glibcvertest glibcvertest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/selfinterposetest selfinterposetest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dnstest dnstest.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/javabcitest javabcitest.o javabci.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/searchtest searchtest.o search.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) test/manual/passfd.c -lpthread -o test/$(OS)/passfd
	$(CC) $(TEST_CFLAGS) test/manual/unixpeer.c -lpthread -o test/$(OS)/unixpeer
	@[ -z "$(CI)" ] || echo "::endgroup::"
	@echo "$${CI:+::group::}Running Tests and Generating Test Coverage"
	test/execute.sh
	@[ -z "$(CI)" ] || echo "::endgroup::"
# see file:///Users/cribl/scope/coverage/index.html

$(YAML_AR): $(YAML_SRC)
	@echo "$${CI:+::group::}Building libyaml"
	cd ./contrib/libyaml && ./bootstrap
	cd ./contrib/libyaml && ./configure LIBS=-ldl
	cd ./contrib/libyaml && $(MAKE)
# don't install on the host.  We only need the static .a file
	@[ -z "$(CI)" ] || echo "::endgroup::"

$(JSON_AR): contrib/cJSON/cJSON.c
	@echo "$${CI:+::group::}Building cJSON"
	cd ./contrib/cJSON && $(MAKE) clean
	cd ./contrib/cJSON && $(MAKE) all
	@[ -z "$(CI)" ] || echo "::endgroup::"

$(TEST_LIB):
	@echo "$${CI:+::group::}Building cmocka"
	cd contrib/cmocka && test -d ./build || mkdir ./build
	cd contrib/cmocka/build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ..
	cd contrib/cmocka/build && $(MAKE)
# don't install on the host.
# We can tell the linker where to find the .dylib file
	@[ -z "$(CI)" ] || echo "::endgroup::"

.PHONY: coreall coreclean coretest corerebuild
