# Note: Please see `gcc -dM -E - < /dev/null` for what the compiler already defines automatically before adding things.

########## Definitions ##########
CWD = $(shell pwd)
OS=linux
CC=gcc
SCOPE_VER:="$(shell git --no-pager describe --abbrev=12 --dirty=+ --always --tags)"
LIBRARY_CFLAGS=-fPIC -g -Wall -Wno-nonnull -Wno-deprecated-declarations -Werror=implicit-function-declaration -Werror=override-init -I contrib/ls-hpack $(if $(DEBUG),-DDEBUG) -DSCOPE_VER=\"$(SCOPE_VER)\"
LOADER_CFLAGS=-fPIC -g -Wall -Wno-nonnull -Wno-deprecated-declarations -Werror=implicit-function-declaration -Werror=override-init -Wno-format-security -Wno-format-truncation -I contrib/ls-hpack $(if $(DEBUG),-DDEBUG) -DSCOPE_VER=\"$(SCOPE_VER)\" 
TEST_CFLAGS=-g -Wall -Wno-nonnull -O0 -coverage -Wno-format-security -Wno-format-truncation -DSCOPE_VER=\"$(SCOPE_VER)\"
YAML_DEFINES=-DYAML_VERSION_MAJOR="0" -DYAML_VERSION_MINOR="2" -DYAML_VERSION_PATCH="2" -DYAML_VERSION_STRING="\"0.2.2\""
CJSON_DEFINES=-DENABLE_LOCALES
YAML_SRC=$(wildcard contrib/libyaml/src/*.c)
YAML_AR=contrib/build/libyaml/src/.libs/libyaml.a
JSON_AR=contrib/cJSON/libcjson.a
PCRE2_AR=contrib/build/pcre2/libpcre2-posix.a contrib/build/pcre2/libpcre2-8.a
OPENSSL_AR=contrib/build/openssl/libssl.a contrib/build/openssl/libcrypto.a
LS_HPACK_AR=contrib/build/ls-hpack/libls-hpack.a
MUSL_AR=contrib/build/musl/lib/libc.a
UNWIND_AR=contrib/build/libunwind/src/.libs/libunwind.a
COREDUMPER_AR=contrib/build/coredumper/.libs/libcoredumper.a
TEST_AR=$(MUSL_AR) ${COREDUMPER_AR} ${UNWIND_AR} $(YAML_AR) $(JSON_AR) $(PCRE2_AR) ${OPENSSL_AR} $(LS_HPACK_AR)
#TEST_LIB=contrib/build/cmocka/src/libcmocka.dylib
TEST_LIB=contrib/build/cmocka/src/libcmocka.so
TEST_LD_FLAGS=-Lcontrib/build/cmocka/src -lcmocka -ldl -lresolv -lrt -lpthread
LIBRARY_INCLUDES=-I./src
LOADER_INCLUDES=-I./src/loader
CMOCKA_INCLUDES=-I./contrib/cmocka/include -I contrib/ls-hpack
LIBRARY_C_FILES:=$(wildcard src/*.c)
LIBRARY_C_FILES:=$(filter-out src/wrap.c, $(LIBRARY_C_FILES))
LIBRARY_C_FILES:=$(filter-out src/sysexec.c, $(LIBRARY_C_FILES))
LIBRARY_C_FILES:=$(filter-out src/wrap_go.c, $(LIBRARY_C_FILES))
LOADER_C_FILES:=$(wildcard src/loader/*.c)
LIBRARY_TEST_C_FILES:=$(wildcard test/unit/library/*.c)
LIBRARY_TEST_C_FILES:=$(filter-out test/unit/library/wraptest.c, $(LIBRARY_TEST_C_FILES))
LOADER_TEST_C_FILES:=$(wildcard test/unit/loader/*.c)
LOADER_TEST_C_FILES:=$(filter-out test/unit/loader/wraptest.c, $(LOADER_TEST_C_FILES))
OS_C_FILES:=os/$(OS)/os.c
ARCH=$(shell uname -m)
LIBSCOPE=lib/$(OS)/$(ARCH)/libscope.so
LIBLOADER=lib/$(OS)/$(ARCH)/libloader.a
SCOPEDYN=bin/$(OS)/$(ARCH)/scopedyn

include os/linux/$(ARCH).mk


########## Contrib ##########
$(MUSL_AR):
	@$(MAKE) -C contrib musl

$(UNWIND_AR):
	@$(MAKE) -C contrib libunwind

${COREDUMPER_AR}:
	@$(MAKE) -C contrib coredumper

$(OPENSSL_AR):
	@$(MAKE) -C contrib openssl

$(PCRE2_AR):
	@$(MAKE) -C contrib pcre2

$(YAML_AR): $(YAML_SRC)
	@$(MAKE) -C contrib libyaml

$(JSON_AR): contrib/cJSON/cJSON.c
	@$(MAKE) -C contrib cJSON

$(TEST_LIB):
	@$(MAKE) -C contrib cmocka


########## Core ##########
coreall: $(LIBSCOPE) $(LIBLOADER) $(SCOPEDYN)

corerebuild:
	@$(RM) $(LIBSCOPE) $(LIBLOADER) $(SCOPEDYN)
	@$(MAKE) coreall

coreclean:
	$(RM) $(LIBSCOPE) $(LIBLOADER) $(SCOPEDYN)
	$(RM) test/linux/*test

$(LIBLOADER): src/loader/loader.c src/loader/nsinfo.c src/loader/nsfile.c src/loader/loaderutils.c src/loader/inject.c src/loader/libdir.c src/loader/ns.c src/loader/setup.c src/loader/patch.c src/loader/libver.c
	@echo "$${CI:+::group::}Building $@"
	$(CC) -c $(LOADER_CFLAGS) $(YAML_DEFINES) -I./src $(INCLUDES) $^
	ar rcs $@ *.o
	$(RM) *.o
	@[ -z "$(CI)" ] || echo "::endgroup::"

$(SCOPEDYN): src/loader/scopedyn.c src/loader/loaderutils.c src/loader/libdir.c src/loader/libver.c src/loader/ns.c src/loader/nsinfo.c src/loader/setup.c src/loader/patch.c src/loader/nsfile.c
	@echo "$${CI:+::group::}Building $@"
	gcc -Wall -g $(LOADER_CFLAGS) \
	src/loader/scopedyn.c src/loader/loaderutils.c src/loader/libdir.c src/loader/libver.c src/loader/ns.c src/loader/nsinfo.c src/loader/setup.c src/loader/patch.c src/loader/nsfile.c \
	-ldl -lrt -o $@ -I./os/$(OS) $(INCLUDES)
	@[ -z "$(CI)" ] || echo "::endgroup::"

########## Tests ##########
libtest: $(LIBRARY_C_FILES) $(LIBRARY_TEST_C_FILES) $(YAML_AR) $(JSON_AR) $(TEST_LIB)
	@echo "$${CI:+::group::}Building Library Tests"
	$(CC) -c $(TEST_CFLAGS) $(LIBRARY_C_FILES) $(LIBRARY_INCLUDES) $(LIBRARY_TEST_C_FILES) $(INCLUDES) $(CMOCKA_INCLUDES) $(OS_C_FILES)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/ipctest ipctest.o ipc.o ipc_resp.o cfgutils.o cfg.o mtc.o log.o evtformat.o ctl.o transport.o backoff.o mtcformat.o strset.o com.o scopestdlib.o dbg.o circbuf.o linklist.o fn.o utils.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o scopecoredump.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=jsonConfigurationObject -Wl,--wrap=doAndReplaceConfig
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/vdsotest vdsotest.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/strsettest strsettest.o strset.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgutilstest cfgutilstest.o cfgutils.o cfg.o mtc.o log.o evtformat.o ctl.o transport.o backoff.o mtcformat.o strset.o com.o scopestdlib.o dbg.o circbuf.o linklist.o fn.o utils.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/cfgtest cfgtest.o cfg.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/transporttest transporttest.o transport.o backoff.o scopestdlib.o dbg.o log.o fn.o utils.o plattime.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/backofftest backofftest.o backoff.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/logtest logtest.o log.o transport.o backoff.o scopestdlib.o dbg.o fn.o utils.o plattime.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/utilstest utilstest.o scopestdlib.o dbg.o fn.o utils.o plattime.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtctest mtctest.o mtc.o log.o transport.o backoff.o mtcformat.o strset.o com.o ctl.o evtformat.o cfg.o cfgutils.o scopestdlib.o dbg.o circbuf.o linklist.o fn.o utils.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cfgLogStreamEnable
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/evtformattest evtformattest.o evtformat.o log.o transport.o backoff.o mtcformat.o strset.o scopestdlib.o dbg.o cfg.o com.o ctl.o mtc.o circbuf.o cfgutils.o linklist.o fn.o utils.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/ctltest ctltest.o ctl.o log.o transport.o backoff.o scopestdlib.o dbg.o cfgutils.o cfg.o com.o mtc.o evtformat.o mtcformat.o strset.o circbuf.o linklist.o fn.o utils.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cbufGet
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpstatetest httpstatetest.o httpstate.o plattime.o search.o fn.o utils.o os.o scopestdlib.o dbg.o test.o com.o cfg.o cfgutils.o mtc.o mtcformat.o strset.o ctl.o transport.o backoff.o linklist.o log.o evtformat.o circbuf.o state.o metriccapture.o report.o httpagg.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cmdPostEvent -lrt
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpheadertest httpheadertest.o report.o httpagg.o state.o com.o httpstate.o metriccapture.o plattime.o fn.o utils.o os.o ctl.o log.o transport.o backoff.o scopestdlib.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o strset.o circbuf.o linklist.o search.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cmdSendHttp -Wl,--wrap=cmdPostEvent
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/httpaggtest httpaggtest.o httpagg.o fn.o utils.o scopestdlib.o dbg.o plattime.o os.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/reporttest reporttest.o report.o httpagg.o state.o httpstate.o metriccapture.o com.o plattime.o fn.o utils.o os.o ctl.o log.o transport.o backoff.o scopestdlib.o dbg.o cfgutils.o cfg.o mtc.o evtformat.o mtcformat.o strset.o circbuf.o linklist.o search.o test.o $(TEST_AR) $(TEST_LD_FLAGS) -Wl,--wrap=cmdSendEvent -Wl,--wrap=cmdSendMetric
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/mtcformattest mtcformattest.o mtcformat.o strset.o scopestdlib.o dbg.o log.o transport.o backoff.o com.o ctl.o mtc.o evtformat.o cfg.o cfgutils.o linklist.o fn.o utils.o circbuf.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/circbuftest circbuftest.o circbuf.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/linklisttest linklisttest.o linklist.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/comtest comtest.o com.o ctl.o log.o transport.o backoff.o evtformat.o circbuf.o mtcformat.o strset.o cfgutils.o cfg.o mtc.o scopestdlib.o dbg.o linklist.o fn.o utils.o os.o test.o report.o search.o httpagg.o state.o httpstate.o metriccapture.o plattime.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dbgtest dbgtest.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/glibcvertest glibcvertest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/selfinterposetest selfinterposetest.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/dnstest dnstest.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/javabcitest javabcitest.o javabci.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/searchtest searchtest.o search.o scopestdlib.o dbg.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) test/manual/passfd.c -lpthread -o test/$(OS)/passfdtest
	$(CC) $(TEST_CFLAGS) test/manual/unixpeer.c -lpthread -o test/$(OS)/unixpeertest
	@[ -z "$(CI)" ] || echo "::endgroup::"

loadertest: $(LOADER_C_FILES) $(LOADER_TEST_C_FILES) $(YAML_AR) $(JSON_AR) $(TEST_LIB)
	@echo "$${CI:+::group::}Building Loader Tests"
	$(CC) -c $(TEST_CFLAGS) $(LOADER_C_FILES) $(LOADER_INCLUDES) $(LOADER_TEST_C_FILES) $(INCLUDES) $(CMOCKA_INCLUDES) 
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/libvertest libvertest.o libver.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/libdirtest libdirtest.o libdir.o nsfile.o libver.o loaderutils.o loader.o patch.o nsinfo.o ns.o inject.o setup.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/setuptest setuptest.o setup.o patch.o libdir.o nsfile.o libver.o loaderutils.o loader.o nsinfo.o ns.o inject.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	$(CC) $(TEST_CFLAGS) -o test/$(OS)/nsinfotest nsinfotest.o nsinfo.o test.o $(TEST_AR) $(TEST_LD_FLAGS)
	@[ -z "$(CI)" ] || echo "::endgroup::"

# ensure $USER is available since some of the tests expect it
runtests: export USER ?= $(shell id -u -n)
runtests:
	@echo "$${CI:+::group::}Running Tests and Generating Test Coverage"
	test/unit/execute.sh
	@[ -z "$(CI)" ] || echo "::endgroup::"
	# see file:///Users/cribl/scope/coverage/index.html


.PHONY: coreall coreclean libtest loadertest runtests corerebuild
