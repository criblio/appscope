# Use __aarch64__ already defined by GCC
#ARCH_CFLAGS=-D__ARM64__
ARCH_CFLAGS=-D__FUNCHOOK__

ARCH_LD_FLAGS=-lcapstone
ARCH_BINARY=elf64-littleaarch64
ARCH_OBJ=$(ARCH)
FUNCHOOK_AR=contrib/funchook/build/libfunchook.a contrib/funchook/build/capstone_src-prefix/src/capstone_src-build/libcapstone.a
$(FUNCHOOK_AR):
	@echo "Building funchook and capstone"
	cd contrib/funchook && mkdir -p build
	cd contrib/funchook/build && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd contrib/funchook/build && make capstone_src funchook-static

LD_FLAGS=$(PCRE2_AR) -ldl -lpthread -lrt -lresolv -Lcontrib/funchook/build -lfunchook -Lcontrib/funchook/build/capstone_src-prefix/src/capstone_src-build -lcapstone
INCLUDES=-I./contrib/libyaml/include -I./contrib/cJSON -I./os/$(OS) -I./contrib/pcre2/src -I./contrib/pcre2/build -I./contrib/funchook/build/capstone_src-prefix/src/capstone_src/include -I./contrib/jni -I./contrib/jni/linux/ -I./contrib/openssl/include

$(LIBSCOPE): src/wrap.c src/state.c src/httpstate.c src/report.c src/httpagg.c src/plattime.c src/fn.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c src/scopeelf.c src/utils.c $(YAML_SRC) contrib/cJSON/cJSON.c src/javabci.c src/javaagent.c
	$(MAKE) $(FUNCHOOK_AR)
	$(MAKE) $(PCRE2_AR)
	$(MAKE) $(OPENSSL_AR)

	@echo "Building libscope.so ..."
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -shared -fvisibility=hidden -fno-stack-protector -DSCOPE_VER=\"$(SCOPE_VER)\" $(CJSON_DEFINES) $(YAML_DEFINES) -pthread -o $@ $(INCLUDES) $^ $(LD_FLAGS) ${OPENSSL_AR} -Wl,--version-script=libscope.map

	$(CC) -c $(CFLAGS) $(ARCH_CFLAGS) -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) $(INCLUDES) $^
	rm -rf ./test/selfinterpose && mkdir ./test/selfinterpose && mv *.o ./test/selfinterpose/
