# Use __aarch64__ already defined by GCC
#ARCH_CFLAGS=-D__ARM64__

ARCH_LD_FLAGS=
ARCH_BINARY=elf64-littleaarch64
ARCH_OBJ=$(ARCH)
FUNCHOOK_AR=contrib/funchook/build/libfunchook.a
$(FUNCHOOK_AR):
	@echo "Building funchook and distorm"
	cd contrib/funchook && mkdir -p build
	cd contrib/funchook/build && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd contrib/funchook/build && make capstone_src funchook-static

LD_FLAGS=$(PCRE2_AR) -ldl -lpthread -lrt -lresolv -Lcontrib/funchook/build -lfunchook
INCLUDES=-I./contrib/libyaml/include -I./contrib/cJSON -I./os/$(OS) -I./contrib/pcre2/src -I./contrib/pcre2/build -I./contrib/jni -I./contrib/jni/linux/ -I./contrib/openssl/include

$(LIBSCOPE): src/wrap.c src/state.c src/httpstate.c src/report.c src/httpagg.c src/plattime.c src/fn.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c src/scopeelf.c src/utils.c $(YAML_SRC) contrib/cJSON/cJSON.c src/javabci.c src/javaagent.c
	@echo "Building libscope.so ..."
	make $(FUNCHOOK_AR)
	make $(PCRE2_AR)
	make $(OPENSSL_AR)
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) -shared -fvisibility=hidden -fno-stack-protector -DSCOPE_VER=\"$(SCOPE_VER)\" $(CJSON_DEFINES) $(YAML_DEFINES) -o $@ $(INCLUDES) $^ $(LD_FLAGS) ${OPENSSL_AR} -Wl,--version-script=libscope.map
	$(CC) -c $(CFLAGS) $(ARCH_CFLAGS) -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) $(INCLUDES) $^
	rm -rf ./test/selfinterpose && mkdir ./test/selfinterpose && mv *.o ./test/selfinterpose/
