ARCH_CFLAGS=-D__GO__ -D__FUNCHOOK__ -D__x86_64__
ARCH_LD_FLAGS=-lcapstone
ARCH_BINARY=elf64-x86-64
ARCH_OBJ=i386
FUNCHOOK_AR=contrib/funchook/build/libfunchook.a contrib/funchook/build/capstone_src-prefix/src/capstone_src-build/libcapstone.a

$(FUNCHOOK_AR):
	@echo "$${CI:+::group::}Building funchook and capstone"
	cd contrib/funchook && mkdir -p build
	cd contrib/funchook/build && cmake -DCMAKE_BUILD_TYPE=Release -DFUNCHOOK_DISASM=capstone ..
	cd contrib/funchook/build && make capstone_src funchook-static
	@[ -z "$(CI)" ] || echo "::endgroup::"

LD_FLAGS=$(PCRE2_AR) -ldl -lpthread -lrt -Lcontrib/funchook/build -lfunchook -Lcontrib/funchook/build/capstone_src-prefix/src/capstone_src-build -lcapstone
INCLUDES=-I./contrib/libyaml/include -I./contrib/cJSON -I./os/$(OS) -I./contrib/pcre2/src -I./contrib/pcre2/build -I./contrib/funchook/build/capstone_src-prefix/src/capstone_src/include -I./contrib/jni -I./contrib/jni/linux/ -I./contrib/openssl/include

#ARCH_RM=&& rm ./test/selfinterpose/wrap_go.o
#ARCH_COPY=cp ./lib/$(OS)/$(ARCH)/libscope.so ./lib/$(OS)/libscope.so && \
#	objcopy -I binary -O elf64-x86-64 -B i386 ./lib/$(OS)/libscope.so ./lib/$(OS)/libscope.o && \
#	rm -f ./lib/$(OS)/libscope.so

$(LIBSCOPE): src/wrap.c src/state.c src/httpstate.c src/report.c src/httpagg.c src/plattime.c src/fn.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c src/sysexec.c src/gocontext.S src/scopeelf.c src/wrap_go.c src/utils.c src/bashmem.c $(YAML_SRC) contrib/cJSON/cJSON.c src/javabci.c src/javaagent.c
	make $(FUNCHOOK_AR)
	make $(PCRE2_AR)
	make $(OPENSSL_AR)
	@echo "$${CI:+::group::}Building $@"
	$(CC) $(CFLAGS) $(ARCH_CFLAGS) \
		-shared -fvisibility=hidden -fno-stack-protector \
		-DSCOPE_VER=\"$(SCOPE_VER)\" $(CJSON_DEFINES) $(YAML_DEFINES) \
		-o $@ $(INCLUDES) $^ $(LD_FLAGS) ${OPENSSL_AR} \
		-Wl,--version-script=libscope.map
	$(CC) -c $(CFLAGS) $(ARCH_CFLAGS) -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) $(INCLUDES) $^
	$(RM) -r ./test/selfinterpose && \
		mkdir ./test/selfinterpose && \
		mv *.o ./test/selfinterpose/ && \
		$(RM) ./test/selfinterpose/wrap_go.o
	@[ -z "$(CI)" ] || echo "::endgroup::"
