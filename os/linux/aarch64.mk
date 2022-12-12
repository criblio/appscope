ARCH_LD_FLAGS=-lcapstone
ARCH_BINARY=elf64-littleaarch64
ARCH_OBJ=$(ARCH)

LD_FLAGS=$(MUSL_AR) $(UNWIND_AR) $(PCRE2_AR) $(LS_HPACK_AR) $(YAML_AR) $(JSON_AR) -ldl -lpthread -lrt -lresolv -Lcontrib/build/funchook -lfunchook -Lcontrib/build/funchook/capstone_src-prefix/src/capstone_src-build -lcapstone -z noexecstack
INCLUDES=-I./contrib/libyaml/include -I./contrib/cJSON -I./os/$(OS) -I./contrib/pcre2/src -I./contrib/build/pcre2 -I./contrib/funchook/capstone_src/include/ -I./contrib/jni -I./contrib/jni/linux/ -I./contrib/openssl/include -I./contrib/build/openssl/include -I./contrib/build/libunwind/include  -I./contrib/libunwind/include/

$(LIBSCOPE): src/wrap.c src/state.c src/httpstate.c src/metriccapture.c src/report.c src/httpagg.c src/plattime.c src/fn.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/scopestdlib.c src/dbg.c src/search.c src/scopeelf.c src/utils.c src/strset.c src/javabci.c src/javaagent.c src/backoff.c
	@$(MAKE) -C contrib funchook pcre2 openssl ls-hpack musl libyaml libunwind cJSON
	@echo "$${CI:+::group::}Building $@"
	$(CC) $(CFLAGS) \
		-shared -fvisibility=hidden -fno-stack-protector \
		-DSCOPE_VER=\"$(SCOPE_VER)\" $(CJSON_DEFINES) $(YAML_DEFINES) \
		-pthread -o $@ $(INCLUDES) $^ $(LD_FLAGS) ${OPENSSL_AR} \
		-Wl,--version-script=libscope.map
	$(CC) -c $(CFLAGS) -DSCOPE_VER=\"$(SCOPE_VER)\" $(YAML_DEFINES) $(INCLUDES) $^
	$(RM) -r ./test/selfinterpose && \
		mkdir ./test/selfinterpose && \
		mv *.o ./test/selfinterpose/
	@[ -z "$(CI)" ] || echo "::endgroup::"
