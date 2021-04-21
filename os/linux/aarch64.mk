ARCH_CFLAGS=
ARCH_LD_FLAGS=
FUNCHOOK_AR=contrib/funchook/build/libfunchook.a
$(FUNCHOOK_AR):
	@echo "Building funchook and distorm"
	cd contrib/funchook && mkdir -p build
	cd contrib/funchook/build && cmake -DCMAKE_BUILD_TYPE=Release ..

SCOPE_SRC=src/wrap.c src/state.c src/httpstate.c src/report.c src/httpagg.c src/plattime.c src/fn.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c src/scopeelf.c src/utils.c $(YAML_SRC) contrib/cJSON/cJSON.c src/javabci.c src/javaagent.c
ARCH_RM=
ARCH_COPY=objcopy -I binary -O elf64-littleaarch64 -B aarch64 ./lib/$(OS)/libscope.so ./lib/$(OS)/libscope.o
