ARCH_CFLAGS=-D__GO__ -D__FUNCHOOK__ -D__x86_64__
ARCH_LD_FLAGS=-ldistorm 
FUNCHOOK_AR=contrib/funchook/build/libfunchook.a contrib/funchook/build/libdistorm.a

$(FUNCHOOK_AR):
	@echo "Building funchook and distorm"
	cd contrib/funchook && mkdir -p build
	cd contrib/funchook/build && cmake -DCMAKE_BUILD_TYPE=Release ..
	cd contrib/funchook/build && make distorm funchook-static

LIBSCOPE_SRC=src/wrap.c src/state.c src/httpstate.c src/report.c src/httpagg.c src/plattime.c src/fn.c os/$(OS)/os.c src/cfgutils.c src/cfg.c src/transport.c src/log.c src/mtc.c src/circbuf.c src/linklist.c src/evtformat.c src/ctl.c src/mtcformat.c src/com.c src/dbg.c src/search.c src/sysexec.c src/gocontext.S src/scopeelf.c src/wrap_go.c src/utils.c $(YAML_SRC) contrib/cJSON/cJSON.c src/javabci.c src/javaagent.c
ARCH_RM=&& rm ./test/selfinterpose/wrap_go.o
ARCH_COPY=objcopy -I binary -O elf64-x86-64 -B i386 ./lib/$(OS)/libscope.so ./lib/$(OS)/libscope.o
