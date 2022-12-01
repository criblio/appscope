#
# AppScope - Contrib Builds
#
# The `contrib/` folder contains a number of external dependencies we keep
# locally to ensure things don't change under us. The logic here builds them.
# 
# To support caching the results, we build most of these into subdirectories of
# `./build/` and try not to make any changes/additions in the source folders
# themselves. In CI, we can hash the source folder to get a key for caching the
# results of the build if nothing actually changes.
#

ARCH ?= $(shell uname -m)

all: cJSON cmocka funchook libyaml openssl pcre2 ls-hpack musl libunwind
clean:
	-@$(MAKE) -C cJSON clean
	-@$(RM) -r build/*
.PHONY: all clean

cJSON: cJSON/libcjson.a
cmocka: build/cmocka/src/libcmocka.so
funchook: build/funchook/libfunchook.a
libyaml: build/libyaml/src/.libs/libyaml.a
libunwind: build/libunwind/src/.libs/libunwind.a
openssl: build/openssl/libssl.a build/openssl/libcrypto.a
pcre2: build/pcre2/libpcre2-posix.a build/pcre2/libpcre2-8.a
ls-hpack: build/ls-hpack/libls-hpack.a
musl: build/musl/lib/libc.a
.PHONY: cJSON cmocka funchook libyaml openssl pcre2 ls-hpack musl

# Pro Tip: add these to `cmake` commands to be able to debug the resulting libraries
#
#CMAKE_DEBUG_OPTS = -DCMAKE_BUILD_TYPE=DEBUG -DCMAKE_C_FLAGS_DEBUG="-g -O0" -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"

# Can't be cached because the build is in the source folder.
cJSON/libcjson.a:
	@echo "$${CI:+::group::}Building cJSON"
	cd cJSON && $(MAKE)
	objcopy --redefine-syms redefine_syms.lst cJSON/libcjson.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/cmocka/src/libcmocka.so:
	@echo "$${CI:+::group::}Building cmocka"
	@$(RM) -r build/cmocka
	@mkdir build/cmocka
	cd build/cmocka && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ../../cmocka && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/funchook/libfunchook.a:
	@echo "$${CI:+::group::}Building funchook"
	@$(RM) -r build/funchook
	@mkdir build/funchook
	if [ "x86_64" = "$(ARCH)" ]; then \
		cd build/funchook && cmake -DCMAKE_BUILD_TYPE=Release -DFUNCHOOK_DISASM=capstone ../../funchook; \
	elif [ "aarch64" = "$(ARCH)" ]; then \
		cd build/funchook && cmake -DCMAKE_BUILD_TYPE=Release ../../funchook; \
	else \
		echo >&2 "error: unsupported ARCH; \"$(ARCH)\""; \
		exit 1; \
	fi
	cd build/funchook && $(MAKE) capstone_src funchook-static
	objcopy --redefine-syms redefine_syms.lst build/funchook/libfunchook.a
	objcopy --redefine-syms redefine_syms.lst build/funchook/capstone_src-prefix/src/capstone_src-build/libcapstone.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

# Can't be cached because we have to `autoreconf` the source folder.
build/libyaml/src/.libs/libyaml.a:
	@echo "$${CI:+::group::}Building libyaml"
	@$(RM) -r build/libyaml
	@mkdir build/libyaml
	cd build/libyaml && autoreconf -fvi ../../libyaml && ../../libyaml/configure CFLAGS=-fPIC LIBS=-ldl && $(MAKE)
	objcopy --redefine-syms redefine_syms.lst build/libyaml/src/.libs/libyaml.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/openssl/libssl.a build/openssl/libcrypto.a:
	@echo "$${CI:+::group::}Building openssl"
	@$(RM) -r build/openssl
	@mkdir build/openssl
	cd build/openssl && ../../openssl/Configure no-engine no-async no-tests no-unit-test no-threads && make -j8
	# Yuck.  Avoids naming conflict between our wrap.c and libssl.a
	objcopy --redefine-sym SSL_read=SCOPE_SSL_read build/openssl/libssl.a
	objcopy --redefine-sym SSL_write=SCOPE_SSL_write build/openssl/libssl.a
	objcopy --redefine-syms redefine_syms.lst build/openssl/libcrypto.a
	objcopy --redefine-syms redefine_syms.lst build/openssl/libssl.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/pcre2/libpcre2-posix.a build/pcre2/libpcre2-8.a:
	@echo "$${CI:+::group::}Building pcre2"
	@$(RM) -r build/pcre2
	@mkdir build/pcre2
	cd build/pcre2 && cmake ../../pcre2 && $(MAKE)
	objcopy --redefine-syms redefine_syms.lst build/pcre2/libpcre2-posix.a
	objcopy --redefine-syms redefine_syms.lst build/pcre2/libpcre2-8.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/ls-hpack/libls-hpack.a:
	@echo "$${CI:+::group::}Building ls-hpack"
	@$(RM) -r build/ls-hpack
	@mkdir build/ls-hpack
	cd build/ls-hpack && cmake ../../ls-hpack -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release -DLSHPACK_XXH=ON && $(MAKE)
	objcopy --redefine-syms redefine_syms.lst build/ls-hpack/libls-hpack.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/libunwind/src/.libs/libunwind.a:
	@echo "$${CI:+::group::}Building libunwind"
	@$(RM) -r build/libunwind
	@mkdir build/libunwind
# To enable debug please add below --enable-debug
	cd build/libunwind && autoreconf -fvi ../../libunwind && ../../libunwind/configure --disable-pthread-api --disable-shared --disable-minidebuginfo --disable-documentation --disable-tests CFLAGS=-fPIC && $(MAKE)
	objcopy --redefine-syms redefine_syms.lst build/libunwind/src/.libs/libunwind.a
	if [ "x86_64" = "$(ARCH)" ]; then \
		objcopy --redefine-sym dl_iterate_phdr=SCOPE_DlIteratePhdr build/libunwind/src/.libs/libunwind.a; \
	fi
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/musl/lib/libc.a:
	@echo "$${CI:+::group::}Building musl"
	@$(RM) -r build/musl
	@mkdir build/musl
# To enable debug please add below --enable-debug --enable-optimize=no
	cd build/musl && ../../musl/configure --disable-shared --disable-wrapper CFLAGS=-ffixed-x28 && $(MAKE)
	cp build/musl/lib/libc.a build/musl/lib/libc_orig.a
	objcopy --prefix-symbols=scopelibc_ build/musl/lib/libc.a
# TODO verify if the code below be done via mfloat-abi=soft not sure if aarch64 support soft though
# https://developer.arm.com/documentation/dui0774/b/compiler-command-line-options/-mfloat-abi
# -mgeneral-regs-only maybe ? https://gcc.gnu.org/onlinedocs/gcc/AArch64-Options.html
	if [ "aarch64" = "$(ARCH)" ]; then \
		objcopy --redefine-syms aarch64_musl_syms.lst build/musl/lib/libc.a; \
	fi
	@[ -z "$(CI)" ] || echo "::endgroup::"
