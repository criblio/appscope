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

all: cJSON cmocka funchook libyaml openssl pcre2
clean:
	-@$(MAKE) -C cJSON clean
	-@$(RM) -r build/*
.PHONY: all clean

cJSON: cJSON/libcjson.a
cmocka: build/cmocka
funchook: build/funchook
libyaml: build/libyaml
openssl: build/openssl
pcre2: build/pcre2
.PHONY: cJSON cmocka funchook libyaml openssl pcre2

# Can't be cached because the build is in the source folder.
cJSON/libcjson.a:
	@echo "$${CI:+::group::}Building cJSON"
	cd cJSON && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/cmocka:
	@echo "$${CI:+::group::}Building cmocka"
	mkdir build/cmocka
	cd build/cmocka && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ../../cmocka && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/funchook:
	@echo "$${CI:+::group::}Building funchook"
	mkdir build/funchook
	if [ "x86_64" = "$(ARCH)" ]; then \
		cd build/funchook && cmake -DCMAKE_BUILD_TYPE=Release -DFUNCHOOK_DISASM=capstone ../../funchook; \
	elif [ "aarch64" = "$(ARCH)" ]; then \
		cd build/funchook && cmake -DCMAKE_BUILD_TYPE=Release ../../funchook; \
	else \
		echo >&2 "error: unsupported ARCH; \"$(ARCH)\""; \
		exit 1; \
	fi
	cd build/funchook && $(MAKE) capstone_src funchook-static
	@[ -z "$(CI)" ] || echo "::endgroup::"

# Can't be cached because we have to `autoreconf` the source folder.
build/libyaml:
	@echo "$${CI:+::group::}Building libyaml"
	mkdir build/libyaml
	cd build/libyaml && autoreconf -fvi ../../libyaml && ../../libyaml/configure LIBS=-ldl && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/openssl:
	@echo "$${CI:+::group::}Building openssl"
	mkdir build/openssl
	cd build/openssl && ../../openssl/Configure no-async && make -j8
	# Yuck.  Avoids naming conflict between our wrap.c and libssl.a
	objcopy --redefine-sym SSL_read=SCOPE_SSL_read build/openssl/libssl.a
	objcopy --redefine-sym SSL_write=SCOPE_SSL_write build/openssl/libssl.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

build/pcre2:
	@echo "$${CI:+::group::}Building pcre2"
	mkdir build/pcre2
	cd build/pcre2 && cmake ../../pcre2 && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

