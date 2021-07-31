ARCH ?= $(shell uname -m)

all: cJSON cmocka funchook libyaml openssl pcre2
clean:
	-@$(MAKE) -C cJSON clean
	-@$(RM) -r cmocka/build
	-@$(RM) -r funchook/build
	-@$(RM) -r libyaml/build
	-@$(RM) -r openssl/build
	-@$(RM) -r pcre2/build
.PHONY: all clean

cJSON: cJSON/libcjson.a
cmocka: cmocka/build
funchook: funchook/build
libyaml: libyaml/build
openssl: openssl/build
pcre2: pcre2/build
.PHONY: cJSON cmocka funchook libyaml openssl pcre2

cJSON/libcjson.a:
	@echo "$${CI:+::group::}Building cJSON"
	cd cJSON && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

cmocka/build:
	@echo "$${CI:+::group::}Building cmocka"
	mkdir cmocka/build
	cd cmocka/build && cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug .. && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

funchook/build:
	@echo "$${CI:+::group::}Building funchook"
	mkdir funchook/build
	if [ "x86_64" = "$(ARCH)" ]; then \
		cd funchook/build && cmake -DCMAKE_BUILD_TYPE=Release -DFUNCHOOK_DISASM=capstone ..; \
	elif [ "aarch64" = "$(ARCH)" ]; then \
		cd funchook/build && cmake -DCMAKE_BUILD_TYPE=Release ..; \
	else \
		echo >&2 "error: unsupported ARCH; \"$(ARCH)\""; \
		exit 1; \
	fi
	cd funchook/build && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

libyaml/build:
	@echo "$${CI:+::group::}Building libyaml"
	mkdir libyaml/build
	cd libyaml/build && autoreconf -fvi .. && ../configure LIBS=-ldl && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

openssl/build:
	@echo "$${CI:+::group::}Building openssl"
	mkdir openssl/build
	cd openssl/build && ../Configure no-async && make -j8
	# Yuck.  Avoids naming conflict between our wrap.c and libssl.a
	objcopy --redefine-sym SSL_read=SCOPE_SSL_read openssl/build/libssl.a
	objcopy --redefine-sym SSL_write=SCOPE_SSL_write openssl/build/libssl.a
	@[ -z "$(CI)" ] || echo "::endgroup::"

pcre2/build:
	@echo "$${CI:+::group::}Building pcre2"
	mkdir pcre2/build
	cd pcre2/build && cmake .. && $(MAKE)
	@[ -z "$(CI)" ] || echo "::endgroup::"

