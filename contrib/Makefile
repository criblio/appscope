#
# Cribl AppScope - Build Contribs
#

# --
# Notes:
#
#   We're not using Automake here since all we need is to relay through to the
#   build system provided by each of our contribs. We need to support the usual
#   targets including `all`, `check`, `clear, etc.
#
#   The *_LIBS variables below are the .a's we need to link against elsewhere 
#   but the lists are truncated so they only include the first one. This is to
#   avoid the job running twice when we use `-j` to build or test in parallel.
#
#   We separate out the "configure" logic from the step to actually run make
#   for each library so we can run parallel builds.
# --

CMAKE ?= $(shell which cmake)
OBJCOPY ?= $(shell which objcopy)

all: cJSON cmocka funchook libyaml openssl pcre2
.PHONY: all

CJSON_LIBS = cJSON/libcjson.so
cJSON: $(CJSON_LIBS)
$(CJSON_LIBS): cJSON/build
	$(MAKE) -C cJSON/build
cJSON/build:
	mkdir $@
	cd $@ && $(CMAKE) -DENABLE_CUSTOM_COMPILER_FLAGS=OFF -DENABLE_CJSON_TEST=OFF -DCMAKE_BUILD_TYPE=Release ..
.PHONY: cJSON

FUNCHOOK_LIBS = funchook/build/libfunchook.a #funchook/build/libdistorm.a
funchook: $(FUNCHOOK_LIBS)
$(FUNCHOOK_LIBS): funchook/build
	$(MAKE) -C funchook/build
funchook/build:
	mkdir $@
	cd $@ && $(CMAKE) -DCMAKE_BUILD_TYPE=Release ..
.PHONY: funchook

PCRE2_LIBS = pcre2/build/libpcre2-8.a #pcre2/build/libpcre2-posix.a
pcre2: $(PCRE2_LIBS)
$(PCRE2_LIBS): pcre2/build
	$(MAKE) -C pcre2/build
pcre2/build:
	mkdir $@
	cd $@ && $(CMAKE) ..
.PHONY: pcre2

YAML_LIBS = libyaml/src/libyaml.la
libyaml: $(YAML_LIBS)
$(YAML_LIBS): libyaml/Makefile
	$(MAKE) -C libyaml
libyaml/Makefile:
	cd libyaml && \
		./bootstrap && \
		./configure LIBS=-ldl --disable-shared --with-pic CFLAGS="-fPIC -DPIC"
.PHONY: libyaml

CMOCKA_LIBS = Gcmocka/build/src/libcmocka.so
cmocka: $(CMOCKA_LIBS)
$(CMOCKA_LIBS): cmocka/build
	$(MAKE) -C cmocka/build
cmocka/build:
	mkdir $@
	cd $@ && $(CMAKE) -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug ..
.PHONY: cmocka

#
# Yuck. The objcopy's avoid naming conflicts between our wrap.c and libssl.a.
#
OPENSSL_LIBS = openssl/libssl.a #openssl/libcrypto.a
openssl: $(OPENSSL_LIBS)
$(OPENSSL_LIBS): openssl/Makefile
	$(MAKE) -C openssl
	$(OBJCOPY) --redefine-sym SSL_read=SCOPE_SSL_read openssl/libssl.a
	$(OBJCOPY) --redefine-sym SSL_write=SCOPE_SSL_write openssl/libssl.a
openssl/Makefile:
	cd openssl && ./Configure no-async
.PHONY: openssl

check: all
.PHONY: check

mostlyclean distclean maintainer-clean: clean
	$(RM) libyaml/Makefile
	$(RM) openssl/Makefile
clean:
	$(RM) -r cJSON/build
	$(RM) -r funchook/build
	$(RM) -r pcre2/build
	cd libyaml && $(MAKE) clean
	$(RM) -r cmocka/build
	cd openssl && $(MAKE) clean
.PHONY: mostlyclean distclean maintainer-clean clean
