#
# Cribl AppScope - Build Contribs
#

CMAKE ?= $(shell which cmake)
OBJCOPY ?= $(shell which objcopy)

CJSON_LIBS = cJSON/libcjson.so
FUNCHOOK_LIBS = funchook/build/libfunchook.a funchook/build/libdistorm.a
PCRE2_LIBS = pcre2/build/libpcre2-8.a pcre2/build/libpcre2-posix.a
YAML_LIBS = libyaml/src/libyaml.la
CMOCKA_LIBS = cmocka/build/src/libcmocka.dylib
OPENSSL_LIBS = openssl/libssl.a openssl/libcrypto.a

LIBS = $(CJSON_LIBS) $(FUNCHOOK_LIBS) $(PCRE2_LIBS) $(YAML_LIBS) $(CMOCKA_LIBS) $(OPENSSL_LIBS)

all: $(LIBS)

$(CJSON_LIBS):
	cd cJSON && $(MAKE)

$(FUNCHOOK_LIBS):
	@[ -d funchook/build ] || mkdir funchook/build
	cd funchook/build && \
		$(CMAKE) -DCMAKE_BUILD_TYPE=Release .. && \
		$(MAKE)

$(PCRE2_LIBS):
	@[ -d pcre2/build ] || mkdir pcre2/build
	cd pcre2/build && \
		$(CMAKE) .. && \
		$(MAKE)

$(YAML_LIBS):
	cd libyaml && \
		./bootstrap && \
		./configure LIBS=-ldl --disable-shared --with-pic CFLAGS="-fPIC -DPIC" && \
		$(MAKE)

$(CMOCKA_LIBS):
	@[ -d cmocka/build ] || mkdir cmocka/build
	cd cmocka/build && \
		$(CMAKE) -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Debug .. && \
		$(MAKE)

# Yuck.  The objcopy's avoid naming conflicts between our wrap.c and libssl.a
$(OPENSSL_LIBS):
	cd openssl && ./Configure && $(MAKE)
	$(OBJCOPY) --redefine-sym SSL_read=SCOPE_SSL_read openssl/libssl.a
	$(OBJCOPY) --redefine-sym SSL_write=SCOPE_SSL_write openssl/libssl.a

check:

mostlyclean distclean maintainer-clean: clean
clean:
	cd cJSON && $(MAKE) clean
	$(RM) -r funchook/build
	$(RM) -r pcre2/build
	cd libyaml && $(MAKE) clean
	$(RM) -r cmocka/build
	cd openssl && $(MAKE) clean

