#
# Cribl AppScope™ - Automake Input
#
# See docs/BUILD.md for details.
#

# -----------------------------------------------------------------------------
# Targets
#
lib_LTLIBRARIES = libscope.la
bin_PROGRAMS = ldscope

# -----------------------------------------------------------------------------
# Details for the ../contrib Packages
#
CJSON_DIR = ../contrib/cJSON
CJSON_DEFINES = -DENABLE_LOCALES

FUNCHOOK_DIR = ../contrib/funchook

JNI_DIR = ../contrib/jni

PCRE2_DIR = ../contrib/pcre2

YAML_DIR = ../contrib/libyaml
YAML_DEFINES = -DYAML_VERSION_MAJOR=0 \
	       -DYAML_VERSION_MINOR=2 \
	       -DYAML_VERSION_PATCH=2 \
	       -DYAML_VERSION_STRING=\"0.2.2\"

# -----------------------------------------------------------------------------
# Building libscope.so
#
libscope_la_SOURCES = wrap.c \
		      state.c \
		      httpstate.c \
		      report.c \
		      httpagg.c \
		      plattime.c \
		      fn.c \
		      ../os/linux/os.c \
		      cfgutils.c \
		      cfg.c \
		      transport.c \
		      log.c \
		      mtc.c \
		      circbuf.c \
		      linklist.c \
		      evtformat.c \
		      ctl.c \
		      mtcformat.c \
		      com.c \
		      dbg.c \
		      search.c \
		      sysexec.c \
		      gocontext.S \
		      scopeelf.c \
		      wrap_go.c \
		      utils.c \
		      bashmem.c \
		      javabci.c \
		      javaagent.c

# These are compiled into the library this way so they are included but are not
# visible. See `nm .libs/libscope.so`.
libscope_la_SOURCES += $(CJSON_DIR)/cJSON.c
libscope_la_SOURCES += $(YAML_DIR)/src/*.c

libscope_la_LDFLAGS = -e __scope_main \
		      -ldl -lpthread -lrt -lresolv \
		      -L$(FUNCHOOK_DIR)/build -l:libfunchook.a -l:libdistorm.a \
		      -L$(PCRE2_DIR)/build -l:libpcre2-posix.a -l:libpcre2-8.a 

# -----------------------------------------------------------------------------
# Building ldscope
#
ldscope_SOURCES = \
        fn.c \
	../os/linux/os.c \
        dbg.c \
        scopeelf.c \
        scope.c \
        utils.c \
	inject.c

ldscope_LDADD = -lrt -ldl libscope.o

libscope.o: libscope.la
	$(OBJCOPY) -I binary -O elf64-x86-64 -B i386 .libs/libscope.so libscope.o

# This is needed since there are sources used in both the program and the library.
# See https://www.gnu.org/software/automake/manual/html_node/Objects-created-both-with-libtool-and-without.html
ldscope_CFLAGS = $(AM_CFLAGS)

# -----------------------------------------------------------------------------
# Common
#

AM_CPPFLAGS = 
AM_CFLAGS =
AM_LDFLAGS =
LDADD = -ldl -lpthread -lrt -lresolv

if DEBUG
AM_CFLAGS += -g -O0 -DDEBUG
else
AM_CFLAGS += -DNDEBUG
endif

# Added to match the original os/linux/Makefile
AM_CFLAGS += -Wall -Wno-nonnull -Wno-deprecated-declarations
AM_CFLAGS += -fvisibility=hidden 
AM_CFLAGS += -fno-stack-protector

# TODO: Replace SCOPE_VER with PACKAGE_VERSION in the code
AM_CPPFLAGS += -DSCOPE_VER=\"$(PACKAGE_VERSION)\"

# Contribs
AM_CPPFLAGS += -I$(JNI_DIR) -I$(JNI_DIR)/linux/
AM_CPPFLAGS += -I$(YAML_DIR)/include $(YAML_DEFINES)
AM_CPPFLAGS += -I$(CJSON_DIR) $(CJSON_DEFINES)
AM_CPPFLAGS += -I$(PCRE2_DIR)/src -I$(PCRE2_DIR)/build
AM_CPPFLAGS += -I$(FUNCHOOK_DIR)/src -I$(FUNCHOOK_DIR)/distorm/include

AM_CPPFLAGS += -I../os/linux
			
# Some things not cleaned by default
maintainer-clean-local:
	$(RM) Makefile.in
	$(RM) -r $(DEPDIR)

