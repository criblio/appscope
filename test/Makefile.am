#
# Cribl AppScope - Automake Input
#
# See docs/BUILD.md for details.
#

CJSON_DIR = ../contrib/cJSON
FUNCHOOK_DIR = ../contrib/funchook
JNI_DIR = ../contrib/jni
PCRE2_DIR = ../contrib/pcre2
YAML_DIR = ../contrib/libyaml
CMOCKA_DIR = ../contrib/cmocka

# -----------------------------------------------------------------------------
# Targets
#
check_PROGRAMS = cfgutilstest cfgtest transporttest logtest mtctest evtformattest ctltest httpstatetest httpheadertest httpaggtest reporttest mtcformattest circbuftest linklisttest comtest dbgtest glibcvertest selfinterposetest dnstest javabcitest searchtest 

TESTS = $(check_PROGRAMS)

TESTS_ENVIRONMENT = \
export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(CMOCKA_DIR)/build/src/;

# Some of these are mocking symbols in the library's objects so we need to be
# explicit as to which objects to use instead of using all of them by default.
httpstatetest_LDADD = ../src/httpstate.lo \
                      ../src/plattime.lo \
		      ../src/search.lo \
		      ../src/fn.lo \
		      ../os/linux/os.lo \
		      ../src/dbg.lo \
		      test.o
httpaggtest_LDADD = ../src/httpagg.lo \
		    ../src/fn.lo \
		    ../src/utils.lo \
		    ../src/dbg.lo \
		    test.o

# -----------------------------------------------------------------------------
# libscope Objects
#
# We are linking against the .o's (well, the .lo's from libtool) build for
# libscope.so earlier and not the .so because many of the library's symbols are
# not accessible otherwise.
#
libscope_OBJECTS = ../src/wrap.lo \
		   ../src/state.lo \
		   ../src/httpstate.lo \
		   ../src/report.lo \
		   ../src/httpagg.lo \
		   ../src/plattime.lo \
		   ../src/fn.lo \
		   ../os/linux/os.lo \
		   ../src/cfgutils.lo \
		   ../src/cfg.lo \
		   ../src/transport.lo \
		   ../src/log.lo \
		   ../src/mtc.lo \
		   ../src/circbuf.lo \
		   ../src/linklist.lo \
		   ../src/evtformat.lo \
		   ../src/ctl.lo \
		   ../src/mtcformat.lo \
		   ../src/com.lo \
		   ../src/dbg.lo \
		   ../src/search.lo \
		   ../src/sysexec.lo \
		   ../src/gocontext.lo \
		   ../src/scopeelf.lo \
		   ../src/wrap_go.lo \
		   ../src/utils.lo \
		   ../src/bashmem.lo \
		   ../src/javabci.lo \
		   ../src/javaagent.lo

# -----------------------------------------------------------------------------
# Common
#
AM_CPPFLAGS = -I../src
AM_LDFLAGS = -ldl -lpthread -lrt -lresolv
LDADD = test.o $(libscope_OBJECTS)

# only Linux now
AM_CPPFLAGS += -I../os/linux

# TODO: Replace SCOPE_VER with PACKAGE_VERSION in the code
AM_CPPFLAGS += -DSCOPE_VER=\"$(PACKAGE_VERSION)\"

# cmocka
AM_CPPFLAGS += -I$(CMOCKA_DIR)/include
AM_LDFLAGS += -L$(CMOCKA_DIR)/build/src -lcmocka

# includes for contribs
AM_CPPFLAGS += -I$(JNI_DIR) -I$(JNI_DIR)/linux/
AM_CPPFLAGS += -I$(CJSON_DIR)
AM_CPPFLAGS += -I$(PCRE2_DIR)/src -I$(PCRE2_DIR)/build
AM_CPPFLAGS += -I$(FUNCHOOK_DIR)/src -I$(FUNCHOOK_DIR)/distorm/include
AM_CPPFLAGS += -I$(YAML_DIR)/include

# libs for other contribs
AM_LDFLAGS += -L$(CJSON_DIR) -l:libcjson.a
AM_LDFLAGS += -L$(PCRE2_DIR)/build -l:libpcre2-posix.a -l:libpcre2-8.a
AM_LDFLAGS += -L$(FUNCHOOK_DIR)/build -l:libfunchook.a -l:libdistorm.a
AM_LDFLAGS += -L$(YAML_DIR)/src/.libs -l:libyaml.a
