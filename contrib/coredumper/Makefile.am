## Process this file with automake to produce Makefile.in

# This is so we can #include <google/foo>
AM_CPPFLAGS = -I$(top_srcdir)/src

ACLOCAL_AMFLAGS = -I m4

googleincludedir = $(includedir)/google
## The .h files you want to install (that is, .h files that people
## who install this package can include in their own applications.)
googleinclude_HEADERS =

docdir = $(prefix)/share/doc/$(PACKAGE)-$(VERSION)
## This is for HTML and other documentation you want to install.
## Add your documentation files (in doc/) in addition to these
## top-level boilerplate files.  Also add a TODO file if you have one.
dist_doc_DATA = AUTHORS COPYING ChangeLog INSTALL NEWS README TODO

## The libraries (.so's) you want to install
lib_LTLIBRARIES =

## unittests you want to run when people type 'make check'.
## TESTS is for binary unittests, check_SCRIPTS for script-based unittests.
## TESTS_ENVIRONMENT sets environment variables for when you run unittest,
## but it only seems to take effect for *binary* unittests (argh!)
TESTS =
check_SCRIPTS =
TESTS_ENVIRONMENT =
# Every time you add a unittest to check_SCRIPTS, add it here too
noinst_SCRIPTS =

# The manual pages that should be installed
man3_MANS = man/GetCoreDump.man man/WriteCoreDump.man \
            man/CoreDumpParameters.man

install-data-hook:
	ln -sf GetCoreDump.3 $(DESTDIR)$(man3dir)/GetCompressedCoreDump.3
	ln -sf GetCoreDump.3 $(DESTDIR)$(man3dir)/GetCoreDumpWith.3
	ln -sf WriteCoreDump.3 $(DESTDIR)$(man3dir)/WriteCompressedCoreDump.3
	ln -sf WriteCoreDump.3 $(DESTDIR)$(man3dir)/WriteCoreDumpWith.3
	ln -sf WriteCoreDump.3 $(DESTDIR)$(man3dir)/WriteCoreDumpLimited.3
	ln -sf WriteCoreDump.3 $(DESTDIR)$(man3dir)/WriteCoreDumpLimitedByPriority.3
	ln -sf CoreDumpParameters.3 $(DESTDIR)$(man3dir)/ClearCoreDumpParameters.3
	ln -sf CoreDumpParameters.3 $(DESTDIR)$(man3dir)/SetCoreDumpLimited.3
	ln -sf CoreDumpParameters.3 $(DESTDIR)$(man3dir)/SetCoreDumpCompressed.3
	ln -sf CoreDumpParameters.3 $(DESTDIR)$(man3dir)/SetCoreDumpLimitedByPriority.3
	ln -sf CoreDumpParameters.3 $(DESTDIR)$(man3dir)/SetCoreDumpNotes.3


uninstall-hook:
	rm -f $(DESTDIR)$(man3dir)/GetCompressedCoreDump.3
	rm -f $(DESTDIR)$(man3dir)/GetCoreDumpWith.3
	rm -f $(DESTDIR)$(man3dir)/WriteCoreDumpWith.3
	rm -f $(DESTDIR)$(man3dir)/WriteCompressedCoreDump.3
	rm -f $(DESTDIR)$(man3dir)/WriteCoreDumpLimited.3
	rm -f $(DESTDIR)$(man3dir)/WriteCoreDumpLimitedByPriority.3
	rm -f $(DESTDIR)$(man3dir)/ClearCoreDumpParameters.3
	rm -f $(DESTDIR)$(man3dir)/SetCoreDumpLimited.3
	rm -f $(DESTDIR)$(man3dir)/SetCoreDumpCompressed.3
	rm -f $(DESTDIR)$(man3dir)/SetCoreDumpLimitedByPriority.3
	rm -f $(DESTDIR)$(man3dir)/SetCoreDumpNotes.3

## vvvv RULES TO MAKE THE LIBRARIES, BINARIES, AND UNITTESTS

lib_LTLIBRARIES += libcoredumper.la
LIBCOREDUMPER_INCLUDES = src/google/coredumper.h src/elfcore.h \
                         src/linux_syscall_support.h src/linuxthreads.h \
                         src/thread_lister.h
googleinclude_HEADERS += src/google/coredumper.h
libcoredumper_la_SOURCES = src/coredumper.c \
                           src/thread_lister.c \
                           src/elfcore.c \
                           src/linuxthreads.c \
                           $(LIBCOREDUMPER_INCLUDES)
libcoredumper_la_LDFLAGS = -export-symbols $(top_srcdir)/src/libcoredumper.sym \
                           -version-info 1:0:0

TESTS += coredumper_unittest linux_syscall_support_unittest
COREDUMP_UNITTEST_INCLUDES = src/google/coredumper.h
coredumper_unittest_SOURCES = src/coredumper_unittest.c
coredumper_unittest_LDADD = libcoredumper.la -lpthread
LINUX_SYSCALL_SUPPORT_INCLUDES = src/linux_syscall_support.h
linux_syscall_support_unittest_SOURCES = src/linux_syscall_support_unittest.cc

## ^^^^ END OF RULES TO MAKE THE LIBRARIES, BINARIES, AND UNITTESTS


## This should always include $(TESTS), but may also include other
## binaries that you compile but don't want automatically installed.
noinst_PROGRAMS = $(TESTS)

rpm: dist-gzip packages/rpm.sh packages/rpm/rpm.spec
	@cd packages && ./rpm.sh ${PACKAGE} ${VERSION}

deb: dist-gzip packages/deb.sh packages/deb/*
	@cd packages && ./deb.sh ${PACKAGE} ${VERSION}

libtool: $(LIBTOOL_DEPS)
	$(SHELL) ./config.status --recheck

EXTRA_DIST = ${man3_MANS} man/GetCoreDump.man man/WriteCoreDump.man \
             packages/rpm.sh packages/rpm/rpm.spec packages/deb.sh \
             packages/deb libtool src/libcoredumper.sym examples/Makefile \
             examples/tftpd.c

DISTCLEANFILES = core-test
