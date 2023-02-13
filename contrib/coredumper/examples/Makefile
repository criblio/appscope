# This makefile will probably work out of the box for you, but if not, try
# changing PTHREAD to be -pthread (possibly needed for FreeBSD), and/or CC 
# to be cc (possibly needed for Solaris).  

PORT = 9900
LIBTOOL = ../libtool
CC = gcc
PTHREAD = -lpthread

all:    tftpd

# This rule works both in the source tree, and stand-alone. In the latter case
# it assumes that the libraries and headers are installed system-wide.
tftpd:  tftpd.c
	@if [ -r ../src/coredumper.c ]; then                                  \
		( cd .. && { [ -r Makefile ] || ./configure; } && $(MAKE) ) &&\
		echo $(CC) -o $@ -g -Wall -O2 -lcoredumper $(PTHREAD)         \
		     tftpd.c &&                                               \
		$(LIBTOOL) --mode=link $(CC) -o $@ -g -Wall -O2 -I../src -L.. \
	        	   -lcoredumper $(PTHREAD) $^ ../libcoredumper.la;    \
	else                                                                  \
		echo $(CC) -o $@ -g -Wall -O2 -lcoredumper $(PTHREAD) tftpd.c;\
		$(CC) -o $@ -g -Wall -O2 -lcoredumper $(PTHREAD) tftpd.c;     \
	fi

distclean: clean
clean:
	@echo $(RM) -f tftpd core
	@if [ -r ../src/coredumper.c ]; then                                  \
		$(LIBTOOL) --mode=clean $(RM) -f tftpd core;                  \
	else                                                                  \
		$(RM) -f tftpd core;                                          \
	fi

check:  tftpd
	@echo "Starting TFTP server on port $(PORT)" &&                       \
	rm -f core && set -m &&                                               \
	{ ./tftpd --port $(PORT) </dev/null & } && pid="$$!" &&               \
	trap "kill -15 -$$pid >/dev/null 2>&1" EXIT &&                        \
	sleep 1 &&                                                            \
	echo "Downloading tftp://localhost:$(PORT)/core" &&                   \
	{ echo;                                                               \
	  echo "connect localhost $(PORT)";                                   \
	  echo "binary";                                                      \
	  echo "get core";                                                    \
	  echo "quit"; } | tftp >/dev/null &&                                 \
	kill -15 -$$pid >/dev/null 2>&1 &&                                    \
	{ wait $$pid; [ $$? -eq 143 ]; } &&                                   \
	echo "Validating core file contents" &&                               \
	readelf -a core &&                                                    \
	echo "PASS" ||                                                        \
	echo "FAILED"
