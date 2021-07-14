
all: envloop tcpserver

envloop: envloop.c
	gcc -o $@ $<

tcpserver: tcpserver.c
	gcc -lpthread -o $@ $<

clean:
	$(RM) envloop tcpserver

.PHONY: all clean
