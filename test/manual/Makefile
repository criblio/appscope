
all: envloop tcpserver

envloop: envloop.c
	gcc -o $@ $<

tcpserver: tcpserver.c
	gcc -g $< \
		-I../../contrib/build/openssl/include \
		-I../../contrib/openssl/include \
		-L ../../contrib/build/openssl \
		-Wl,-R$(shell pwd)/../../contrib/build/openssl \
		-lpthread -lssl -lcrypto \
		-o $@

clean:
	$(RM) envloop tcpserver

.PHONY: all clean
