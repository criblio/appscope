FROM alpine:latest

RUN apk add go openssl make file bash curl autoconf automake libtool openssl-dev rust nodejs ruby python3 php php-openssl apache2 apache2-ssl python3-dev py3-pip libffi-dev cargo && \
    go clean -cache

RUN mkdir -p /go

RUN mkdir -p /go/thread
COPY ./go/thread/fileThread.go /go/thread
RUN cd /go/thread && CGO_ENABLED=0 go build fileThread.go

RUN mkdir -p /go/net
COPY ./go/net/plainServer.go /go/net
RUN cd /go/net && go build -buildmode=pie -o plainServerDynamic plainServer.go
RUN cd /go/net && CGO_ENABLED=0 go build -o plainServerStatic plainServer.go
COPY ./go/net/tlsServer.go /go/net
RUN cd /go/net && go build -o tlsServerDynamic tlsServer.go
RUN cd /go/net && CGO_ENABLED=0 go build -o tlsServerStatic tlsServer.go
RUN cd /go/net && openssl genrsa -out server.key 2048 && \
                  openssl ecparam -genkey -name secp384r1 -out server.key && \
                  openssl req -new -x509 -sha256 -key server.key -out server.crt \
       -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"
COPY ./go/net/plainClient.go /go/net
RUN cd /go/net && go build -o plainClientDynamic plainClient.go
RUN cd /go/net && CGO_ENABLED=0 go build -o plainClientStatic plainClient.go
COPY ./go/net/tlsClient.go /go/net
RUN cd /go/net && go build -o tlsClientDynamic tlsClient.go
RUN cd /go/net && CGO_ENABLED=0 go build -o tlsClientStatic tlsClient.go

RUN mkdir -p /go/cgo
COPY ./go/cgo/Makefile /go/cgo
COPY ./go/cgo/myc.c /go/cgo
COPY ./go/cgo/myc.h /go/cgo
COPY ./go/cgo/mygo.go /go/cgo
RUN cd /go/cgo && make all

# Building two versions of curl. Download the source first.
RUN mkdir -p /opt/test && \
    cd /opt/test && \
    wget https://curl.haxx.se/download/curl-7.69.1.tar.gz
# Build the SSL version
RUN cd /opt/test && \
    tar xvzf curl-7.69.1.tar.gz && \
    mv curl-7.69.1 curlssl && \
    cd curlssl && \
    ./buildconf && \
    ./configure --with-ssl --without-gnutls --without-nss && \
    make -j8
# Build the TLS version
RUN cd /opt/test && \
    tar xvzf curl-7.69.1.tar.gz && \
    mv curl-7.69.1 curltls && \
    cd curltls && \
    ./buildconf && \
    ./configure --without-ssl --with-gnutls --without-nss && \
    make -j8
# Build the NSS version
RUN cd /opt/test && \
    tar xvzf curl-7.69.1.tar.gz && \
    mv curl-7.69.1 curlnss && \
    cd curlnss && \
    ./buildconf && \
    ./configure --without-ssl --without-gnutls --with-nss && \
    make -j8
# Save the results and clean up
RUN cd /opt/test && \
    cp curlssl/src/.libs/curl curl-ssl && \
    cp curltls/src/.libs/curl curl-tls && \
    cp curlnss/src/.libs/curl curl-nss && \
    rm -rf curlssl curltls curlnss curl-7.69.1.tar.gz

# Install test-runner pieces
RUN mkdir /opt/test/logs /opt/test/bin
COPY ./tls/test_cert.pem /opt/test/.
# Node
COPY ./tls/nodehttp.ts /opt/test/bin/nodehttp.ts
# Ruby
COPY ./tls/ruby/server.rb /opt/test/bin/.
COPY ./tls/ruby/client.rb /opt/test/bin/.
# Python
COPY ./tls/testssl.py /opt/test/bin/testssl.py
RUN pip install pyopenssl
# PHP
COPY ./tls/php/sslclient.php /opt/test/bin
# Rust
COPY ./tls/rust/http_test /opt/test/rust
RUN cd /opt/test/rust && \
    cargo build --release && \
    cp target/release/http_test /opt/test/bin && \
    cd /opt/test && \
    rm -rf /opt/test/rust

# Debug Tools
COPY ./tls/alias /root/.alias
COPY ./tls/gdbinit /root/.gdbinit

# Tests
COPY ./alpine/test_go.sh /go
COPY ./go/test_go_struct.sh /go
COPY ./alpine/test_tls.sh /opt/test/bin/test_tls.sh
COPY ./alpine/test_scope.sh /opt/test/bin/test_scope.sh
COPY ./alpine/test_all.sh /opt/test/bin/test_all.sh

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

# We set these in test_all.sh now
#ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
#ENV SCOPE_EVENT_DEST=file:///go/events.log

# Don't need these
#ENV SCOPE_METRIC_DEST=udp://localhost:8125
#ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
#ENV LD_PRELOAD=/usr/lib/libscope.so

COPY alpine/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
