FROM alpine:latest

RUN apk add go openssl make file bash curl autoconf automake libtool openssl-dev rust nodejs ruby python3 php php-openssl apache2 apache2-ssl python3-dev py3-pip libffi-dev cargo && \
    go clean -cache

RUN mkdir -p /go /go/thread /go/net /go/cgo

COPY ./go/thread/* /go/thread/
COPY ./go/net/* /go/net/
COPY ./go/cgo/* /go/cgo/

# Build all the Go test programs
RUN cd /go/thread && \
      CGO_ENABLED=0 go build fileThread.go && \
    cd /go/net && \
      go build -buildmode=pie -o plainServerDynamic plainServer.go && \
      CGO_ENABLED=0 go build -o plainServerStatic plainServer.go && \
      go build -o tlsServerDynamic tlsServer.go && \
      CGO_ENABLED=0 go build -o tlsServerStatic tlsServer.go && \
      openssl genrsa -out server.key 2048 && \
      openssl ecparam -genkey -name secp384r1 -out server.key && \
      openssl req -new -x509 -sha256 -key server.key -out server.crt \
        -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost" && \
      go build -o plainClientDynamic plainClient.go && \
      CGO_ENABLED=0 go build -o plainClientStatic plainClient.go && \
      go build -o tlsClientDynamic tlsClient.go && \
      CGO_ENABLED=0 go build -o tlsClientStatic tlsClient.go && \
    cd /go/cgo && \
      make all

# Building three versions of curl; curl-ssl, curl-tls, and curl-nss
RUN mkdir -p /opt/test && \
    cd /opt/test && \
      wget https://curl.haxx.se/download/curl-7.69.1.tar.gz && \
      tar xvzf curl-7.69.1.tar.gz && \
      mv curl-7.69.1 curlssl && \
    cd /opt/test/curlssl && \
      ./buildconf && \
      ./configure --with-ssl --without-gnutls --without-nss && \
      make -j8 && \
    cd /opt/test && \
      tar xvzf curl-7.69.1.tar.gz && \
      mv curl-7.69.1 curltls && \
    cd /opt/test/curltls && \
      ./buildconf && \
      ./configure --without-ssl --with-gnutls --without-nss && \
      make -j8 && \
    cd /opt/test && \
      tar xvzf curl-7.69.1.tar.gz && \
      mv curl-7.69.1 curlnss && \
    cd /opt/test/curlnss && \
      ./buildconf && \
      ./configure --without-ssl --without-gnutls --with-nss && \
      make -j8 && \
    cd /opt/test && \
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

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

# We set these in test_all.sh now
#ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
#ENV SCOPE_EVENT_DEST=file:///go/events.log

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY alpine/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

