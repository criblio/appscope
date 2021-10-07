ARG GO_IMAGE_VER=golang:1.14
FROM $GO_IMAGE_VER

COPY ./go/unsupported/test_go.sh /go

RUN mkdir /go/thread
COPY ./go/thread/fileThread.go /go/thread
RUN cd /go/thread && \
    CGO_ENABLED=0 go build fileThread.go

RUN mkdir /go/net
COPY ./go/net/plainServer.go /go/net
RUN cd /go/net && \
    go build -o plainServerDynamic plainServer.go
RUN cd /go/net && \
    CGO_ENABLED=0 go build -o plainServerStatic plainServer.go
COPY ./go/net/tlsServer.go /go/net
RUN cd /go/net && \
    go build -o tlsServerDynamic tlsServer.go
RUN cd /go/net && \
    CGO_ENABLED=0 go build -o tlsServerStatic tlsServer.go
RUN cd /go/net && \
    openssl genrsa -out server.key 2048 && \
    openssl ecparam -genkey -name secp384r1 -out server.key && \
    openssl req -new -x509 -sha256 -key server.key -out server.crt \
     -days 3650 \
     -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"
COPY ./go/net/plainClient.go /go/net
RUN cd /go/net && \
    go build -o plainClientDynamic plainClient.go
RUN cd /go/net && \
    CGO_ENABLED=0 go build -o plainClientStatic plainClient.go
COPY ./go/net/tlsClient.go /go/net
RUN cd /go/net && \
    go build -o tlsClientDynamic tlsClient.go
RUN cd /go/net && \
    CGO_ENABLED=0 go build -o tlsClientStatic tlsClient.go

RUN mkdir /go/cgo
COPY ./go/cgo/Makefile /go/cgo
COPY ./go/cgo/myc.c /go/cgo
COPY ./go/cgo/myc.h /go/cgo
COPY ./go/cgo/mygo.go /go/cgo
RUN cd /go/cgo && \
    make all

RUN mkdir -p /go/influx
COPY ./go/influx/* /go/influx/
COPY ./go/influx/influxdb-selfsigned.key /etc/ssl/.
COPY ./go/influx/influxdb-selfsigned.crt /etc/ssl/.

ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///go/events.log
ENV SCOPE_LOG_LEVEL=warning
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_ALLOW_CONSTRUCT_DBG=true

RUN echo "export PATH=/usr/local/scope:/usr/local/scope/bin:${PATH}" >/etc/profile.d/path.sh
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/libscope.so /usr/local/scope/lib/libscope.so

COPY go/unsupported/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

