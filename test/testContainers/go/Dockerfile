ARG GO_IMAGE_VER=golang:1.14
FROM $GO_IMAGE_VER

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y vim curl file

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
#COPY ./libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ADD http://cdn.cribl.io/dl/scope/latest/linux/scope /usr/bin/scope
#COPY ./scope /usr/bin/scope
RUN chmod 755 /usr/bin/scope

COPY ./go/test_go.sh /go
COPY ./go/test_go_struct.sh /go

RUN mkdir /go/thread
COPY ./go/thread/fileThread.go /go/thread
RUN cd /go/thread && CGO_ENABLED=0 go build fileThread.go

RUN mkdir /go/net
COPY ./go/net/plainServer.go /go/net
RUN cd /go/net && go build -o plainServerDynamic plainServer.go
RUN cd /go/net && CGO_ENABLED=0 go build -o plainServerStatic plainServer.go
COPY ./go/net/tlsServer.go /go/net
RUN cd /go/net && go build -o tlsServerDynamic tlsServer.go
RUN cd /go/net && CGO_ENABLED=0 go build -o tlsServerStatic tlsServer.go
RUN cd /go/net && openssl genrsa -out server.key 2048 && \
                  openssl ecparam -genkey -name secp384r1 -out server.key && \
                  openssl req -new -x509 -sha256 -key server.key -out server.crt \
       -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"

RUN mkdir /go/cgo
COPY ./go/cgo/Makefile /go/cgo
COPY ./go/cgo/myc.c /go/cgo
COPY ./go/cgo/myc.h /go/cgo
COPY ./go/cgo/mygo.go /go/cgo
RUN cd /go/cgo && make all

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///go/events.log
#ENV SCOPE_METRIC_DEST=udp://localhost:8125
#ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
#ENV LD_PRELOAD=/usr/lib/libscope.so

#ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["/go/test_go.sh"]

