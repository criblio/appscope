ARG GO_IMAGE_VER=golang:1.14
FROM $GO_IMAGE_VER

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y vim

#ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
COPY ./libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
#ADD http://cdn.cribl.io/dl/scope/latest/linux/scope /usr/bin/scope
COPY ./scope /usr/bin/scope

RUN mkdir /go/thread
COPY ./go/thread/fileThread.go /go/thread
RUN cd /go/thread && CGO_ENABLED=0 go build fileThread.go

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
#ENV SCOPE_EVENT_DEST=file:///go/events.log
#ENV SCOPE_METRIC_DEST=udp://localhost:8125
#ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
#ENV LD_PRELOAD=/usr/lib/libscope.so

#ENTRYPOINT ["tail", "-f", "/dev/null"]
#ENTRYPOINT ["/go/thread/fileThread"]
ENTRYPOINT ["scope", "/go/thread/fileThread"]

