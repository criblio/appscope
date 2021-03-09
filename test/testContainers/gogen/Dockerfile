FROM frolvlad/alpine-glibc
# FROM alpine:latest
WORKDIR /gogen
RUN apk update && apk add \
    ca-certificates \
    openssl \
    wget \
    bash
RUN wget https://api.gogen.io/linux/gogen --no-check-certificate && chmod 755 gogen

# tcpserver is an app built from git@bitbucket.org:cribl/scope.git
# gcc -g test/manual/tcpserver.c -lpthread -o tcpserver
COPY tcpserver /usr/bin/tcpserver
COPY gogen.sh /gogen/gogen.sh

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
#ENV SCOPE_EVENT_LOGFILE=true
#ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
#ENV SCOPE_EVENT_METRIC_NAME=net\.tx
ENV SCOPE_SUMMARY_PERIOD=1
#ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///gogen/events.log

#ENTRYPOINT ["/gogen/gogen"]
#CMD ["--help"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["/gogen/gogen.sh"]

