FROM frolvlad/alpine-glibc
# FROM alpine:latest
WORKDIR /gogen
RUN apk update && apk add \
    ca-certificates \
    openssl \
    wget \
    bash
RUN wget https://api.gogen.io/linux/gogen --no-check-certificate
RUN chmod 755 ./gogen

#ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
#COPY ./libscope.so /usr/lib/libscope.so
#RUN chmod 755 /usr/lib/libscope.so
#ADD http://cdn.cribl.io/dl/scope/latest/linux/scope /usr/bin/scope
#COPY ./scope /usr/bin/scope
#RUN chmod 755 /usr/bin/scope

ADD gogen/tcpserver /usr/bin/tcpserver
ADD gogen/gogen.sh /gogen/gogen.sh

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

