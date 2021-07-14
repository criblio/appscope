# FROM alpine:latest
FROM frolvlad/alpine-glibc

WORKDIR /gogen
RUN apk update && apk add \
    ca-certificates \
    openssl \
    wget \
    bash
RUN wget https://api.gogen.io/linux/gogen --no-check-certificate
RUN chmod 755 ./gogen

# tcpserver is an app built from git@bitbucket.org:cribl/scope.git
# gcc -g test/manual/tcpserver.c -lpthread -o tcpserver
COPY gogen/tcpserver /usr/bin/tcpserver

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
#ENV SCOPE_EVENT_LOGFILE=true
#ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
#ENV SCOPE_EVENT_METRIC_NAME=net\.tx
ENV SCOPE_SUMMARY_PERIOD=1
#ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///gogen/events.log

#ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
RUN echo "export PATH=/usr/local/scope:/usr/local/scope/bin:${PATH}" >/etc/profile.d/path.sh
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/libscope.so /usr/local/scope/lib/libscope.so

COPY gogen/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

