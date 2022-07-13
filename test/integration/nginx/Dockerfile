FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
    nginx \
    openssl \
    curl \
    emacs \
    gdb \
    lsof \
    netcat \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/appscope/ && \
    mkdir /usr/local/scope && \
    mkdir /usr/local/scope/bin && \
    mkdir /usr/local/scope/lib && \
    ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
    ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
    ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

RUN openssl genrsa -out ca.key 2048 && \
    openssl req -new -key ca.key -out ca.csr -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost" && \
    openssl x509 -req -days 3650 -in ca.csr -signkey ca.key -out ca.crt && \
    cp ca.crt /etc/ssl/certs/ && \
    cp ca.key /etc/ssl/private/ && \
    cp ca.csr /etc/ssl/private/

COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit
COPY ./nginx/scope-test /usr/local/scope/scope-test
COPY ./nginx/test_nginx.sh /opt/test/bin/test_nginx.sh
COPY ./nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker-entrypoint.sh /

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_EVENT_DEST=file:///tmp/events.json
ENV SCOPE_METRIC_DEST=file:///tmp/metrics.json
ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["test"]

