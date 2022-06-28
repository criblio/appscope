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

COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit
COPY ./nginx/scope-test /usr/local/scope/scope-test
COPY ./nginx/test_nginx.sh /opt/test/bin/test_nginx.sh
COPY ./nginx/nginx.conf /etc/nginx/nginx.conf
COPY ./nginx/ca.crt /etc/ssl/certs
COPY ./nginx/ca.csr /etc/ssl/private
COPY ./nginx/ca.key /etc/ssl/private
COPY docker-entrypoint.sh /

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"

ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["test"]

