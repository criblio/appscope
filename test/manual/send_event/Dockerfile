FROM ubuntu:latest

RUN apt update && apt install -y \
  curl \
  emacs \
  gdb \
  net-tools \
  vim \
  apache2-utils \
  nginx \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/test && \
    mkdir /usr/local/scope && \
    mkdir /usr/local/scope/bin && \
    mkdir /usr/local/scope/lib

COPY conf_1 /opt/test/conf_1
COPY conf_2 /opt/test/conf_2
COPY tcpserver /usr/bin/tcpserver
COPY scope-test /usr/local/scope/

RUN  ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
