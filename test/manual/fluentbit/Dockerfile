FROM ubuntu:latest

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  docker-compose \
  curl \
  wget \
  emacs \
  gdb \
  net-tools \
  vim \
  apache2-utils \
  nginx \
  gnupg \
  libpq5 \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/test/fluentbit
COPY fluentbit/services /opt/test/fluentbit/services
COPY fluentbit/docker-compose.yml /opt/test/fluentbit/docker-compose.yml

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit




RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY fluentbit/scope-test /usr/local/scope/

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]




