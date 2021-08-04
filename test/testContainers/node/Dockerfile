FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
      emacs \
      gdb \
      lsof \
      vim \
      npm \
      git \
    && rm -rf /var/lib/apt/lists/*

# Tests
COPY ./node/test_node.sh /opt/test/bin/test_node.sh
COPY ./node/test_all.sh /opt/test/bin/test_all.sh
COPY ./node/package.json /package.json
COPY ./node/package-lock.json /package-lock.json

# tcpserver is an app built from git@bitbucket.org:cribl/scope.git
# gcc -g test/manual/tcpserver.c -lpthread -o tcpserver
# this version has a 1 second artificial delay to simulate latency
# experienced when connecting to a remote host
COPY node/tcpserver /usr/bin/tcpserver

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY node/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

