FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
      emacs \
      gdb \
      lsof \
      netcat \
      vim \
    && rm -rf /var/lib/apt/lists/*

# Tests
COPY ./cli/test_cli.sh /opt/test/bin/test_cli.sh
COPY ./cli/test_cli_dest.sh /opt/test/bin/test_cli_dest.sh
COPY ./cli/test_edge.sh /opt/test/bin/test_edge.sh
COPY ./cli/test_options.sh /opt/test/bin/test_options.sh
COPY ./cli/expected.yml /

RUN mkdir /opt/test-runner && \
    mkdir -p /var/run/appscope/ && \
    mkdir -p /opt/cribl/state/ && \
    mkdir -p /opt/cribl/home/state && \
    touch /opt/test-runner/empty_file

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_CLI_SKIP_HOST=true
ENV SCOPE_FILTER=false

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY cli/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

