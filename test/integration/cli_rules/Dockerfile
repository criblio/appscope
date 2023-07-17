FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
      curl \
      git \
      htop \
      netcat \
      nginx \
      memcached \
      redis \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
RUN echo "export PATH=/usr/local/scope:/usr/local/scope/bin:${PATH}" >/etc/profile.d/path.sh

ENV SCOPE_CLI_SKIP_HOST=true
COPY scope-profile.sh /etc/profile.d/scope_profile.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     mkdir -p /opt/test-runner/logs/ && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY cli_rules/scope-test /usr/local/scope/scope-test
COPY cli_rules/scope_rules1.yml /opt/test-runner/scope_rules1.yml
COPY cli_rules/scope_rules2.yml /opt/test-runner/scope_rules2.yml

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
