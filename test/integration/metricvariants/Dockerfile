FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
      binutils \
      curl \
      netcat \
    && rm -rf /var/lib/apt/lists/*

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=9
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_DEST=file:///opt/test-runner/logs/metrics.log

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     mkdir -p /opt/test-runner/logs/ && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY metricvariants/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
