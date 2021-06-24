FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Tests
COPY ./attach/test_attach.sh /opt/test/bin/test_attach.sh
COPY ./attach/test_all.sh /opt/test/bin/test_all.sh
COPY ./attach/expected.yml /

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

COPY attach/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
