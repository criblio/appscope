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

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/libscope.so /usr/local/scope/lib/libscope.so

COPY attach/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

