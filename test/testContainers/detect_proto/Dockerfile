FROM centos:latest

COPY mongodb.repo /etc/yum.repos.d/.
RUN yum -y install epel-release redis wget gdb openssl mongodb-org \
  && yum clean all \
  && mkdir -p /opt/test-runner/logs /opt/test-runner/bin

COPY mongod.conf /etc/
COPY test_protocols.sh mongo.js scope_protocol.yml /opt/test-runner/bin/

COPY loaddata.sh entrypoint.sh /sbin/
ADD session.csv /data/session.csv

RUN /usr/lib/systemd/systemd --system &
CMD ["start"]

ENV LD_PRELOAD=/usr/lib/libscope.so
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_TAG_service=redis
ENV SCOPE_EVENT_ENABLE=true
ENV SCOPE_EVENT_LOGFILE=false
ENV SCOPE_EVENT_CONSOLE=false
ENV SCOPE_EVENT_METRICS=true
ENV SCOPE_EVENT_HTTP=false
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_ENABLE=false
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_HOME=/opt/test-runner/bin

ENTRYPOINT ["/sbin/entrypoint.sh"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
