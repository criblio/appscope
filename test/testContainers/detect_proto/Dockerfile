FROM centos:latest

COPY mongodb.repo /etc/yum.repos.d/.
RUN yum -y update && \
    yum -y install epel-release redis wget gdb openssl mongodb-org && \
    mkdir -p /opt/test-runner/logs && mkdir /opt/test-runner/bin


# RUN yum -y update
# RUN yum -y install epel-release
# RUN yum -y install redis
# RUN yum -y install wget
# RUN yum -y install gdb
# RUN yum -y install openssl

# COPY mongodb.repo /etc/yum.repos.d/.
# RUN yum -y install mongodb-org

# RUN mkdir /opt/test-runner/
# RUN mkdir /opt/test-runner/logs/
# RUN mkdir /opt/test-runner/bin

COPY mongod.conf /etc/.
COPY scope_protocol.yml /opt/test-runner/bin/.
COPY mongo.js /opt/test-runner/bin/.

#ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
#RUN chmod 755 /usr/lib/libscope.so

COPY test_protocols.sh /opt/test-runner/bin/test_protocols.sh
RUN chmod +x /opt/test-runner/bin/test_protocols.sh

ADD entrypoint.sh /sbin/entrypoint.sh
ADD loaddata.sh /sbin/loaddata.sh
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
