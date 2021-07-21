FROM centos:latest

COPY detect_proto/mongodb.repo /etc/yum.repos.d/.
RUN yum -y update
RUN yum -y install epel-release
RUN yum -y install redis wget gdb openssl mongodb-org


RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/
RUN mkdir /opt/test-runner/bin

COPY detect_proto/mongod.conf /etc/.
COPY detect_proto/scope_protocol.yml /opt/test-runner/bin/.
COPY detect_proto/mongo.js /opt/test-runner/bin/.

COPY detect_proto/test_protocols.sh /opt/test-runner/bin/test_protocols.sh
RUN chmod +x /opt/test-runner/bin/test_protocols.sh

ADD detect_proto/loaddata.sh /sbin/loaddata.sh
ADD detect_proto/session.csv /data/session.csv

RUN /usr/lib/systemd/systemd --system &
CMD ["start"]

ENV LD_PRELOAD=/usr/local/scope/lib/libscope.so

ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_TAG_service=redis
ENV SCOPE_EVENT_ENABLE=true
ENV SCOPE_EVENT_LOGFILE=false
ENV SCOPE_EVENT_CONSOLE=false
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=false
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_ENABLE=false
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_HOME=/opt/test-runner/bin

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY detect_proto/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

