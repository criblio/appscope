FROM centos:6

RUN yum -y update
RUN yum -y install centos-release-scl
RUN yum -y install epel-release
RUN yum -y install redis
RUN yum -y install wget
RUN yum -y install gdb

RUN wget https://repo.mongodb.org/yum/redhat/6Server/mongodb-org/4.2/x86_64/RPMS/mongodb-org-4.2.6-1.el6.x86_64.rpm
RUN wget https://repo.mongodb.org/yum/redhat/6Server/mongodb-org/4.2/x86_64/RPMS/mongodb-org-server-4.2.6-1.el6.x86_64.rpm
RUN wget https://repo.mongodb.org/yum/redhat/6Server/mongodb-org/4.2/x86_64/RPMS/mongodb-org-shell-4.2.6-1.el6.x86_64.rpm
RUN wget https://repo.mongodb.org/yum/redhat/6Server/mongodb-org/4.2/x86_64/RPMS/mongodb-org-mongos-4.2.6-1.el6.x86_64.rpm
RUN wget https://repo.mongodb.org/yum/redhat/6Server/mongodb-org/4.2/x86_64/RPMS/mongodb-org-tools-4.2.6-1.el6.x86_64.rpm


RUN rpm -ihv mongodb-org-tools-4.2.6-1.el6.x86_64.rpm mongodb-org-mongos-4.2.6-1.el6.x86_64.rpm mongodb-org-shell-4.2.6-1.el6.x86_64.rpm mongodb-org-4.2.6-1.el6.x86_64.rpm mongodb-org-server-4.2.6-1.el6.x86_64.rpm

RUN rm *.rpm

COPY mongod.conf /etc/.

ADD entrypoint.sh /sbin/entrypoint.sh
ADD loaddata.sh /sbin/loaddata.sh
ADD session.csv /data/session.csv
CMD ["start"]

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/
RUN mkdir /opt/test-runner/bin

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
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

COPY test_protocols.sh /opt/test-runner/bin/test_protocols.sh
RUN chmod +x /opt/test-runner/bin/test_protocols.sh

ENTRYPOINT ["/sbin/entrypoint.sh"]
