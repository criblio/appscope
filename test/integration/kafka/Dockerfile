FROM centos:7

RUN yum -y update && \
    yum -y install centos-release-scl && \
    yum -y install rh-python38 python-virtualenv && \
    yum -y install epel-release && \
    yum -y install java-1.8.0-openjdk && \
    yum -y install mc net-tools lsof vim emacs gdb

RUN mkdir -p /opt/test-runner/logs/

RUN source scl_source enable rh-python38 &&\
    virtualenv -p $(which python) /opt/test-runner/

# Originally used 2.1.1 and 2.11, but they're too old now.
ENV KAFKA_VERSION=2.7.0 KAFKA_SCALA_VERSION=2.13
ENV KAFKA_RELEASE_ARCHIVE kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION}.tgz

RUN mkdir /kafka /data /logs

# Download Kafka binary distribution
#COPY ./kafka/kafka_2.13-2.5.0.tgz /tmp/kafka_2.13-2.5.0.tgz

ADD https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/${KAFKA_RELEASE_ARCHIVE} /tmp/
ADD https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/${KAFKA_RELEASE_ARCHIVE}.md5 /tmp/

WORKDIR /tmp

# Check artifact digest integrity
RUN echo VERIFY CHECKSUM: && \
  gpg --print-md MD5 ${KAFKA_RELEASE_ARCHIVE} 2>/dev/null && \
  cat ${KAFKA_RELEASE_ARCHIVE}.md5

# Install Kafka to /kafka
RUN tar -zx -C /kafka --strip-components=1 -f ${KAFKA_RELEASE_ARCHIVE} && \
  rm -rf kafka_*

ADD ./kafka/config /kafka/config

# Set up a user to run Kafka
RUN groupadd kafka && \
  useradd -d /kafka -g kafka -s /bin/false kafka && \
  chown -R kafka:kafka /kafka /data /logs

#ENV SCOPE_OUT_DEST=udp://localhost:8125
#ENV SCOPE_OUT_VERBOSITY=4
#ENV SCOPE_EVENT_DEST=tcp://172.16.198.132:9109
ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt
RUN /opt/test-runner/bin/pip install kafka-python

COPY ./test_runner/ /opt/test-runner/

# Switching to Python 3.8 required this hack. Not sure where the kafka packages are coming from.
RUN sed -i 's/\basync\b/is_async/g' /opt/test-runner/lib/python3.8/site-packages/kafka/producer/*.py

ENV PATH /kafka/bin:$PATH
WORKDIR /kafka

# broker, jmx
VOLUME [ "/data", "/logs" ]

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY kafka/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
