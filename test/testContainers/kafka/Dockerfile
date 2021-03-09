FROM centos:7

RUN yum -y install centos-release-scl \
  && yum -y install rh-python36 epel-release java-1.8.0-openjdk \
  && yum clean all

RUN mkdir -p /opt/test-runner/logs/ /kafka /data /logs

RUN source scl_source enable rh-python36 && virtualenv /opt/test-runner/

# Originally used 2.1.1 and 2.11, but they're too old now.
ENV KAFKA_VERSION=2.7.0 KAFKA_SCALA_VERSION=2.13
ENV KAFKA_RELEASE_ARCHIVE kafka_${KAFKA_SCALA_VERSION}-${KAFKA_VERSION}.tgz

# Download Kafka binary distribution
#COPY ./kafka/kafka_2.13-2.5.0.tgz /tmp/kafka_2.13-2.5.0.tgz

ADD http://www.us.apache.org/dist/kafka/${KAFKA_VERSION}/${KAFKA_RELEASE_ARCHIVE} /tmp/
ADD https://dist.apache.org/repos/dist/release/kafka/${KAFKA_VERSION}/${KAFKA_RELEASE_ARCHIVE}.md5 /tmp/

WORKDIR /tmp

# Check artifact digest integrity
RUN echo VERIFY CHECKSUM: && \
  gpg --print-md MD5 ${KAFKA_RELEASE_ARCHIVE} 2>/dev/null && \
  cat ${KAFKA_RELEASE_ARCHIVE}.md5

# Install Kafka to /kafka
RUN tar -zx -C /kafka --strip-components=1 -f ${KAFKA_RELEASE_ARCHIVE} && \
  rm -rf kafka_*

COPY ./kafka/config /kafka/config

# Set up a user to run Kafka
RUN groupadd kafka && \
  useradd -d /kafka -g kafka -s /bin/false kafka && \
  chown -R kafka:kafka /kafka /data /logs

#ENV SCOPE_OUT_DEST=udp://localhost:8125
#ENV SCOPE_OUT_VERBOSITY=4
#ENV SCOPE_EVENT_DEST=tcp://172.16.198.132:9109
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

COPY ./test_runner/ /opt/test-runner/
RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt \
  && /opt/test-runner/bin/pip install kafka-python

ENV PATH /kafka/bin:$PATH
WORKDIR /kafka

# broker, jmx
VOLUME [ "/data", "/logs" ]

ENTRYPOINT ["/opt/test-runner/bin/python", "/opt/test-runner/app.py", "--target", "kafka","-l","/opt/test-runner/logs/"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
