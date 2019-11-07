FROM docker.elastic.co/elasticsearch/elasticsearch:7.4.2
ENV discovery.type=single-node

RUN yum -y update
RUN yum -y install centos-release-scl
RUN yum -y install rh-python36

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/

RUN source scl_source enable rh-python36 && virtualenv /opt/test-runner/

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ENV SCOPE_OUT_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_OUT_VERBOSITY=4

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt

COPY ./test_runner/ /opt/test-runner/

ENTRYPOINT ["/opt/test-runner/bin/python", "/opt/test-runner/app.py", "--target", "elastic","-l","/opt/test-runner/logs/"]