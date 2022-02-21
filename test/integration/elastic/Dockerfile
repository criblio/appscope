FROM docker.elastic.co/elasticsearch/elasticsearch:7.17.0
ENV discovery.type=single-node

RUN apt update && \
    apt install python3-pip -y
RUN pip3 install virtualenv

RUN mkdir /opt/test-runner/ /opt/test-runner/logs/

RUN virtualenv -p $(which python3) /opt/test-runner/

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_OUT_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_OUT_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt

COPY ./test_runner/ /opt/test-runner/

# Switching to Python 3.8 required this hack. Not sure where the kafka packages are coming from.
RUN sed -i 's/\basync\b/is_async/g' /opt/test-runner/lib/python3.8/site-packages/kafka/producer/*.py

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY elastic/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

