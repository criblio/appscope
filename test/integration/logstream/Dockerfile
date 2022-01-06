FROM cribl/cribl:latest

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        net-tools \
        python3.6 \
        python3-setuptools \
        python3-pip \
    && apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY logstream/http_status.csv /opt/cribl/data/lookups/http_status.csv
COPY logstream/scripts/ /opt/cribl/scripts/
COPY logstream/cribl/ /opt/cribl/local/cribl/

RUN mkdir -p /opt/test-runner/logs/

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
#ENV SCOPE_EVENT_DEST=tcp://172.16.198.132:9109

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN pip3 install wheel
RUN pip3 install -r /opt/test-runner/requirements.txt

COPY ./test_runner/ /opt/test-runner/

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY logstream/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

