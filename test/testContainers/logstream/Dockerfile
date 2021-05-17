FROM cribl/cribl:latest

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.6 \
        python3-setuptools \
        python3-pip \
    && apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY logstream/http_status.csv /opt/cribl/data/lookups/http_status.csv
COPY logstream/scripts/ /opt/cribl/scripts/
COPY logstream/cribl/ /opt/cribl/local/cribl/

RUN mkdir -p /opt/test-runner/logs/

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

COPY logstream/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

