FROM cribl/cribl:latest

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        python3.6 \
        python3-setuptools \
        python3-pip \
    && apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY cribl/http_status.csv /opt/cribl/data/lookups/http_status.csv
COPY cribl/scripts/ /opt/cribl/scripts/
COPY cribl/cribl/ /opt/cribl/local/cribl/

#ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
#COPY ./libscope.so /usr/lib/libscope.so
#RUN chmod 755 /usr/lib/libscope.so
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
RUN pip3 install wheel && pip3 install -r /opt/test-runner/requirements.txt

COPY ./test_runner/ /opt/test-runner/
RUN mkdir -p /opt/test-runner/logs/

ENTRYPOINT ["python3.6", "/opt/test-runner/app.py", "--target", "cribl","-l","/opt/test-runner/logs/"]
