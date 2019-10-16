FROM cribl/cribl:latest
COPY http_status.csv /opt/cribl/data/lookups/http_status.csv
COPY scripts/ /opt/cribl/scripts/
ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ENV LD_PRELOAD=/usr/lib/libscope.so
ENV SCOPE_OUT_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_OUT_VERBOSITY=4
