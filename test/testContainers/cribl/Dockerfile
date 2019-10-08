FROM cribl/cribl:latest
COPY http_status.csv /opt/cribl/data/lookups/http_status.csv
COPY scripts/ /opt/cribl/scripts/
ADD http://cdn.cribl.io/dl/scope/latest/linux/libwrap.so /usr/lib/libwrap.so
RUN chmod 755 /usr/lib/libwrap.so
ENV LD_PRELOAD=/usr/lib/libwrap.so
ENV SCOPE_OUT_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_OUT_VERBOSITY=4
