FROM cribl/cribl:latest

# Dependencies
RUN apt update && apt install -y \
  curl \
&& rm -rf /var/lib/apt/lists/*

# Configure Cribl LogStream
ENV CRIBL_NOAUTH=1
COPY http/cribl/ /opt/cribl/local/cribl/

# Test Environment
RUN mkdir -p /opt/test/bin
COPY http/scope-test /opt/test/bin
ENV SCOPE_CRIBL=tcp://localhost:10090

# Startup
COPY http/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
