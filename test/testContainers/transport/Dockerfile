FROM cribl/cribl:latest

RUN apt update && apt install -y \
  curl \
&& rm -rf /var/lib/apt/lists/*

ENV CRIBL_NOAUTH=1
COPY transport/cribl/ /opt/cribl/local/cribl/

RUN mkdir -p /opt/test/bin
COPY transport/scope-test /opt/test/bin

COPY transport/docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
