FROM cribl/cribl:3.4.1

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -o Acquire::Check-Valid-Until=false update && apt install -y \
  curl \
  net-tools \
  socat \
  lsof \
  golang-go \
&& rm -rf /var/lib/apt/lists/*

ENV CRIBL_NOAUTH=1
COPY transport/cribl/ /opt/cribl/local/cribl/

RUN mkdir -p /opt/test/bin
COPY transport/scope-test /opt/test/bin

COPY transport/plainClient.go /opt/test/bin
RUN cd /opt/test/bin && \
    openssl genrsa -out server.key 2048 && \
    openssl ecparam -genkey -name secp384r1 -out server.key && \
    openssl req -new -x509 -sha256 \
      -key server.key \
      -out server.crt \
      -days 3650 \
      -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"
RUN  cd /opt/test/bin && \
     CGO_ENABLED=0 go build -o plainClientStatic plainClient.go

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY transport/scope-test /usr/local/scope/

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
