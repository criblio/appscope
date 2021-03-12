ARG BASE_IMAGE=ubuntu:20.04
FROM $BASE_IMAGE

RUN apt-get -o Acquire::Check-Valid-Until=false update \
  && apt-get install -y curl bison git make gcc \
  && curl -sSL https://raw.githubusercontent.com/moovweb/gvm/master/binscripts/gvm-installer | bash -x \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /go/net && cd /go/net && openssl genrsa -out server.key 2048 && \
  openssl ecparam -genkey -name secp384r1 -out server.key && \
  openssl req -new -x509 -sha256 -key server.key -out server.crt \
  -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"

COPY ./ /go
ARG GO_VERSION
RUN /go/build_go.sh $GO_VERSION

FROM ubuntu:20.04

RUN apt-get update && apt-get install -y curl bsdmainutils binutils \
  && rm -rf /var/lib/apt/lists/*
COPY ./influx/influxdb-selfsigned.key ./influx/influxdb-selfsigned.crt /etc/ssl/

ENV SCOPE_LOG_LEVEL=error
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///go/events.log

ARG GO_VERSION
COPY --from=0 /root/.gvm/gos/go$GO_VERSION/bin/go /usr/bin/go
COPY --from=0 /go /go
