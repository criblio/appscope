FROM ubuntu:20.04

RUN apt update -y && \
    apt install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
COPY bin/linux/${TARGETARCH}/scope /usr/local/bin/scope
