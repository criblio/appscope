FROM ubuntu:20.04

RUN apt update -y && \
    apt install -y ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY bin/linux/scope /usr/local/bin/scope
