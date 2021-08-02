FROM ubuntu:20.04

RUN apt update -y && \
    apt install -y ca-certificates && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
COPY bin/linux/${TARGETARCH}/scope       /usr/local/bin/scope
COPY bin/linux/${TARGETARCH}/ldscope     /usr/local/bin/ldscope
COPY lib/linux/${TARGETARCH}/libscope.so /usr/local/lib/libscope.so

RUN mkdir /usr/local/scope /usr/local/scope/x86_64 /usr/local/scope/aarch64 \
 && ln -s x86_64 /usr/local/scope/amd64 \
 && ln -s aarch64 /usr/local/scope/arm64
   
COPY bin/linux/x86_64/scope       /usr/local/scope/x86_64/scope
COPY bin/linux/x86_64/ldscope     /usr/local/scope/x86_64/ldscope
COPY lib/linux/x86_64/libscope.so /usr/local/scope/x86_64/libscope.so
COPY conf/scope.yml               /usr/local/scope/x86_64/scope.yml
COPY conf/scope_protocol.yml      /usr/local/scope/x86_64/scope_protocol.yml
   
COPY bin/linux/aarch64/scope       /usr/local/scope/aarch64/scope
COPY bin/linux/aarch64/ldscope     /usr/local/scope/aarch64/ldscope
COPY lib/linux/aarch64/libscope.so /usr/local/scope/aarch64/libscope.so
COPY conf/scope.yml                /usr/local/scope/aarch64/scope.yml
COPY conf/scope_protocol.yml       /usr/local/scope/aarch64/scope_protocol.yml
