FROM ubuntu:20.04
ARG TARGETARCH
COPY bin/linux /tmp/scope 
RUN echo $TARGETARCH | sed -e s/amd64/x86_64/ | sed -e s/arm64/aarch64/ > /tmp/scope/cribl-arch && \
    cp /tmp/scope/$(cat /tmp/scope/cribl-arch)/scope /usr/local/bin/scope && \
    rm -rf /tmp/scope