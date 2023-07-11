FROM postgres:14

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt install -y \
      binutils \
      findutils \
      less \
      psmisc \
      procps \
    && rm -rf /var/lib/apt/lists/*

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     mkdir -p /opt/test-runner/logs/ && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY postgres/scope-test /usr/local/scope/scope-test
ENV PATH="/usr/local/scope:/usr/local/scope/bin/:/usr/lib/postgresql/14/bin:${PATH}"
RUN echo "export PATH=${PATH}" >> /etc/profile.d/path.sh

USER postgres

ENV POSTGRES_USER LoremIpsum
ENV POSTGRES_PASSWORD LoremIpsum

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
