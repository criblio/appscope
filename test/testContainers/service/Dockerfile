FROM amazonlinux:2

RUN yum install -y tar gzip

RUN ARCH=$(uname -m | sed -e 's/x86_64/x64/' -e 's/aarch64/arm64/') \
 && curl -Lo /tmp/cribl.tgz https://cdn.cribl.io/dl/3.1.0/cribl-3.1.0-f765e418-linux-${ARCH}.tgz \
 && cd /opt \
 && tar zxf /tmp/cribl.tgz \
 && /opt/cribl/bin/cribl boot-start enable \
 && rm -f /tmp/cribl.tgz

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY service/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
