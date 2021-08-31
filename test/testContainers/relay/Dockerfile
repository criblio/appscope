FROM cribl/cribl:3.1.0-RC4

RUN apt update && apt install -y \
  curl \
  emacs \
  gdb \
  net-tools \
  vim \
  apache2-utils \
  nginx \
&& rm -rf /var/lib/apt/lists/*

ENV CRIBL_NOAUTH=1
COPY relay/cribl/ /opt/cribl/local/cribl/

RUN mkdir /opt/test
COPY relay/conf_1 /opt/test/conf_1



COPY relay/tcpserver /usr/bin/tcpserver

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY relay/scope-test /usr/local/scope/

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
