FROM cribl/cribl:3.1.0-RC4

RUN apt update && apt install -y \
  curl \
  wget \
  emacs \
  gdb \
  net-tools \
  vim \
  apache2-utils \
  nginx \
  gnupg \
  libpq5 \
&& rm -rf /var/lib/apt/lists/*

RUN wget -qO - https://packages.fluentbit.io/fluentbit.key | apt-key add -
RUN echo "deb https://packages.fluentbit.io/ubuntu/bionic bionic main" > /etc/apt/sources.list 
RUN apt update && apt install -y \
  td-agent-bit

RUN  mkdir /opt/test && \
     mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY fluentbit/cribl/ /opt/cribl/local/cribl/
COPY fluentbit/conf_1 /opt/test/conf_1
COPY fluentbit/tcpserver /usr/bin/tcpserver
COPY fluentbit/scope-test /usr/local/scope/
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit
COPY docker-entrypoint.sh /

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
ENV CRIBL_NOAUTH=1

ENTRYPOINT ["/docker-entrypoint.sh"]

CMD ["test"]
