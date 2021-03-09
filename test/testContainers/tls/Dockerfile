FROM centos:7

RUN yum -y install centos-release-scl \
  && yum -y install rh-python36 epel-release gnutls-devel openssl-devel wget automake python-pip ruby php httpd mod_ssl libtool \
  && yum -y --enablerepo=centos-sclo-rh-testing install devtoolset-8-gdb \
  && source scl_source enable devtoolset-8 \
  && curl -sL https://rpm.nodesource.com/setup_11.x | bash - \
  && yum -y install nodejs && yum clean all

RUN mkdir -p /opt/test /opt/test-runner/logs /opt/test-runner/bin

RUN cd /opt/test && wget https://curl.haxx.se/download/curl-7.69.1.tar.gz \
  && tar xvzf curl-7.69.1.tar.gz \
  && cp -r /opt/test/curl-7.69.1 /opt/test/curlssl \
  && cd /opt/test/curlssl && ./buildconf \
  && ./configure --with-ssl --without-gnutls \
  && make \
  && cp -r /opt/test/curl-7.69.1 /opt/test/curltls \
  && cd /opt/test/curltls && ./buildconf \
  && ./configure --with-gnutls --without-ssl \
  && make \
  && rm -f /opt/test/curl-7.69.1.tar.gz

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
COPY rust/http_test/target/debug/http_test test_tls.sh nodehttp.ts testssl.py /opt/test-runner/bin/

COPY test_cert.pem /opt/test/

COPY ruby /opt/test-runner/ruby/

RUN /opt/rh/rh-python36/root/usr/bin/python3.6 -m pip install --upgrade pip
#RUN /opt/rh/rh-python36/root/usr/bin/pip3.6 install pyopenssl

COPY php /opt/test-runner/php/

COPY alias /root/.alias
COPY gdbinit /root/.gdbinit

#ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
#RUN chmod 755 /usr/lib/libscope.so
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=false
ENV SCOPE_EVENT_CONSOLE=false
ENV SCOPE_EVENT_METRIC=false
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV LD_PRELOAD=/usr/lib/libscope.so

ENTRYPOINT ["/opt/test-runner/bin/test_tls.sh"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
