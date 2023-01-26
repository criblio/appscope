FROM centos:7

RUN yum -y update && \
    yum -y install centos-release-scl && \
    yum -y install rh-python38 python-virtualenv && \
    yum -y install epel-release && \
    yum -y groupinstall 'Development Tools' && \
    yum -y install gnutls-devel openssl-devel wget automake python-pip && \
    yum -y install ruby php httpd mod_ssl && \
    yum -y --enablerepo=centos-sclo-rh-testing install devtoolset-8-gdb && \
    source scl_source enable devtoolset-8 && \
    curl -sL https://rpm.nodesource.com/setup_12.x | bash - && \
    yum -y install nodejs

RUN mkdir /opt/test /opt/test-runner/ /opt/test-runner/logs /opt/test-runner/bin

RUN cd /opt/test && \
      wget https://curl.haxx.se/download/curl-7.69.1.tar.gz && \
      tar xvzf curl-7.69.1.tar.gz && \
      mv /opt/test/curl-7.69.1 /opt/test/curlssl && \
    cd /opt/test/curlssl && \
      ./buildconf && \
      ./configure --with-ssl --without-gnutls && \
      make && \
    cd /opt/test && \
      tar xvzf curl-7.69.1.tar.gz && \
      mv /opt/test/curl-7.69.1 /opt/test/curltls && \
    cd /opt/test/curltls && \
      ./buildconf && \
      ./configure --with-gnutls --without-ssl && \
      make
      
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
COPY ./tls/rust/http_test/target/debug/http_test /opt/test-runner/bin/.

RUN rm -f /opt/test/curl-7.69.1.tar.gz
COPY ./tls/test_cert.pem /opt/test/.

COPY ./tls/nodehttp.ts /opt/test-runner/bin/nodehttp.ts
COPY ./tls/test_tls.sh /opt/test-runner/bin/test_tls.sh
RUN chmod +x /opt/test-runner/bin/test_tls.sh

RUN mkdir /opt/test-runner/ruby
COPY ./tls/ruby/server.rb /opt/test-runner/ruby
COPY ./tls/ruby/client.rb /opt/test-runner/ruby
RUN chmod u+x /opt/test-runner/ruby/*rb

RUN /opt/rh/rh-python38/root/usr/bin/python3.8 -m pip install --upgrade pip
RUN /opt/rh/rh-python38/root/usr/local/bin/pip3.8 install pyopenssl
COPY ./tls/testssl.py /opt/test-runner/bin/testssl.py

RUN mkdir /opt/test-runner/php
COPY ./tls/php/sslclient.php /opt/test-runner/php

COPY ./tls/alias /root/.alias
COPY ./tls/gdbinit /root/.gdbinit

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=false
ENV SCOPE_EVENT_CONSOLE=false
ENV SCOPE_EVENT_METRIC=false
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
#ENV LD_PRELOAD=/usr/local/scope/lib/libscope.so

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN mkdir /usr/local/scope && \
    mkdir /usr/local/scope/bin && \
    mkdir /usr/local/scope/lib && \
    ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
    ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY tls/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
