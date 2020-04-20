FROM centos:6

RUN yum -y update
RUN yum -y install centos-release-scl
RUN yum -y install rh-python36
RUN yum -y install epel-release
RUN yum -y groupinstall 'Development Tools'
RUN yum -y install gnutls-devel
RUN yum -y install openssl-devel
RUN yum -y install wget automake

run curl -sL https://rpm.nodesource.com/setup_11.x | bash -
RUN yum -y install nodejs

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/

RUN mkdir /opt/test
RUN cd /opt/test && wget https://curl.haxx.se/download/curl-7.69.1.tar.gz

RUN cd /opt/test && tar xvzf curl-7.69.1.tar.gz
RUN mv /opt/test/curl-7.69.1 /opt/test/curlssl
RUN cd /opt/test/curlssl && ./buildconf
RUN cd /opt/test/curlssl && ./configure --with-ssl --without-gnutls
RUN cd /opt/test/curlssl && make

RUN cd /opt/test && tar xvzf curl-7.69.1.tar.gz
RUN mv /opt/test/curl-7.69.1 /opt/test/curltls
RUN cd /opt/test/curltls && ./buildconf
RUN cd /opt/test/curltls && ./configure --with-gnutls --without-ssl
RUN cd /opt/test/curltls && make

RUN rm -f /opt/test/curl-7.69.1.tar.gz
COPY ./tls/test_cert.pem /opt/test/.

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=false
ENV SCOPE_EVENT_CONSOLE=false
ENV SCOPE_EVENT_METRICS=false
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV LD_PRELOAD=/usr/lib/libscope.so

COPY ./tls/nodehttp.ts /opt/test-runner/bin/nodehttp.ts
COPY ./tls/test_tls.sh /opt/test-runner/bin/test_tls.sh
RUN chmod +x /opt/test-runner/bin/test_tls.sh

ENTRYPOINT ["/opt/test-runner/bin/test_tls.sh"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
