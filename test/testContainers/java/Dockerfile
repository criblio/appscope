FROM openjdk:14

RUN yum install -y wget

ENV TOMCAT_MAJOR_VERSION 8
ENV TOMCAT_MINOR_VERSION 8.0.49
ENV CATALINA_HOME /opt/tomcat
ENV PATH $PATH:$CATALINA_HOME/bin

RUN wget -q https://archive.apache.org/dist/tomcat/tomcat-${TOMCAT_MAJOR_VERSION}/v${TOMCAT_MINOR_VERSION}/bin/apache-tomcat-${TOMCAT_MINOR_VERSION}.tar.gz && \
    tar zxf apache-tomcat-*.tar.gz && \
    rm apache-tomcat-*.tar.gz && \
    mv apache-tomcat* $CATALINA_HOME

RUN mkdir /opt/tomcat/certs && \
    mkdir -p /opt/test-runner/logs && \
    mkdir -p /opt/test-runner/bin

RUN keytool -genkey -alias "tomcat" -dname "CN=scope,O=cribl" -keyalg RSA -storetype pkcs12 -keystore /opt/tomcat/certs/tomcat.p12 -storepass changeit -keypass changeit

COPY ./java/server.xml $CATALINA_HOME/conf/server.xml
COPY ./java/test-ssl.sh /opt/test-runner/bin/test-ssl.sh
RUN chmod +x /opt/test-runner/bin/test-ssl.sh

EXPOSE 8443

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=false
ENV SCOPE_EVENT_CONSOLE=false
ENV SCOPE_EVENT_METRIC=false
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_SUMMARY_PERIOD=1
ENV SCOPE_EVENT_DEST=file:///opt/test-runner/logs/events.log
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV LD_PRELOAD=/usr/lib/libscope.so

CMD /opt/test-runner/bin/test-ssl.sh