ARG JDK_IMAGE
FROM $JDK_IMAGE

ARG TOMCAT_MAJOR_VERSION=8
ARG TOMCAT_MINOR_VERSION=8.0.49
ARG KEYTOOL=keytool
ARG DNLD_HEXDUMP

ENV CATALINA_HOME /opt/tomcat
ENV PATH="${PATH}:${CATALINA_HOME}/bin"

ADD https://archive.apache.org/dist/tomcat/tomcat-${TOMCAT_MAJOR_VERSION}/v${TOMCAT_MINOR_VERSION}/bin/apache-tomcat-${TOMCAT_MINOR_VERSION}.tar.gz /apache-tomcat.tar.gz

# for hexdump.  java 11 and under seems to use apt-get, 12 and 13 uses yum, 14 uses microdnf
RUN eval $DNLD_HEXDUMP

RUN tar zxf apache-tomcat.tar.gz && \
    rm apache-tomcat.tar.gz && \
    mv apache-tomcat* $CATALINA_HOME

RUN mkdir /opt/tomcat/certs && \
    mkdir -p /opt/test-runner/logs && \
    mkdir -p /opt/test-runner/bin

RUN ${KEYTOOL} -genkey -alias "tomcat" -dname "CN=scope,O=cribl" -keyalg RSA -storetype pkcs12 -keystore /opt/tomcat/certs/tomcat.p12 -storepass changeit -keypass changeit

COPY ./java/server.xml $CATALINA_HOME/conf/server.xml
COPY ./java/test-ssl.sh /opt/test-runner/bin/test-ssl.sh
RUN chmod +x /opt/test-runner/bin/test-ssl.sh

RUN mkdir -p /opt/javassl
COPY ./java/SSLSocketClient.java /opt/javassl/SSLSocketClient.java
RUN javac -d /opt/javassl/ /opt/javassl/SSLSocketClient.java

RUN mkdir -p /opt/java_http
COPY ./java/SimpleHttpServer.java /opt/java_http/SimpleHttpServer.java
RUN javac -d /opt/java_http/ /opt/java_http/SimpleHttpServer.java

RUN mkdir -p /opt/java_test
COPY ./java/JavaTest.java /opt/java_test/JavaTest.java
RUN cd /opt/java_test/ && javac JavaTest.java && javac -d . JavaTest.java

COPY ./java/AttachTest.java /opt/java_test/AttachTest.java
RUN cd /opt/java_test/ && javac AttachTest.java && javac -d . AttachTest.java

COPY ./java/InjectTest.java /opt/java_test/InjectTest.java
RUN cd /opt/java_test/ && javac InjectTest.java && javac -d . InjectTest.java

RUN mkdir -p /opt/java_native_test
COPY ./java/JavaTestNative.java /opt/java_native_test/JavaTest.java
RUN cd /opt/java_native_test/ && javac JavaTest.java && javac -d . JavaTest.java

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

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
RUN echo "export PATH=${PATH}" >> /etc/profile.d/path.sh
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 04EE7237B7D453EC
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138
# cd /tmp
# apt update
# apt -y download libcrypt1
# dpkg-deb -x libcrypt1_1%3a4.4.25-2_amd64.deb  .
# cp -av lib/x86_64-linux-gnu/* /lib/x86_64-linux-gnu/
# apt -y --fix-broken install
# apt update && DEBIAN_FRONTEND=noninteractive apt install -y gdb wget libcrypt1 python3-pip

# python3-pip wget
# RUN wget -O ~/.gdbinit-gef.py -q https://github.com/hugsy/gef/raw/master/gef.py && echo source ~/.gdbinit-gef.py >> /root/.gdbinit
# RUN pip3 install capstone unicorn ropper
ENV LC_CTYPE=C.UTF-8

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY java/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

