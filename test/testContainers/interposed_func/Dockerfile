FROM centos:6

RUN yum -y update
RUN yum -y install centos-release-scl
RUN yum -y install libtool
RUN yum -y groupinstall "Development Tools"
RUN yum -y install flex-devel

RUN yum -y install devtoolset-7  wget automake

RUN yum -y install rh-python36

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/

RUN source scl_source enable rh-python36 && virtualenv /opt/test-runner/

RUN mkdir -p /opt/test

RUN cd /opt/test &&  wget https://github.com/linux-test-project/ltp/archive/master.zip && unzip master.zip && rm master.zip && mv ltp-master ltp

RUN cd /opt/test/ltp && make autotools && ./configure && make -j

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_OUT_VERBOSITY=4

COPY ./interposed_func/altp /opt/test/altp
RUN cd /opt/test/altp && make -j

COPY ./interposed_func_test_runner /opt/test-runner/

ENTRYPOINT ["/opt/test-runner/bin/python", "/opt/test-runner/runner.py"]
CMD ["--config", "/opt/test-runner/.runner.conf"]
