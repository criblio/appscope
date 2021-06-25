FROM centos:7

RUN yum -y update && \
    yum -y install centos-release-scl && \
    yum -y groupinstall "Development Tools" && \
    yum -y install libtool flex-devel devtoolset-7 wget automake rh-python36

RUN mkdir -p /opt/test-runner/logs/ /opt/test

RUN source scl_source enable rh-python36 && virtualenv /opt/test-runner/

RUN cd /opt/test && \
    wget https://github.com/linux-test-project/ltp/archive/refs/tags/20210524.zip && \
    unzip 20210524.zip && \
    rm 20210524.zip && \
    mv ltp-20210524 ltp && \
    cd /opt/test/ltp && \
    make autotools && \
    ./configure && \
    make -j

RUN yum -y install vim emacs gdb tcpdump lsof 

ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
#ENV SCOPE_EVENT_DEST=tcp://172.16.198.132:9109
ENV SCOPE_THREAD_DELAY="dup205"

COPY ./syscalls/altp /opt/test/altp
RUN cd /opt/test/altp && make -j

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt

COPY ./test_runner /opt/test-runner/
COPY ./syscalls/syscall_tests_conf.json /opt/test-runner/syscall_tests_conf.json

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit
RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/libscope.so /usr/local/scope/lib/libscope.so

COPY ./syscalls/scope-test /usr/local/scope/

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
