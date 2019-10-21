FROM centos:6

RUN yum -y update

RUN yum -y install centos-release-scl

RUN yum -y install libtool

RUN yum -y groupinstall "Development Tools"

RUN yum -y install flex-devel

RUN yum -y install devtoolset-7 python27 wget automake

RUN scl enable python27 bash

RUN mkdir -p /opt/test

RUN cd /opt/test &&  wget https://github.com/linux-test-project/ltp/archive/master.zip && unzip master.zip && rm master.zip && mv ltp-master ltp

RUN cd /opt/test/ltp && make autotools && ./configure && make -j

ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so

COPY ./interposed_func/altp /opt/test/altp
RUN cd /opt/test/altp && make -j

COPY ./interposed_func_test_runner /opt/scope-test/

ENTRYPOINT ["python", "/opt/scope-test/runner.py"]
CMD ["--config", "/opt/scope-test/.runner.conf"]
