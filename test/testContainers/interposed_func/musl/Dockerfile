FROM alpine:latest

#RUN apk update
#RUN apk add make gcc wget pkgconf automake libtool autoconf bash
#RUN apk add --no-cache musl-dev
RUN apk add python3 bash

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/

RUN mkdir -p /opt/test

RUN cd /opt/test &&  wget https://github.com/linux-test-project/ltp/archive/master.zip && unzip master.zip && rm master.zip && mv ltp-master ltp

RUN /opt/test/ltp/travis/alpine.sh

RUN rm /opt/test/ltp/testcases/cve/cve-2016-7042.c
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/add_key/
#RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/cma/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/confstr/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/fmtmsg/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/getcontext/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/keyctl/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/request_key/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/rt_tgsigqueueinfo
RUN rm /opt/test/ltp/testcases/kernel/syscalls/sendto/sendto03.c
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/setrlimit
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/setsockopt/
RUN rm -r /opt/test/ltp/testcases/kernel/syscalls/timer_create/
RUN rm -r /opt/test/ltp/testcases/kernel/sched/
COPY ./interposed_func/musl/Makefile.musl /opt/test/ltp/testcases/kernel/sched/Makefile
RUN rm -r /opt/test/ltp/testcases/network
COPY ./interposed_func/musl/Makefile.musl /opt/test/ltp/testcases/network/Makefile

RUN cd /opt/test/ltp && make autotools && ./configure && make -j

ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true
ENV SCOPE_THREAD_DELAY="dup205"

COPY ./interposed_func/altp /opt/test/altp
RUN cd /opt/test/altp && make -j

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN python3 -m ensurepip --default-pip
RUN python3 -m pip install -r /opt/test-runner/requirements.txt

COPY ./test_runner /opt/test-runner/
COPY ./interposed_func/musl/syscall_tests_conf.json /opt/test-runner/syscall_tests_conf.json
COPY ./interposed_func/musl/app.py /opt/test-runner/app.py
COPY ./interposed_func/musl/scope.py /opt/test-runner/scope.py

ENTRYPOINT ["python3", "/opt/test-runner/app.py", "--target", "syscalls", "-l", "/opt/test-runner/logs/", "-s", "/opt/test-runner/bin/libscope.so", "--syscalls_tests_config", "/opt/test-runner/syscall_tests_conf.json"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
# python3 /opt/test-runner/app.py -s /opt/test-runner/bin/libscope.so --target syscalls -l /opt/test-runner/logs/ --syscalls_tests_config /opt/test-runner/syscall_tests_conf.json
