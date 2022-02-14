FROM quay.io/centos/centos:stream8

#RUN curl https://www.getpagespeed.com/files/centos6-eol.repo --output /etc/yum.repos.d/CentOS-Base.repo
#RUN curl https://www.getpagespeed.com/files/centos6-epel-eol.repo --output /etc/yum.repos.d/epel.repo

#RUN yum -y update

#RUN curl https://www.getpagespeed.com/files/centos6-scl-eol.repo --output /etc/yum.repos.d/CentOS-SCLo-scl.repo
#RUN curl https://www.getpagespeed.com/files/centos6-scl-rh-eol.repo --output /etc/yum.repos.d/CentOS-SCLo-scl-rh.repo
#RUN yum -y install centos-release-scl
#RUN yum -y install rh-python36
RUN yum -y install python3 && \
    yum -y install epel-release && \
    yum -y install nginx httpd-tools

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/

#RUN source scl_source enable rh-python36 && virtualenv /opt/test-runner/

ENV SCOPE_CRIBL_ENABLE=false
ENV SCOPE_METRIC_DEST=udp://localhost:8125
ENV SCOPE_EVENT_DEST=file:///dev/null
ENV SCOPE_EVENT_ENABLE=false
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_METRIC_VERBOSITY=4
ENV SCOPE_EVENT_LOGFILE=true
ENV SCOPE_EVENT_CONSOLE=true
ENV SCOPE_EVENT_METRIC=true
ENV SCOPE_EVENT_HTTP=true

COPY ./nginx/nginx.conf /etc/nginx/nginx.conf
COPY ./nginx/post.json /opt/test-runner/post.json

COPY ./test_runner/requirements.txt /opt/test-runner/requirements.txt
#RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt
RUN python3 -m pip install -r /opt/test-runner/requirements.txt

COPY ./test_runner/ /opt/test-runner/

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY nginx/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]

