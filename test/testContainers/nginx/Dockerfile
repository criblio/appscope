FROM centos:6

RUN yum -y update
RUN yum -y install centos-release-scl
RUN yum -y install rh-python36
RUN yum -y install epel-release
RUN yum -y install nginx httpd-tools

RUN mkdir /opt/test-runner/

RUN source scl_source enable rh-python36 && virtualenv /opt/test-runner/

ADD http://cdn.cribl.io/dl/scope/latest/linux/libwrap.so /usr/lib/libwrap.so
RUN chmod 755 /usr/lib/libwrap.so
ENV SCOPE_OUT_DEST=file:///tmp/scope_out.log
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///tmp/scope.log
ENV SCOPE_OUT_VERBOSITY=4

COPY ./applications_test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN /opt/test-runner/bin/pip install -r /opt/test-runner/requirements.txt

COPY ./applications_test_runner/ /opt/test-runner/

ENTRYPOINT ["/opt/test-runner/bin/python", "/opt/test-runner/app.py"]