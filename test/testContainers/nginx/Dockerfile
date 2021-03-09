FROM centos:latest

RUN yum -y install python3 epel-release nginx httpd-tools && yum clean all

ENV SCOPE_METRIC_DEST=udp://localhost:8125
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
RUN python3 -m pip install -r /opt/test-runner/requirements.txt

COPY ./test_runner/ /opt/test-runner/
RUN mkdir -p /opt/test-runner/logs/

ENTRYPOINT ["python3", "/opt/test-runner/app.py", "--target", "nginx","-l","/opt/test-runner/logs/", "--scope_path", "/opt/test-runner/bin/libscope.so"]
#ENTRYPOINT ["tail", "-f", "/dev/null"]
# python3 /opt/test-runner/app.py --target nginx -l /opt/test-runner/logs/ --scope_path=/opt/test-runner/bin/libscope.so
# nginx -g "daemon off;"
# ab -n 10000 -c 4 -p ./post.json http://localhost/log/
