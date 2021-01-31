FROM ubuntu:20.04
WORKDIR /root
# COPY scope.yml /etc/scope/scope.yml
COPY build/scope /usr/local/bin/scope
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y --no-install-recommends wget curl ca-certificates jq python3 vim netcat-traditional nginx openssl libwww-perl openjdk-14-jre
COPY demo/entrypoint.sh /sbin/entrypoint.sh
COPY demo/nginx/nginx.conf /etc/nginx/nginx.conf
COPY demo/nginx/html/* /opt/html/
COPY demo/demo-scripts/* /root/

ENTRYPOINT ["/sbin/entrypoint.sh"]
