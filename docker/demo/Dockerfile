ARG VERSION
FROM scope:${VERSION}
WORKDIR /root
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && \
    apt-get install -y --no-install-recommends wget curl ca-certificates jq \
            python3 vim netcat-traditional nginx openssl libwww-perl openjdk-14-jre \
            bat && \
    ln -s /usr/bin/batcat /usr/local/bin/bat
COPY docker/demo/entrypoint.sh /sbin/entrypoint.sh
COPY docker/demo/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/demo/nginx/html/* /opt/html/
COPY docker/demo/demo-scripts/* /root/

ENTRYPOINT ["/sbin/entrypoint.sh"]
