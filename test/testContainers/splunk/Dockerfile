FROM cribl/splunk:7.3.0
ARG CRIBL_DISTRO=cribl-splunk-app-linux-x64
RUN sh -c 'echo dash dash/sh boolean false | debconf-set-selections' && \
    sh -c 'DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash' && \
    echo "deb [check-valid-until=no] http://archive.debian.org/debian jessie-backports main" > /etc/apt/sources.list.d/jessie-backports.list && \
    sed -i '/deb http:\/\/deb.debian.org\/debian jessie-updates main/d' /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y vim nano curl ca-certificates jq && \
    rm -rf /var/lib/apt/lists/*
# RUN VERSION=$(curl -s https://cdn.cribl.io/versions.json | jq -r .version) && \
#     curl -Lso /tmp/${CRIBL_DISTRO}.tgz https://cdn.cribl.io/dl/$(echo ${VERSION} | cut -d '-' -f 1)/${CRIBL_DISTRO}-${VERSION}.tgz && \
RUN curl -Lso /tmp/${CRIBL_DISTRO}.tgz https://cdn.cribl.io/dl/latest/${CRIBL_DISTRO}.tgz && \
    tar -zxvf /tmp/${CRIBL_DISTRO}.tgz -C /var/opt/splunk/etc/apps && \
    rm /tmp/${CRIBL_DISTRO}.tgz
COPY hack/splunk/etc /var/opt/splunk/etc
COPY config.sh /var/opt/splunk/etc/apps/cribl/bin/config.sh
COPY cribldemo/ /var/opt/splunk/etc/apps/cribl/
ENV DEMOUSER_NAME "Demo User"
ENV DEMOUSER_EMAIL "demo@cribl.io"
ENV SPLUNK_USER root
ENV SPLUNK_START_ARGS "--accept-license --answer-yes --seed-passwd cribldemo"
ENV SPLUNK_BEFORE_START_CMD "version ${SPLUNK_START_ARGS}"
ENV SPLUNK_BEFORE_START_CMD_1 "cmd /bin/bash /opt/splunk/etc/apps/cribl/bin/timeout.sh &"
ENV SPLUNK_BEFORE_START_CMD_2 "cmd /bin/bash /opt/splunk/etc/apps/cribl/bin/intercom.sh"
ENV SPLUNK_BEFORE_START_CMD_3 "cmd /bin/bash /opt/splunk/etc/apps/cribl/bin/config.sh"
ADD http://cdn.cribl.io/dl/scope/latest/linux/libwrap.so /usr/lib/libwrap.so
RUN chmod 755 /usr/lib/libwrap.so
ADD scope.cfg /etc/scope/scope.cfg
RUN cp /var/opt/splunk/etc/splunk-launch.conf.default /var/opt/splunk/etc/splunk-launch.conf && \
    echo -e "\nLD_PRELOAD=/usr/lib/libwrap.so\n" >> /var/opt/splunk/etc/splunk-launch.conf
