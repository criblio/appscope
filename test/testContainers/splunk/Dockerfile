FROM cribl/splunk:7.3.0



ARG CRIBL_DISTRO=cribl-splunk-app-linux-x64
RUN sh -c 'echo dash dash/sh boolean false | debconf-set-selections' && \
    sh -c 'DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash' && \
    echo "deb [check-valid-until=no] http://archive.debian.org/debian jessie-backports main" > /etc/apt/sources.list.d/jessie-backports.list && \
    sed -i '/deb http:\/\/deb.debian.org\/debian jessie-updates main/d' /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y vim nano curl ca-certificates jq

RUN apt-get install -y gcc make \
    build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

# manual installation of python 3.6 as default distro version is 3.4
RUN curl https://www.python.org/ftp/python/3.6.5/Python-3.6.5.tgz | tar xvz

RUN cd Python-3.6.5 \
    && ./configure --enable-optimizations --prefix=/usr \
    && make altinstall \
    && cd .. \
    && rm -r Python-3.6.5
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.6 get-pip.py \
    && rm -f get-pip.py
#removing all intermediate dependencies. All the stuff below comes up to 200+MB
RUN apt-get remove -y --purge gcc make build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev \
    libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
RUN rm -rf /var/lib/apt/lists/*

# RUN VERSION=$(curl -s https://cdn.cribl.io/versions.json | jq -r .version) && \
#     curl -Lso /tmp/${CRIBL_DISTRO}.tgz https://cdn.cribl.io/dl/$(echo ${VERSION} | cut -d '-' -f 1)/${CRIBL_DISTRO}-${VERSION}.tgz && \
RUN curl -Lso /tmp/${CRIBL_DISTRO}.tgz https://cdn.cribl.io/dl/latest/${CRIBL_DISTRO}.tgz && \
    tar -zxvf /tmp/${CRIBL_DISTRO}.tgz -C /var/opt/splunk/etc/apps && \
    rm /tmp/${CRIBL_DISTRO}.tgz

COPY splunk/hack/splunk/etc /var/opt/splunk/etc
COPY splunk/config.sh /var/opt/splunk/etc/apps/cribl/bin/config.sh
COPY splunk/cribldemo/ /var/opt/splunk/etc/apps/cribl/


RUN cp /var/opt/splunk/etc/splunk-launch.conf.default /var/opt/splunk/etc/splunk-launch.conf

ENV DEMOUSER_NAME "Demo User"
ENV DEMOUSER_EMAIL "demo@cribl.io"
ENV SPLUNK_USER root
ENV SPLUNK_START_ARGS "--accept-license --answer-yes --seed-passwd cribldemo"
ENV SPLUNK_BEFORE_START_CMD "version ${SPLUNK_START_ARGS}"
ENV SPLUNK_BEFORE_START_CMD_1 "cmd /bin/bash /opt/splunk/etc/apps/cribl/bin/timeout.sh &"
ENV SPLUNK_BEFORE_START_CMD_2 "cmd /bin/bash /opt/splunk/etc/apps/cribl/bin/intercom.sh"
ENV SPLUNK_BEFORE_START_CMD_3 "cmd /bin/bash /opt/splunk/etc/apps/cribl/bin/config.sh"

RUN mkdir /opt/test-runner/
RUN mkdir /opt/test-runner/logs/


ADD http://cdn.cribl.io/dl/scope/latest/linux/libscope.so /usr/lib/libscope.so
RUN chmod 755 /usr/lib/libscope.so
ENV SCOPE_OUT_DEST=udp://localhost:8125
ENV SCOPE_LOG_LEVEL=info
ENV SCOPE_LOG_DEST=file:///opt/test-runner/logs/scope.log
ENV SCOPE_OUT_VERBOSITY=4


COPY ./applications_test_runner/requirements.txt /opt/test-runner/requirements.txt
RUN pip3.6 install -r /opt/test-runner/requirements.txt

COPY ./applications_test_runner/ /opt/test-runner/

COPY splunk/entrypoint.sh /opt/test-runner/entrypoint.sh
RUN chmod 755 /opt/test-runner/entrypoint.sh


ENTRYPOINT ["/opt/test-runner/entrypoint.sh", "-l","/opt/test-runner/logs/"]
