FROM node

RUN mkdir -p /usr/local/scope/lib
WORKDIR /usr/local/scope/
RUN npm install hot-shots && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y vim gawk gdb emacs lsof tcpdump wget curl netcat && \
    ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

COPY metriccapture/scope-test /usr/local/scope/
COPY metriccapture/hotshot.ts /usr/local/scope/

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
