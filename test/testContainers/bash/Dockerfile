FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y vim git gawk build-essential gdb emacs lsof tcpdump wget curl autoconf automake libtool && \
    apt-get clean

RUN git clone https://git.savannah.gnu.org/git/bash.git && \
    mkdir /mybin/ /mybin/nomem /mybin/mem && \
    cd bash && \
    ./configure && make && cp ./bash /mybin/mem/bash && \
    ./configure --without-bash-malloc && make && cp ./bash /mybin/nomem/bash

COPY ./bash/runbashmem.sh /mybin
COPY ./bash/runbashnomem.sh /mybin

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     ln -s /opt/appscope/bin/linux/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/bin/linux/ldscope /usr/local/scope/bin/ldscope && \
     ln -s /opt/appscope/lib/linux/libscope.so /usr/local/scope/lib/libscope.so

COPY ./bash/scope-test /usr/local/scope/

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
