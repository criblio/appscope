FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y vim git gawk build-essential && \
    git clone https://git.savannah.gnu.org/git/bash.git && \
    mkdir /mybin/ /mybin/nomem /mybin/mem && \
    cd bash && \
    ./configure && make && cp ./bash /mybin/mem/bash && \
    ./configure --without-bash-malloc && make && cp ./bash /mybin/nomem/bash

COPY ./bash/runbashmem.sh /mybin
COPY ./bash/runbashnomem.sh /mybin
COPY ./bash/test_bash.sh /mybin

#ENV LD_PRELOAD=/usr/lib/libscope.so

#ENTRYPOINT ["tail", "-f", "/dev/null"]
ENTRYPOINT ["/mybin/test_bash.sh"]

