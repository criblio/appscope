FROM ubuntu:21.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  curl \
  wget \
  emacs \
  gdb \
  net-tools \
  vim \
  apache2-utils \
  nginx \
  gnupg \
  libpq5 \
  gcc \
  git \
  flex \
  bison \
  cmake \
  make \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /test_configs
#COPY fluent-bit /usr/bin/fluent-bit
COPY config/fluent-bit.conf /test_configs/fluent-bit.conf
COPY config/fluent-bit_multiple_inputs.conf /test_configs/fluent-bit_multiple_inputs.conf
COPY config/fluent-bit_to_file.conf /test_configs/fluent-bit_to_file.conf
COPY gdbinit /root/.gdbinit

RUN git clone https://github.com/criblio/fluent-bit
WORKDIR fluent-bit/build
RUN git checkout dev-appscope
RUN cmake ../
RUN make
RUN cp bin/fluent-bit /usr/bin/fluent-bit


#RUN wget -qO - https://packages.fluentbit.io/fluentbit.key | apt-key add -
#RUN echo "deb https://packages.fluentbit.io/ubuntu/bionic bionic main" > /etc/apt/sources.list 
#RUN apt update && apt install -y td-agent-bit
