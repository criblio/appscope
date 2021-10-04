FROM nginx:latest

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  curl \
  wget \
  emacs \
  gdb \
  net-tools \
  vim \
  apache2-utils \
  gnupg \
  libpq5 \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir /test_configs
COPY config/scope.yml /test_configs/scope/scope.yml
COPY config/nginx.conf /test_configs/nginx.conf
COPY config/nginx_1_worker.conf /test_configs/nginx_1_worker.conf
COPY ldscope /usr/bin/ldscope
COPY gdbinit /root/.gdbinit



