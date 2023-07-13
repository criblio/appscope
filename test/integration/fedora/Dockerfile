FROM fedora:39

RUN dnf update -y && dnf install -y \
  gcc \
  ncurses-devel \
	&& dnf clean all

ENV PATH="/usr/local/scope:/usr/local/scope/bin:${PATH}"
COPY scope-profile.sh /etc/profile.d/scope.sh
COPY gdbinit /root/.gdbinit

RUN  mkdir /opt/test
COPY ./fedora/nsterm.c /opt/test/nsterm.c
RUN gcc -g -o /opt/test/nsterm /opt/test/nsterm.c -lncurses

RUN  mkdir /usr/local/scope && \
     mkdir /usr/local/scope/bin && \
     mkdir /usr/local/scope/lib && \
     mkdir -p /opt/test-runner/logs/ && \
     ln -s /opt/appscope/bin/linux/$(uname -m)/scope /usr/local/scope/bin/scope && \
     ln -s /opt/appscope/lib/linux/$(uname -m)/libscope.so /usr/local/scope/lib/libscope.so

# This is for GitHub runner
ENV TERM=xterm

COPY fedora/scope-test /usr/local/scope/scope-test

COPY docker-entrypoint.sh /
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["test"]
