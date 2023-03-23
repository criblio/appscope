ARG GO_IMAGE_VER
FROM $GO_IMAGE_VER

RUN apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y bash-completion binutils bsdmainutils cgdb gcc

WORKDIR /opt/go

RUN mkdir /opt/go/output

COPY gosym/test.sh /opt/go/test.sh
RUN chmod +x /opt/go/test.sh

COPY gosym/gosym.c /opt/go/gosym.c

# this binary is used only for symbol detection
# Note: it is used as input for gosym binary
COPY gosym/gosymtest.go /opt/go/gosymtest.go

RUN cd /opt/go && \
    gcc -g -o gosym gosym.c && \ 
    go build -o gosymtest gosymtest.go

CMD /opt/go/test.sh
