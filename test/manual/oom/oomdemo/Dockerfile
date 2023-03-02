FROM ubuntu:latest
COPY main.c /opt/main.c

RUN apt update -y && \
    apt install -y gcc && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN gcc /opt/main.c -o /opt/oom_demo_process

CMD ["/opt/oom_demo_process"]
