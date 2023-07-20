---
title: Application Crash Analysis with AppScope
---

## Application Crash Analysis with AppScope

Here you will learn how to use AppScope and eBPF technology to:

- Generate useful files when an application crashes (backtrace, core dump, snapshot)
- Send those files to a network destination, before the environment is torn down

You will need:

- A Linux host
- Docker

For example’s sake we will:
- Use nginx as our crashing process
  - Any process is fine.
- Run nginx in a docker container
  - On the host would be fine too.
- Use Cribl Cloud as our network destination, and route our crash data to S3
  - Any network destination is fine… or just view the files on disk


### Preparation

Set up your Cribl Cloud instance:
```
TBD
```

Download the AppScope binary:
```
cd ~/Downloads  
curl -Lo scope https://cdn.cribl.io/dl/scope/1.4.1/linux/$(uname -m)/scope
curl -Ls https://cdn.cribl.io/dl/scope/1.4.1/linux/$(uname -m)/scope.md5 | md5sum -c 
chmod +x scope
```

Download and build the `scope-ebpf` project:
_Tip: eBPF code falls under a different license to the AppScope project, hence the separate repository_
```
git clone git@github.com:criblio/scope-ebpf.git
cd scope-ebpf
make all
```

Deploy the eBPF module to listen for crashing applications:
```
sudo ~/Downloads/scope-ebpf/bin/scope-ebpf &
```

Deploy the AppScope daemon, to receive messages from eBPF and send crash files to a network destination
```
sudo ~/Downloads/scope daemon --filedest tcp://<path-to-cribl-cloud-tcp>:<port>
```

Our preparation is complete! The eBPF kernel module should be loaded, listening for application crash signals. The AppScope Daemon should be running, waiting for the eBPF module to tell it that an application has crashed. Once it receives that notification, it will look for crash files and send them on to the configured network destination.


### Crash Investigation

Start an nginx container:
```
docker run --rm nginx
```

Attach AppScope to the containerized nginx process, and enable backtrace and core dump on crash:
_Tip: AppScope can be loaded into an application when it starts, or after it has started, with attach._
```
sudo ~/Downloads/scope attach --backtrace --coredump nginx
# choose the master process from the list
```

Force a crash of nginx by sending the process a BUS error signal:
_Tip: AppScope will respond to SIGBUS, SIGINT, SIGSEGV, SIGFPE signals_
```
sigkill -s SIGBUS nginx 
```

In your S3 bucket you should be able to see 3 files (info, backtrace, snapshot) that give meaningful insight into the crash.

You can also inspect the crash files, including the core dump, in /tmp/appscope/<pid of nginx>/

