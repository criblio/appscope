### Signal Delivery
In support of a crash analysis feature, we started with signal delivery. We want to be informed when an application experiences a failure that results in an abort. We start by filtering on 4 signals: SIGSEGV, SIGBUS, SIGILL and SIGFPE.
### Go
We want to use libbpf to create the eBPF code. Next, we wanted to determine if Go packages supporting eBPF seemed reasonable and useful. We start with the Go package github.com/cilium/ebpf. It seems reasonable and appears to work. 
### Tracepoint
Lots of ways to link to kernel execution using eBPF. Looking specifically at signals, it's clear that trace points are required. Available trace points are defined in /sys/kernel/debug/tracing/events. From that, it's clear we want signal/signal_deliver.
### Interface
The params passed to the signal_deliver trace point are defined in /sys/kernel/tracing/events/signal/signal_deliver/format. An object is defined and populated by the eBPF kernel code. The Go code must have the requisite object definition in order to read this detail. This object definition and the trace point definition are the primary connections between kernel code and the user mode Go code.
### Maps
A map, in eBPF terminology, is the generic storage for different types of data sharing between kernel and user mode. There are several map types. We want to use the map type associated with the trace point we are using. In this case the map is a performance event array.
### Section Name
There are at least 3 sections that need to be defined for eBPF code running in the kernel. The sections are .maps, version and trace point. These are defined with a compiler  `__attribute__ `directive. These are required in the code loaded in the kernel. 
### Signal Delivery Stops
We experienced an issue where signal delivery stopped after a period of time. We thought it could have been related to buffer overflow and interface between kernel and user mode code. That was not the case. The issues result from the fact that the kernel is able to both unload the eBPF module and disconnect it from an app. This occurs when there are no references to specific objects. The kernel is able to determine that the module is not being used and is free to unload and or disconnect. Object references are removed when the Go runtime garbage collector runs and there is no reference to a relative object. In this case GC removes the reference from the app and the kernel unloads or disconnects, depending on object type. 
### Object Reference and Unload of a Module
The first case discovered was the the unloading of the eBPF module. When we referenced the program and map objects by deferring a close to these objects, the eBPF module was no longer unloaded. 
### Object Reference & BPF Program Connection
The next case discovered was the disconnect of the eBPF module from the Go application. This is a bit less obvious. In summary, the link to the trace point needs to be maintained. Again, deferring a close to the link object resolves this rather odd issue. 
### Filter
By definition, event filters can be applied at the kernel level. Refer to [filters](https://www.kernel.org/doc/html/v5.0/trace/events.html#setting-filters) for details. There are 3 places where a filter for the signal_delivery trace point can be defined. We are not using these. Had issues with the definitions not working as expected. Given that the filtering behavior is very simple, we applied it to the kernel code. 
### Printk
The only thing to know about bpf_printk() is how to view the output. Just do this:
`sudo cat /sys/kernel/tracing/trace_pipe`

### Reading Kernel Objects
Access to kernel objects is available. The function `bpf_probe_read_kernel()` is used to read from a kernel object and update eBPF accessible memory. It's important to note that the read of an object starts with the read of the pointer to the object. For example, to read the pid structure from the current task structure, read the pointer to pid:

`struct pid *pid;`

`bpf_probe_read_kernel(&pid, sizeof(pid), &task->thread_pid);`

In turn, to access a scalar object within an object read directly:

`unsigned in level;`

`bpf_probe_read_kernel(&level, sizeof(level), &pid->level);`


### Kernel Objects
[![](https://mermaid.ink/img/pako:eNp1Uk1PwzAM_SuRTyBtf6AHJLaWG6ft1qDKJGat1iRVPkBo23_HSweZppFD8my_96wkPoBymqCCncepF9taWsHruQ3RJxVFxLDvZiylnQbN-6XEgYi9J9QdwzexXD6J1UMpMlOM9EljkaSzxibzTj48zp1WWXfcfrmjWLdXPFZZ_2-3WbzO4rotnM6ioTChoquugXzJ51DYUOo2dMoZ4yyji3GdjZv2rgMrE3cyODHa_aGbu95xbbLrS3vLYNHAr8KHcsnGzIYFGPIGB82fczhnJMSeDEmoGGr0ewnSnpiXJo2RGj1E56Fib1oApug231b9xjOnHpD_2UD1gWPgLGXN6zwBeRBOPwXmvKk?type=png)](https://mermaid.live/edit#pako:eNp1Uk1PwzAM_SuRTyBtf6AHJLaWG6ft1qDKJGat1iRVPkBo23_HSweZppFD8my_96wkPoBymqCCncepF9taWsHruQ3RJxVFxLDvZiylnQbN-6XEgYi9J9QdwzexXD6J1UMpMlOM9EljkaSzxibzTj48zp1WWXfcfrmjWLdXPFZZ_2-3WbzO4rotnM6ioTChoquugXzJ51DYUOo2dMoZ4yyji3GdjZv2rgMrE3cyODHa_aGbu95xbbLrS3vLYNHAr8KHcsnGzIYFGPIGB82fczhnJMSeDEmoGGr0ewnSnpiXJo2RGj1E56Fib1oApug231b9xjOnHpD_2UD1gWPgLGXN6zwBeRBOPwXmvKk)


### Scope Daemon in a Container
The scope daemon command installs an eBPF module and listens for events on a corresponding eBPF map. There are a couple of requirements needed to utilize eBPF in a container. A libbpf package needs to be installed and a mount point is required when the container is started.

#### Example Dockerfile for Ubuntu 20 & 22:

`FROM ubuntu:latest`

`RUN apt update && apt install -y libbpf0`

#### Example Dockerfile for Ubuntu 18
The libbpf tools are not available as packages for Ubuntu 18. Therefore, we have pulled the repo and built from source. This approach is used in the build container and seems to work.

`FROM ubuntu:18.04`

`RUN git clone https://github.com/libbpf/libbpf.git && cd libbpf/src && make && make install`


#### Example Docker build command:

`docker build . -t bpfcon`


#### Example docker run command:

`docker run -d -it --rm --cap-add=SYS_ADMIN -v <path_where_scope_exists>/:/opt/appscope --name bcon -v /sys/kernel/debug:/sys/kernel/debug:ro bpfcon /opt/appscope/scope daemon`

In the example run command a path where the scope executable exists is used. Not required. Just makes it easier.
