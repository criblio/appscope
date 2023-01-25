# AppScope Debug Design

When a process crashes it can be a challenge to get enough information to help developers troubleshoot the situation. Some examples of the difficulties... it can be hard to reproduce the crash outside the environment where the crash was first seen. Container environments are typically very transient so it can be tough to obtain shell access. Container storage that might give clues many not persist after the container has exited. Installation of debugging tools may be challenging or undesireable. Even outside of containers the procedure to turn on kernel-provided core dumps often varies by linux distro and can be a security liability (particularly with DOS attacks). It can even be challenging to know whether AppScope is playing a role in the crash itself.

As of today, AppScope has one mechanism to report debugging information in the event of a crash. If the environment variable SCOPE_ERROR_SIGNAL_HANDLER is set to "true", and the AppScope library detects one of a number of signals that will result in the abnormal termination of the host process, it can output a stack trace of the offending thread to the AppScope log file before terminating. This doc is intended to be a discussion of how we might improve and extend AppScope's error handling.

There are three things that we expect to be common with any approach to debugging crashes. Firstly, there has to be a way to detect that there is a problem. Secondly, we have to take a snapshot of pertinent information while the problem exists. Lastly, we need to preserve the snapshot so the debugging can take place at a later time. At this time we will not consider the additonal steps of notifying developers or transferring the snapshot to the developers.

It seems like there are two very different approaches to each of these three steps. Should the work of each step be done by the AppScope library inside the process, or from a seperate daemon or process that lives outside of any scoped process.

To talk about detection from the perspective of our library inside the process: we can register for signals that will result in abnormal termination of the process we're in. One concern is that it is possible that the process we're in has already registered handlers for signals that we are also interested in. Our current approach is to not register our handlers by default, but only register our handlers when instructed to. For crashes that happen very infrequently it may be frustrating to need to restart the process with different settings. It may be possible to try to register our handlers, call them first, and then call any handlers the process installs. This is more complex, and we may not understand possible timing windows with this approach.

To talk about detection from the perspective of an external process or daemon:
We could launch a process (daemon) to detect unhealthy processes. Two examples of how this could be done include using eBPF to observe signals, or pinging the process and detecting timeouts. We believe that external detection alone can not respond quickly enough to preserve the state of the crashing process.

To talk about snapshotting from our library inside a process: we have libunwind implemented today, and have a solid start on coredumper. The primary coredumper repo doesn't directly support aarch64 nor musl-based systems. We're investigating whether this has been resolved on other forks, or if existing PRs may provide a solution to this. As a negative of performing the snapshot from inside the library, it's a very [constrained](https://man7.org/linux/man-pages/man7/signal-safety.7.html) operating environment. As an example, our new [IPC mechanism](https://github.com/criblio/appscope/issues/1108) is powerful, but depends on a lot of functionality that is not available from inside a signal handler. As a positive, code here can be made to be portable, and avoid any external (dynamically loaded) libraries or applications.

For snapshots taken from outside the library: from the discussion above about crash detection, it would require some help from the library to freeze the process (aka "stop the world") in a state where we can snapshot it from outside the process. Anything done from outside the crashing process has the clear benefit of being free of the severe signal-safe constraints. We don't yet know if there are existing go libraries that may allow us to generate a core dump from userspace (perform an operation equivalent to gcore or gdb's "generate-core-file" functionality). Without knowing about applicable go libraries, it's possible that it might be challenging to perform this function. Bundling one of these utilites is not super attractive, but is a possibility.

To preserve the snapshot it seems like it will take cooperation of the library and cli. Our initial thought is to to write a file that contains our snapshot to someplace on the filesystem, like /tmp or /dev/shm. At this point it seems important to "stop the world" at least when running in a container environment. The concern about exiting is that if we're running in the foreground as process 1 in a container, as is typical, then if we allow the process to terminate, we expect the container to terminate as well.  It's possible to send a signal from the crashing process to the cli after the snapshot processing is complete then stop the world via raising a SIGSTOP. At this point the cli can navigate the containers as needed to retreive the snapshot.  When it's done gathering the snapshot, it could send a SIGCONT to the crashing process.  Admittedly, this seems fragile, and requires that we have the cli running as a service or daemon for it's role in this scheme. As another important comment, it's unclear if or how any library behavior we add in these fatal crashes could be reconciled with any existing crash handling that the host process might have been trying to do.

It's also possible that we could send this information (meaning socket and send are signal-safe), provided a destination (cli?) is available and that we don't need to resolve DNS (not signal-safe) for example.  At this point...


How to we want to gather it?  Internal, External


Attach?
Arch?
Musl?
Containers?
What more than core/stack traces?  When to capture more?  Where is this captured?
