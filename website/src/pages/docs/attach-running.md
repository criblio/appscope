---
title: Scoping a Running Process
---

## Scoping a Running Process

Some aspects of scoping a running process are not self-evident.

To attach AppScope to a running process:

1. You must run `scope` as root, or with `sudo`.
1. If you attach to a shell, AppScope does not automatically scope its child processes.
1. You can attach to a process that is executing *in a container context*, but not a container context *from the host OS*.

When you attach AppScope to a process, its child processes are not automatically scoped.

You cannot attach to a musl libc process or a static executable's process.

No HTTP/1.1 events and headers are emitted when AppScope attaches to a Go process that uses the `azure-sdk-for-go` package.

No events are emitted from files or sockets that exists before AppScope attaches to a process.

- After AppScope attaches to a process, whatever file descriptors and/or socket descriptors that the process had already opened before that,
AppScope will not report any **new** activity on those file or socket descriptors.

  - For example, suppose a process opens a socket descriptor before AppScope is attached. Subsequent sends and receives on this socket will not produce AppScope events.

   - AppScope events will be produced only for sockets opened **after** AppScope is attached.
