---
title: Scoping a Running Process
---

Some aspects of scoping a running process are not self-evident.

To attach AppScope to a running process:

1. You must run `scope` as root, or with `sudo`.
1. If you attach to a shell, AppScope does not automatically scope its child processes.
1. You can attach to a process that is executing *in a container context*, but not a container context *from the host OS*.

*clarify* Events from I/O that exist before a process is attached are not emitted. For example, activity performed on any open file descriptor will not result in an AppScope event.

You cannot attach to a Musl Libc process or a process for a static executable.
