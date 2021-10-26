---
title: Overview
---

## What Is AppScope

AppScope is an open source, runtime-agnostic instrumentation utility for any Linux command or application. It helps users explore, understand, and gain visibility with **no code modification**. 

AppScope provides the fine-grained observability of a proxy/service mesh, without the latency of a sidecar. It emits APM-like metric and event data, in open formats, to existing log and metric tools.

It’s like [strace](https://strace.io/) meets [tcpdump](https://www.tcpdump.org/) – but with consumable output for events like file access, DNS, and network activity, and StatsD-style metrics for applications. AppScope can also look inside encrypted payloads, offering [WAF](https://en.wikipedia.org/wiki/Web_application_firewall)-like visibility without proxying traffic. 
</br>
</br>

![AppScope in-terminal monitoring](./images/AppScope-GUI-screenshot.png)


## Features: Instrument, Collect, and Observe

- Runtime-agnostic, no dependencies, no code development required.
- Instrument both static and dynamic executables.
- Attach to processes *while they are running* or start when the process does.
- Run on Alpine Linux, another Linux distribution based on musl libc, or on a glibc-based distro.
- Communicate safely and securely with TLS over TCP connections.
- Capture application metrics: **File, Network, Memory, CPU**.
- Capture application events: console content, stdin/out, logs, errors.
- Capture any and all payloads: DNS, HTTP, HTTPS.
- Summarize metrics and detect protocols.
- Normalize and forward metrics and events, in real time, to remote systems.
