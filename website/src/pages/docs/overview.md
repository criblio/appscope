---
title: Overview
---

## What Is AppScope
---

AppScope is an open source instrumentation utility for any Linux command or application, regardless of programming language. It helps users explore, understand, and gain visibility with **NO code modification**. 

It provides fine-grained observability of a proxy/service mesh, without the latency of a sidecar and emits APM-like metric and event data, in open formats, to existing log and metric tools.

It’s like [strace](https://strace.io/) meets [tcpdump](https://www.tcpdump.org/), but with consumable output for events like file access, DNS, and network activity and StatsD-style metrics for applications. It can also look inside encrypted payloads for WAF-like visibility without proxying traffic. 
</br>
</br>


![AppScope in-terminal monitoring](./images/AppScope-GUI-screenshot.png)


## Features: Instrument, Collect and Observe
---

- Runtime-agnostic, no dependencies, no code development required, including static executables
- Capture application metrics: **File, Network, Memory, CPU**
- Capture application events: console content, stdin/out, logs, errors
- Capture any and all payloads: DNS, HTTP, and HTTPS
- Summarize metrics & detect protocols
- Normalize and forward metrics and events, in real-time to remote systems.
