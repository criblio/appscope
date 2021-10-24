---
title: How AppScope Works
---

## How AppScope Works

AppScope extracts information by interposing functions. When an application calls a function, it actually calls a function of the same name in the AppScope libaray (`libscope`), which extracts details from the function. Then the original function call proceeds. This interposition of function calls requires no change to an application: it works with unmodified binaries, and its CPU and memory overhead is minimal.

### Performance Overhead
AppScope collects data with around 2% CPU overhead and **minimal** latency penalty. In most cases, the overhead is markedly less than 2%, which is significantly more efficient than legacy and current monitoring systems.


![AppScope system-level design](./images/AppScope_SysLvlDesign.png)
