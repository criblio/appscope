---
title: Working With AppScope
---

## Working With AppScope

AppScope offers two ways to work:

* Use the CLI when you want to explore in real-time, in an ad hoc way.

* Use the AppScope library (`libscope`) for longer-running, planned procedures. 

This is a guiding principle, not a strict rule. Sometimes you may prefer to plan out a CLI session, or, conversely, explore using the library.

What matters most is whether the command you want to scope can be changed while it is running. If it can, try the CLI; if not, go for the library.

For example:

* TBD example 1

* TBD example 2

* TBD example 3


Here's a decision tree to help you determine whether to use the CLI or the library.

![CLI vs. Library Decision Tree](./images/decision-tree.png)


### Env Vars, Flags, `ldscope`, and the Config File

AppScope's ease of use stems from its flexible set of controls:

* The AppScope library provides an extensive set of environment variables which control settings like TBD foo and bar.

* Although you cannot set environment variables in the CLI, the CLI does provide flags which override or substitute for certain environment variables. 

* AppScope's configuration file, `scope.yml`, can be invoked from either the CLI or the library.

* Finally, AppScope provides the `ldscope` utility, which offers a more convenient way to work with the library in some situations.

Let's see how to use these methods in their respective contexts, namely CLI and library usage.
