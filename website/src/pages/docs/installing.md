---
title: Installing
---

## Installing
---

First, see [Requirements](/docs/requirements) to ensure that you’re completing these steps on a supported system. Next, getting started is easy.

### Get AppScope

You can download as a binary to run on your Linux OS, or as a container.

#### Download as Binary

Use this command to download the CLI binary and make it executable:

```
curl -Lo scope https://cdn.cribl.io/dl/scope/\
$(curl -L https://cdn.cribl.io/dl/scope/latest)/linux/scope && \
chmod 755 ./scope
```

That's it!

#### Download as Container

Visit the AppScope repo on Docker Hub to download and run the most recently tagged container:

https://hub.docker.com/r/cribl/scope

The container provides the AppScope binary on Ubuntu 20.04.

### Explore the CLI

- Run `scope --help` or `scope -h` to view CLI options.
- Try some basic CLI commands in [Using the CLI](/docs/quick-start-guide).
- See the complete [CLI Reference](/docs/cli-reference).