---
title: Updating
---

# Updating

You can update AppScope in place. First, download a fresh version of the binary. (This example assumes that you are running AppScope from `/opt/appscope`, as described [here](/docs/downloading#where-from). If you're running AppScope from a different location, adapt the command accordingly.) 

```
curl -Lo /opt/appscope https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope && chmod 755 /opt/appscope
```

Then, to confirm the overwrite, verify AppScope's version and build date:

```
scope version
```
