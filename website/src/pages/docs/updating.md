---
title: Updating
---

## Updating

You can update AppScope in place. First, download a fresh version of the binary:

```
cd <your-appscope-directory>
curl -Lo scope https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope && chmod 755 ./scope
```

Then, to confirm the overwrite, verify AppScope's version and build date:

```
scope version
```
