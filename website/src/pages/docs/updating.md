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

### Required Edits to `scope.yml` When Updating to 0.7.5 or Later 

We have restructured `scope.yml` a bit in AppScope version 0.7.5. **You only need to care about this if you have defined custom tags.**

Here's how custom tags look in the pre-0.7.5 `scope.yml`:

```
metric:
  enable: true                      # true, false
  format:
    type : statsd                   # statsd, ndjson
    #statsdprefix : 'cribl.scope'    # prepends each statsd metric
    statsdmaxlen : 512              # max size of a formatted statsd string
    verbosity : 4                   # 0-9 (0 is least verbose, 9 is most)
    tags:
      user: $USER
      feeling: elation
  transport:                        # defines how scope output is sent
    type: udp                       # udp, tcp, unix, file, syslog
...
```

Notice that `tags` is a child of `format`, which in turn is a child of `metric`.

Beginning in version 0.7.5, `tags` is a top-level element, and thus a sibling of `metric`:

```
metric:
  enable: true                      # true, false
  format:
    type : statsd                   # statsd, ndjson
    #statsdprefix : 'cribl.scope'    # prepends each statsd metric
    statsdmaxlen : 512              # max size of a formatted statsd string
    verbosity : 4                   # 0-9 (0 is least verbose, 9 is most)
  transport:                        # defines how scope output is sent
    type: udp                       # udp, tcp, unix, file, syslog

tags:
  user: $USER
  feeling: elation
...
```
This means that before you update, you should save your custom tag text somewhere handy. Then after you update, edit the `tags` element in `scope.yml` to include your custom tags. Finally, restart AppScope so that the new configuration takes effect.
