---
title: Updating
---

## Updating

AppScope 1.0.0 makes changes to the `scope.yml` config file (including some default settings), as well as to some metric and event definitions. To update a Maintenance Pre-Release (version 0.8.1 or older) to version 1.0.0, follow this general procedure:

* Familiarize yourself with the new [config file](/docs/config-file) and the summary of changes below.
* Note where anything on which your deployment depends has changed, including:
  * Config file settings that are now in a different location than they were in the the old config file.
  * Defaults that have changed.
  * Environment variables that you use outside of the config file that need to be used differently than before.
  * Metrics or events whose structure or names have changed.
* Edit the new config file and update your notes on environment variable usage.
* Configure any destinations for AppScope data, or scripts that operate on the data, to work with the changed event and/or metrics.

 Then you should be ready to proceed with the update.

### Performing the Update

You can update AppScope in place. First, download a fresh version of the binary. (This example assumes that you are running AppScope from `/opt/scope`, as described [here](/docs/downloading#where-from). If you're running AppScope from a different location, adapt the command accordingly.) 

```
curl -Lo /opt/scope https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope && chmod 755 /opt/scope
```

Then, to confirm the overwrite, verify AppScope's version and build date:

```
scope version
```

### Summary of Changes in AppScope 1.0

The changes described here partly overlap with the changes you'll observe when you read through the new `scope.yml` config file. It's still worth the time to do that, though.

#### Changes to routing

- The cribl backend is now enabled by default.
- The cribl transport type is now `edge` by default instead of `tcp`.
- A new environment variable `SCOPE_CRIBL_ENABLE` determines whether or not to send data to cribl.
- `SCOPE_CRIBL` or `SCOPE_CRIBL_CLOUD` now only define the cribl transport and destination.
- When the `edge` transport is used, ldscope/scope will search for a unix domain socket in the following locations:
  - `/var/run/appscope/appscope.sock`
  - `$CRIBL_HOME/state/appscope.sock` (if above not found)
  - `/opt/cribl/state/appscope.sock` (if above not found)
 
Good to know:
- If you set `SCOPE_CRIBL_ENABLE=false`, the default destination for the ldscope binary will be `tcp`, as before.
- When using the scope binary, the default destination remains the filesystem (scope session directory).
- If you want to target `edge`, `unix`, or `tcp` from the scope binary, you can set the metric `dest/event` `dest/crib` dest protocol accordingly.

#### Changes to schema 

AppScope now conforms to an event and metric schema (found in `/docs/schemas/`). Changes to metrics and events produced include:

- `net_peer_port` and `net_host_port` are now of type `integer`, for example:

```
{"type":"evt","id":"vm-curl-curl wttr.in","_channel":"118389477035177","body":{"sourcetype":"http","_time":1643997015.8988931,"source":"http.req","host":"vm","proc":"curl","cmd":"curl wttr.in","pid":112157,"data":{"http_method":"GET","http_target":"/","http_flavor":"1.1","http_scheme":"http","http_host":"wttr.in","http_user_agent":"curl/7.74.0","net_transport":"IP.TCP","net_peer_ip":"5.9.243.187","net_peer_port":80,"net_host_ip":"10.0.2.15","net_host_port":44214}}}
```

- `http_status_code` used to be a string for http2. It is now an integer for http1 and http2. For example:

```
`{"type":"evt","id":"vm-curl- --http2 -I https://nghttp2.org/","_channel":"118695329693352","body":{"sourcetype":"http","_time":1643997322.1907151,"source":"http.resp","host":"vm","proc":"curl","cmd":"curl --http2 -I https://nghttp2.org/","pid":112210,"data":{"http_flavor":"2.0","http_stream":1,"http_status_code":200,"http_status_text":"OK","http_response_content_length":6616,"net_transport":"IP.TCP","net_peer_ip":"139.162.123.134","net_host_ip":"10.0.2.15","net_peer_port":443,"net_host_port":35352,"http_client_duration":186,"http_host":"nghttp2.org","http_method":"HEAD","http_target":"/","http_user_agent":"curl/7.74.0”}}}`
```

- “id” and “_channel” fields have been added, where previously missing, for consistency.
- All metric types and units are consistent.
- All durations are in milliseconds unless explicitly defined otherwise.
- Event dimension names are consistent, I.e. “file_name” became “file”.
- Summarized metrics now all have “class: summary”, whereas before this was inconsistent.
- Non-aggregated metrics no longer appear alongside aggregated metrics.
- Event names have been changed as follows:

Old | New
-- | --
`http-req` | `http.req`
`http-resp` | `http.resp`
`remote_protocol` | `net.app`
`net.dns.resp` | `dns.resp`
`net.dns.req` | `dns.req`
`net.dns.duration` | `dns.duration`
`fs.op.stat` | `fs.stat`
`fs.op.open` | `fs.open`
`fs.op.close` | `fs.close`
`fs.op.seek` | `fs.seek`
`net.conn_duration` | `net.duration`
`net.conn.open` | `net.open`
`net.conn.close` | `net.close`

- Metric names have been changed as follows:

Old | New
-- | --
`http.requests` | `http.req`
`http.server.duration` | `http.duration.server`
`http.client.duration` | `http.duration.client`
`net.dns` | `dns.req`
`net.dns.duration` | `dns.duration`
`fs.op.stat` | `fs.stat`
`fs.op.open` | `fs.open`
`fs.op.close` | `fs.close`
`fs.op.seek` | `fs.seek`
`net.conn.duration` | `net.duration`
`net.dns.duration` | `dns.duration`
`net.conn_duration` | `net.duration`
