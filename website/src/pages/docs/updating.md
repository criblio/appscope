---
title: Updating
---

## Updating

Changes in AppScope 1.0.0 affect some metric and event definitions, environment variables, and content (including default settings) in the `scope.yml` config file.

To update a Maintenance Pre-Release (version 0.8.1 or older) to version 1.0.0, follow this general procedure:

* Familiarize yourself with the [summary of changes](#summary-of-changes) below.

* If you use a config file, [download](https://github.com/criblio/appscope/blob/master/conf/scope.yml) the new config file from github, and read through it. 

* Make a note of changes that affect anything on which your deployment depends, including:
  * Defaults that have changed.
  * Metrics or events whose structure or names have changed.
  * Environment variables that you use outside of the config file, that need to be used differently than before.

* Based on what you discover:
  * Update how you (plan to) use environment variables.
  * Edit the new config file (if you are using one).
  * Configure any destinations to which you're sending data from AppScope, or scripts that operate on the data, to work with the changed event and/or metrics.

Then, go ahead and update!

### Performing the Update

You can update AppScope in place. First, download a fresh version of the binary. (This example assumes that you are running AppScope from `/opt/scope`, as described [here](/docs/downloading#where-from). If you're running AppScope from a different location, adapt the command accordingly.) 

```
curl -Lo /opt/scope https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope && chmod 755 /opt/scope
```

Then, to confirm the overwrite, verify AppScope's version and build date:

```
scope version
```

<span id="summary-of-changes"> </span>

### Summary of Changes in AppScope 1.0.0

This overview of changes should help you prepare to update AppScope.

#### Changes to Routing 

New routing capabilities in AppScope 1.0.0 required the following changes: 

- The `cribl` backend is now enabled by default, and its transport type now defaults to `edge` instead of `tcp`.
- A new environment variable, `SCOPE_CRIBL_ENABLE`, determines whether or not to send data to the `cribl` backend.
- When `SCOPE_CRIBL_ENABLE` is set to `false`, the default destination for the `ldscope` binary is `tcp`, as before.
- `SCOPE_CRIBL` or `SCOPE_CRIBL_CLOUD` now only define the `cribl` transport and destination.
- When the `edge` transport is used, `ldscope` and `scope` will search for a unix domain socket. The search tries the following locations, in order, only searching until a socket is found:  
  - `/var/run/appscope/appscope.sock`
  - `$CRIBL_HOME/state/appscope.sock`
  - `/opt/cribl/state/appscope.sock`
- In the CLI, the `-m` or `--metricdest`, `-e` or `--eventdest`, and `-c` or `--cribldest` options enable you target `edge`, `unix`, or `tcp`.
  - As before, the default destination for the CLI is the `scope` session directory in the local filesystem.

Learn more about routing [here](/docs/data-routing).

#### Changes to Events and Metrics

AppScope events and metrics are now rigorously defined in schemas found in `/docs/schemas/`. Metrics and events follow more consistent patterns overall. Changes that resolve inconsistencies include:

- All metric types and units are consistent.
- All durations are in milliseconds unless explicitly defined otherwise.
- Non-aggregated metrics no longer appear alongside aggregated metrics.
- All summarized metrics now have `class: summary`.
- Event dimension names are consistent, e.g., `file_name` is now `file`.
- `id` and `_channel` fields have been added, where previously missing.

- The data fields `net_peer_port` and `net_host_port` are now of type `integer`, for example:

```
{"type":"evt","id":"vm-curl-curl wttr.in","_channel":"118389477035177","body":{"sourcetype":"http","_time":1643997015.8988931,"source":"http.req","host":"vm","proc":"curl","cmd":"curl wttr.in","pid":112157,"data":{"http_method":"GET","http_target":"/","http_flavor":"1.1","http_scheme":"http","http_host":"wttr.in","http_user_agent":"curl/7.74.0","net_transport":"IP.TCP","net_peer_ip":"5.9.243.187","net_peer_port":80,"net_host_ip":"10.0.2.15","net_host_port":44214}}}
```

- The data field `http_status_code` used to be a string for HTTP/2 traffic. It is now an integer for both HTTP/1 and HTTP/2 traffic. For example:

```
`{"type":"evt","id":"vm-curl- --http2 -I https://nghttp2.org/","_channel":"118695329693352","body":{"sourcetype":"http","_time":1643997322.1907151,"source":"http.resp","host":"vm","proc":"curl","cmd":"curl --http2 -I https://nghttp2.org/","pid":112210,"data":{"http_flavor":"2.0","http_stream":1,"http_status_code":200,"http_status_text":"OK","http_response_content_length":6616,"net_transport":"IP.TCP","net_peer_ip":"139.162.123.134","net_host_ip":"10.0.2.15","net_peer_port":443,"net_host_port":35352,"http_client_duration":186,"http_host":"nghttp2.org","http_method":"HEAD","http_target":"/","http_user_agent":"curl/7.74.0”}}}`
```

Review the tables below to determine whether event or metric name changes affect your use of AppScope.

##### Event Names that Changed in AppScope 1.0.0

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

##### Metric Names that Changed in AppScope 1.0.0

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
