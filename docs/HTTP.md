# AppScope HTTP Support

The AppScope library can generate events and metrics for HTTP traffic it sees.
This developer doc introduces the the emitted message formats, points to where
the logic is implemented in the code, and explains the relevant runtime configs.

## HTTP Detection

The library emits events like the example below when HTTP protocol detection is
[enabled](#protocol-detection) and a matching network channel is found.

```text
{
  "type": "evt",
  "id": "dev01-curl-.1 http://nghttp2.org/robots.txt",
  "_channel": "694121168537773",
  "body": {
    "sourcetype": "net",
    "_time": 1632317484.042796,
    "source": "remote_protocol",
    "host": "dev01",
    "proc": "curl",
    "cmd": "curl -Lso /dev/null --http1.1 http://nghttp2.org/robots.txt",
    "pid": 1155928,
    "data": {
      "proc": "curl",
      "pid": 1155928,
      "fd": 6,
      "host": "dev01",
      "protocol": "HTTP"
    }
  }
}
```

See `doProtocol()` and `detectProtocol()` in [`src/state.c`](../src/state.c)
for the detection logic and `doDetection()` in [`src/report.c`](../src.report.c)
for the event formatting.

## HTTP Events

The AppScope library passes unencrypted HTTP payloads to the `doHttp()` function
in [`src/httpstate.c`](../src/httpstate.c) when HTTP events are
[enabled](#event-watch). Each chunk of the payload is passed to `doHttpBuffer()`
which handles detection of HTTP/1 vs HTTP/2.

For HTTP/1, `parseHttp1()` looks for complete header blocks that start with a
line containing `HTTP/` and end with `\r\n\r\n`. That chunk of text is posted to
the circular buffer in a `protocol_info` object by `reportHttp1()` with `evtype`
set to `EVT_PROTO` and `ptype` set to `EVT_REQ` (request) or `EVT_RESP`
(response).

For HTTP/2, `parseHttp2()` posts HEADERS, PUSH\_PROMISE, and CONTINUATION frames
using `reportHttp2()`. It uses the same `protocol_info` but sets `ptype` to
`EVT_H2FRAME`.

In the reporting thread, the `EVT_REQ` and `EVT_RESP` objects are passed to
`doHttp1Header()` in [`src/report.c`](../src/report.c) to generate the events
and metrics. The `EVT_H2FRAME` objects are sent to `doHttp2Frame()` which
generates the same events and metrics.

Both `doHttp1Header()` and `doHttp2Frame()` maintain the state of HTTP channels
in linked lists indexed by the channel ID. Details for the most recent request
message is kept for each HTTP/1 channel and HTTP/2 stream. When response
messages are seen, the corresponding request is available there.

### HTTP Request Events

The library emits an event like the example below for each HTTP/1 request header
block found. HTTP/2 HEADERS or PUSH\_PROMISE frames with a `:method` header
(combined with any CONTINUATION frames that follow) result in the same events
except the `http_flavor` value is `2.0`.

```text
{
  "type": "evt",
  "id": "dev01-curl-1 https://nghttp2.org/robots.txt",
  "_channel": "867516817554230",
  "body": {
    "sourcetype": "http",
    "_time": 1632490879.840472,
    "source": "http-req",
    "host": "dev01",
    "proc": "curl",
    "cmd": "curl --http1.1 https://nghttp2.org/robots.txt",
    "pid": 1730496,
    "data": {
      "http_method": "GET",
      "http_target": "/robots.txt",
      "http_flavor": "1.1",
      "http_scheme": "https",
      "http_host": "nghttp2.org",
      "http_user_agent": "curl/7.74.0",
      "net_transport": "IP.TCP",
      "net_peer_ip": "2400:8902::f03c:91ff:fe69:a454",
      "net_peer_port": "443",
      "net_host_ip": "2601:cf:4500:f481:27c7:7ea0:8bdd:35c0",
      "net_host_port": "53398"
    }
  }
}
```

See the [Event Watch](#event-watch) for details on how this event can be blocked
and how additional headers can be included.

### HTTP Response Events

The library emits an event like the example below for each HTTP/1 response
header block found. HTTP/2 HEADERS or PUSH\_PROMISE frames with a `:status`
header (combined with any CONTINUATION frames that follow) result in the same
events except the `http_flavor` value is `2.0`.

```text
{
  "type": "evt",
  "id": "dev01-curl-1 https://nghttp2.org/robots.txt",
  "_channel": "867516817554230",
  "body": {
    "sourcetype": "http",
    "_time": 1632490880.010622,
    "source": "http-resp",
    "host": "dev01",
    "proc": "curl",
    "cmd": "curl --http1.1 https://nghttp2.org/robots.txt",
    "pid": 1730496,
    "data": {
      "http_method": "GET",
      "http_target": "/robots.txt",
      "http_scheme": "https",
      "http_flavor": "1.1",
      "http_status_code": 200,
      "http_status_text": "OK",
      "http_server_duration": 169,
      "http_host": "nghttp2.org",
      "http_user_agent": "curl/7.74.0",
      "net_transport": "IP.TCP",
      "net_peer_ip": "2400:8902::f03c:91ff:fe69:a454",
      "net_peer_port": "443",
      "net_host_ip": "2601:cf:4500:f481:27c7:7ea0:8bdd:35c0",
      "net_host_port": "53398",
      "http_response_content_length": 62
    }
  }
}
```

A number of the fields in the response are taken from the previous response
message on the same HTTP/1 channel or HTTP/2 stream; specifically,
`http_method`, `http_target`, `http_host`, and `http_user_agent`. 

The `http_server_duration` field in the example above is the difference in the
timestamps of the request and response. It will be named with "server" if the
scoped process received the response; i.e. it's the HTTP server. If the process
sent the request (i.e. it's the client), the field is `http_client_duraction`
instead.

If the request included a `content-length` header, that value will be used for
an additional `http_request_content_length` field. That's not shown in the
example above.

See the [Event Watch](#event-watch) for details on how this event can be blocked
and how additional headers can be included.

## HTTP Metrics

The library emits metrics like the example below for HTTP activity.

```text
{
  "type": "metric",
  "body": {
    "_metric": "http.requests",
    "_metric_type": "counter",
    "_time": 1632776110.8551781,
    "_value": 1,
    "host": "dev01",
    "http_status_code": 200,
    "http_target": "/robots.txt",
    "pid": 2453848,
    "proc": "curl",
    "unit": "request"
  }
}
```

There are three HTTP metrics:
* `http.requests` counts request messages.
* `http.client.duration` and `http.server.duration` report the time (in
  milliseconds) between a request and its corresponding response message.
* `http.request.content_length` and `http.response.content_length` report
  the size (in bytes) of the message content.

## HTTP Runtime Configuration

### Protocol Detection

HTTP protocol detection events described [earlier](#http-detection) are
configured by an entry in `scope.yml`'s `protocol` list with the `name` entry
set to `HTTP`. The regex there is applied to the initial unencrypted payload on
each network channel and when it matches, the channel is flagged as HTTP. The
detection event will be emitted if the `detect` property is set in that entry.

If the `protocol` entry for HTTP is omitted, a built-in default is used with
detection events enabled and the regex below.

```text
 HTTP\\/1\\.[0-2]|PRI \\* HTTP\\/2\\.0\r\n\r\nSM\r\n\r\n
```

The library _always_ performs HTTP protocol detection on network channels.

### Event Watch

The runtime configs support an entry under `event > watch` to control the HTTP
event handling. By default it's enabled and configured like below.

```text
event:
  watch:
    - type: http
      name: .*
      field: .*
      value: .*
      headers:
```

The HTTP watch entry controls whether and how the library emits HTTP events.
Omit the entry uder `event > watch` to disable the HTTP events entirely.

The `name` regex is applied against the `source` value and only events that
match are emitted. Using something like `http-resp` would cause the library
to only emit the response events and block the request events.  The default is
`.*` with matches everything thus not blocking anything.

The `field` and `value` regexes are applied against the event's body.data
object and only properties that match are included in emitted events.
The defaults are `.*` with matches everything thus not filtering anything out.

By default emitted request and response events include `body.data` properties
derived from the following message fields and headers:

* content-length
* flavor
* host
* method
* scheme
* status
* target
* user-agent
* x-appscope
* x-forwarded-for

Additional headers from requests and responses will be included if their `name:
value` match any of the `headers` regexes in the watch. Note headers is a list
containing zero or more regex strings and match the whole header, not just the
name or value.

In the example below, the `name` regex blocks the `http-req` events and the
`headers` entries add the matching headers.

```text
event:
  watch:
    - type: http
      name: http-resp
      field: .*
      value: .*
      headers:
        - (?i)X-Forwarded.*  # include forwarding headers (case insensitive)
        - service: cribl.*   # include "service" headers starting with "cribl"
```
