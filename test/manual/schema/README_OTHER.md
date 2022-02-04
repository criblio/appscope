# This defines how to create other events & metrics missing from syscall tests in json format for use in schema validation

## `net.other` event (based on seqpacket tests)

```

gcc -g test/manual/seqpacket/server.c -o seq_server
gcc -g test/manual/seqpacket/client.c -o seq_client

SCOPE_EVENT_METRIC=true scope run -- ./seq_server &
./seq_client 3 4
./seq_client DOWN
grep -m 1 "net.other" ~/.scope/history/seq_server_xxx/events.json > appscope/test/manual/schema/net.other.evt

```
## `net.other` metric (based on seqpacket tests)

```

gcc -g test/manual/seqpacket/server.c -o seq_server
gcc -g test/manual/seqpacket/client.c -o seq_client

SCOPE_METRIC_VERBOSITY=9 scope run -- ./seq_server &
./seq_client 3 4
./seq_client 5 12
./seq_client DOWN
grep -m 1 net.other  ~/.scope/history/seq_server_xxx/metrics.json  > appscope/test/manual/schema/mt_net.other.evt

```

## `http.request.content_length` metric

```

scope run -- curl -X POST https://reqbin.com/echo/post/json -H "Content-Type: application/json" -d '{"productId": 123456, "quantity": 100}'
grep -m 1 content_length  ~/.scope/history/curl_xxx/metrics.json > appscope/test/manual/schema/mt_content_length

```

## `notice` event

The easiest way to trigger this event is to modify the source code - `evtFormatHelper` function:

```
    // rate limited to maxEvtPerSec
    if (evt->ratelimit.maxEvtPerSec == 0) {
        ; // no rate limiting.
    } else if (tv.tv_sec != evt->ratelimit.time) {
        evt->ratelimit.time = tv.tv_sec;
        evt->ratelimit.evtCount = evt->ratelimit.notified = 0;
    } else if (++evt->ratelimit.evtCount >= evt->ratelimit.maxEvtPerSec) {
        // one notice per truncate
        if (evt->ratelimit.notified == 0) {
            cJSON *notice = rateLimitMessage(proc, src, evt->ratelimit.maxEvtPerSec);
            evt->ratelimit.notified = (notice)?1:0;
            return notice;
        }
    }
```

to

```
    // rate limited to maxEvtPerSec
 //   if (evt->ratelimit.maxEvtPerSec == 0) {
 //       ; // no rate limiting.
 //   } else if (tv.tv_sec != evt->ratelimit.time) {
 //       evt->ratelimit.time = tv.tv_sec;
 //       evt->ratelimit.evtCount = evt->ratelimit.notified = 0;
 //  } else if (++evt->ratelimit.evtCount >= evt->ratelimit.maxEvtPerSec) {
        // one notice per truncate
 //       if (evt->ratelimit.notified == 0) {
            cJSON *notice = rateLimitMessage(proc, src, evt->ratelimit.maxEvtPerSec);
            evt->ratelimit.notified = (notice)?1:0;
            return notice;
 //       }
 //   }

```

```

scope run -- curl wttr.in
grep -m 1 notice  ~/.scope/history/curl_xxx/events.json > appscope/test/manual/schema/notice.evt

```
