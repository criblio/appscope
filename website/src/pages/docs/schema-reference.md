---
title: Schema Reference
---

# Schema Reference

In AppScope, events are structured according to one pattern, and metrics are structured according to another. These patterns are defined rigorously, in validatable [JSON Schema](https://json-schema.org/). 

Three [definitions schemas](https://github.com/criblio/appscope/tree/master/docs/schemas/definitions) govern the basic patterns. Then there is an individual schema for each event and metric, documented below. The definitions schemas define the elements that can be present in individual event and metric schemas, as well as the overall structures into which those elements fit. 

When we say "the AppScope schema," we mean the [whole set](https://github.com/criblio/appscope/tree/master/docs/schemas/) of schemas.

In AppScope 1.0.0, a few event and metric schema elements, namely `title` and `description`, have placeholder values. In future these may be made more informative. They are essentially "internal documentation" within the schemas and do not affect how the schemas function in AppScope. In the event that you develop any code that depends on AppScope schemas, be aware that the content of `title` and `description` fields may evolve.

<div class="toc-grid">
<div class="toc-cell">

## Events

**File System**

- [fs.open](#eventfsopen)
- [fs.close](#eventfsclose)
- [fs.duration](#eventfsduration)
- [fs.error](#eventfserror)
- [fs.read](#eventfsread)
- [fs.write](#eventfswrite)
- [fs.delete](#eventfsdelete)
- [fs.seek](#eventfsseek)
- [fs.stat](#eventfsstat)

**Network**

- [net.open](#eventnetopen)
- [net.close](#eventnetclose)
- [net.duration](#eventnetduration)
- [net.error](#eventneterror)
- [net.rx](#eventnetrx)
- [net.tx](#eventnettx)
- [net.app](#eventnetapp)
- [net.port](#eventnetport)
- [net.tcp](#eventnettcp)
- [net.udp](#eventnetudp)
- [net.other](#eventnetother)

**HTTP**

- [http.req](#eventhttpreq)
- [http.resp](#eventhttpresp)

**DNS**

- [dns.req](#eventdnsreq)
- [dns.resp](#eventdnsresp)

**File**

- [file](#eventfile)

**Console**

- [console](#eventconsole)

**System Notification**

- [notice](#eventnotice)

</div>
<div class="toc-cell">

## Metrics

**File System**

- [fs.open](#metricfsopen)
- [fs.close](#metricfsclose)
- [fs.duration](#metricfsduration)
- [fs.error](#metricfserror)
- [fs.read](#metricfsread)
- [fs.write](#metricfswrite)
- [fs.seek](#metricfsseek)
- [fs.stat](#metricfsstat)

**Network**

- [net.open](#metricnetopen)
- [net.close](#metricnetclose)
- [net.duration](#metricnetduration)
- [net.error](#metricneterror)
- [net.rx](#metricnetrx)
- [net.tx](#metricnettx)
- [dns.req](#metricdnsreq)
- [net.port](#metricnetport)
- [net.tcp](#metricnettcp)
- [net.udp](#metricnetudp)
- [net.other](#metricnetother)

**HTTP**

- [http.req](#metrichttpreq)
- [http.request.content_length](#metrichttpreqcontentlength)
- [http.response.content_length](#metrichttprespcontentlength)
- [http.duration.client](#metrichttpdurationclient)
- [http.duration.server](#metrichttpdurationserver)

**Process**

- [proc.fd](#metricprocfd)
- [proc.thread](#metricprocthread)
- [proc.start](#metricprocstart)
- [proc.child](#metricprocchild)
- [proc.cpu](#metricproccpu)
- [proc.cpu.perc](#metricproccpuperc)
- [proc.mem](#metricprocmem)

</div>
</div>

<hr/>

### dns.req [^](#schema-reference) {#eventdnsreq}

Structure of the `dns.req` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-firefox-/usr/lib/firefox/firefox",
  "_channel": "13470757294558",
  "body": {
    "sourcetype": "dns",
    "_time": 1643735942.526987,
    "source": "dns.req",
    "host": "ubuntu",
    "proc": "firefox",
    "cmd": "/usr/lib/firefox/firefox",
    "pid": 6544,
    "data": {
      "domain": "detectportal.firefox.com"
    }
  }
}
```

#### `dns.req` properties {#eventdnsreqprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventdnsreqbody)._ |

#### `dns.req.body` properties {#eventdnsreqbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - dns<br/><br/>Value must be `dns`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - DNS request<br/><br/>Value must be `dns.req`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventdnsreqbodydata)._ |

#### `dns.req.body.data` properties {#eventdnsreqbodydata}

| Property | Description |
|---|---|
| `domain` _required_ (`string`) | domain |

<hr/>

### dns.resp [^](#schema-reference) {#eventdnsresp}

Structure of the `dns.resp` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-firefox-/usr/lib/firefox/firefox",
  "_channel": "13470823778038",
  "body": {
    "sourcetype": "dns",
    "_time": 1643735942.552667,
    "source": "dns.resp",
    "host": "ubuntu",
    "proc": "firefox",
    "cmd": "/usr/lib/firefox/firefox",
    "pid": 6544,
    "data": {
      "duration": 25,
      "domain": "detectportal.firefox.com",
      "addrs": [
        "34.107.221.82"
      ]
    }
  }
}
```

#### `dns.resp` properties {#eventdnsrespprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventdnsrespbody)._ |

#### `dns.resp.body` properties {#eventdnsrespbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - dns<br/><br/>Value must be `dns`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - DNS response<br/><br/>Value must be `dns.resp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventdnsrespbodydata)._ |

#### `dns.resp.body.data` properties {#eventdnsrespbodydata}

| Property | Description |
|---|---|
| `duration` (`number`) | duration<br/><br/>**Example:**<br/>`55` |
| `domain` (`string`) | domain |
| `addrs` (`array`) | addrs |

<hr/>

### fs.close [^](#schema-reference) {#eventfsclose}

Structure of the `fs.close` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890090429747",
  "body": {
    "sourcetype": "fs",
    "_time": 1643735835.455002,
    "source": "fs.close",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "proc": "accept01",
      "pid": 1933,
      "host": "8bc1398c19f3",
      "file": "/dev/shm/ltp_accept01_1931",
      "proc_uid": 0,
      "proc_gid": 0,
      "proc_cgroup": "0::/system.slice/containerd.service",
      "file_perms": 600,
      "file_owner": 0,
      "file_group": 0,
      "file_read_bytes": 0,
      "file_read_ops": 0,
      "file_write_bytes": 0,
      "file_write_ops": 0,
      "duration": 0,
      "op": "close"
    }
  }
}
```

#### `fs.close` properties {#eventfscloseprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsclosebody)._ |

#### `fs.close.body` properties {#eventfsclosebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - fs<br/><br/>Value must be `fs`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Close<br/><br/>Value must be `fs.close`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsclosebodydata)._ |

#### `fs.close.body.data` properties {#eventfsclosebodydata}

| Property | Description |
|---|---|
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` (`string`) | host |
| `file` (`string`) | file |
| `proc_uid` (`integer`) | proc_uid<br/><br/>**Example:**<br/>`0` |
| `proc_gid` (`integer`) | proc_gid<br/><br/>**Example:**<br/>`0` |
| `proc_cgroup` (`string`) | proc_cgroup<br/><br/>**Example:**<br/>`0::/user.slice/user-1000.slice/session-3.scope` |
| `file_perms` (`integer`) | file_perms<br/><br/>**Example:**<br/>`777` |
| `file_owner` (`number`) | file_owner<br/><br/>**Example:**<br/>`0` |
| `file_group` (`number`) | file_group<br/><br/>**Example:**<br/>`0` |
| `file_read_bytes` (`integer`) | file_read_bytes<br/><br/>**Example:**<br/>`512` |
| `file_read_ops` (`integer`) | file_read_ops<br/><br/>**Example:**<br/>`5` |
| `file_write_bytes` (`integer`) | file_write_bytes<br/><br/>**Example:**<br/>`10` |
| `file_write_ops` (`integer`) | file_write_ops<br/><br/>**Example:**<br/>`5` |
| `duration` (`number`) | duration<br/><br/>**Example:**<br/>`55` |
| `op` (`string`) | op_fs_close<br/><br/>**Possible values:**<ul><li>`go_close`</li><li>`closedir`</li><li>`freopen`</li><li>`freopen64`</li><li>`close`</li><li>`fclose`</li><li>`close$NOCANCEL`</li><li>`guarded_close_np`</li><li>`close_nocancel`</li></ul> |

<hr/>

### fs.delete [^](#schema-reference) {#eventfsdelete}

Structure of the `fs.delete` event

#### Example

```json
{
  "type": "evt",
  "id": "b6209181773f-rm-rm test.txt",
  "_channel": "none",
  "body": {
    "sourcetype": "fs",
    "_time": 1643793922.040438,
    "source": "fs.delete",
    "host": "b6209181773f",
    "proc": "rm",
    "cmd": "rm test.txt",
    "pid": 306,
    "data": {
      "proc": "rm",
      "pid": 306,
      "host": "b6209181773f",
      "op": "unlinkat",
      "file": "test.txt",
      "unit": "operation"
    }
  }
}
```

#### `fs.delete` properties {#eventfsdeleteprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsdeletebody)._ |

#### `fs.delete.body` properties {#eventfsdeletebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - fs<br/><br/>Value must be `fs`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Delete<br/><br/>Value must be `fs.delete`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsdeletebodydata)._ |

#### `fs.delete.body.data` properties {#eventfsdeletebodydata}

| Property | Description |
|---|---|
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` (`string`) | host |
| `op` (`string`) | op_fs_delete<br/><br/>**Possible values:**<ul><li>`unlink`</li><li>`unlinkat`</li></ul> |
| `file` (`string`) | file |
| `unit` (`string`) | Unit - operation<br/><br/>Value must be `operation`. |

<hr/>

### fs.duration [^](#schema-reference) {#eventfsduration}

Structure of the `fs.duration` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890091215105",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.455057,
    "source": "fs.duration",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "_metric": "fs.duration",
      "_metric_type": "histogram",
      "_value": 12,
      "proc": "accept01",
      "pid": 1933,
      "fd": 3,
      "op": "fgets_unlocked",
      "file": "/etc/passwd",
      "numops": 1,
      "unit": "microsecond"
    }
  }
}
```

#### `fs.duration` properties {#eventfsdurationprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsdurationbody)._ |

#### `fs.duration.body` properties {#eventfsdurationbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Duration<br/><br/>Value must be `fs.duration`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsdurationbodydata)._ |

#### `fs.duration.body.data` properties {#eventfsdurationbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - File Duration<br/><br/>Value must be `fs.duration`. |
| `_metric_type` (`string`) | histogram<br/><br/>Value must be `histogram`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `op` (`string`) | op |
| `file` (`string`) | file |
| `numops` (`number`) | numops |
| `unit` (`string`) | Unit - microsecond<br/><br/>Value must be `microsecond`. |

<hr/>

### fs.error [^](#schema-reference) {#eventfserror}

Structure of the `fs.error` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890094642989",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.45777,
    "source": "fs.error",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1931,
    "data": {
      "_metric": "fs.error",
      "_metric_type": "counter",
      "_value": 1,
      "proc": "accept01",
      "pid": 1931,
      "op": "access",
      "file": "/dev/shm/ltp_accept01_1931",
      "class": "stat",
      "unit": "operation"
    }
  }
}
```

#### `fs.error` properties {#eventfserrorprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfserrorbody)._ |

#### `fs.error.body` properties {#eventfserrorbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Error<br/><br/>Value must be `fs.error`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfserrorbodydata)._ |

#### `fs.error.body.data` properties {#eventfserrorbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - File Error<br/><br/>Value must be `fs.error`. |
| `_metric_type` (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `op` (`string`) | op |
| `file` (`string`) | file |
| `class` (`string`) | class fs.error<br/><br/>**Possible values:**<ul><li>`open_close`</li><li>`read_write`</li><li>`stat`</li></ul> |
| `unit` (`string`) | Unit - operation<br/><br/>Value must be `operation`. |

<hr/>

### fs.open [^](#schema-reference) {#eventfsopen}

Structure of the `fs.open` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890090429747",
  "body": {
    "sourcetype": "fs",
    "_time": 1643735835.454946,
    "source": "fs.open",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "proc": "accept01",
      "pid": 1933,
      "host": "8bc1398c19f3",
      "file": "/dev/shm/ltp_accept01_1931",
      "proc_uid": 0,
      "proc_gid": 0,
      "proc_cgroup": "0::/system.slice/containerd.service",
      "file_perms": 600,
      "file_owner": 0,
      "file_group": 0,
      "op": "open"
    }
  }
}
```

#### `fs.open` properties {#eventfsopenprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsopenbody)._ |

#### `fs.open.body` properties {#eventfsopenbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - fs<br/><br/>Value must be `fs`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File open<br/><br/>Value must be `fs.open`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsopenbodydata)._ |

#### `fs.open.body.data` properties {#eventfsopenbodydata}

| Property | Description |
|---|---|
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` (`string`) | host |
| `file` (`string`) | file |
| `proc_uid` (`integer`) | proc_uid<br/><br/>**Example:**<br/>`0` |
| `proc_gid` (`integer`) | proc_gid<br/><br/>**Example:**<br/>`0` |
| `proc_cgroup` (`string`) | proc_cgroup<br/><br/>**Example:**<br/>`0::/user.slice/user-1000.slice/session-3.scope` |
| `file_perms` (`integer`) | file_perms<br/><br/>**Example:**<br/>`777` |
| `file_owner` (`number`) | file_owner<br/><br/>**Example:**<br/>`0` |
| `file_group` (`number`) | file_group<br/><br/>**Example:**<br/>`0` |
| `op` (`string`) | op_fs_open<br/><br/>**Possible values:**<ul><li>`open`</li><li>`openat`</li><li>`opendir`</li><li>`creat`</li><li>`fopen`</li><li>`freopen`</li><li>`open64`</li><li>`openat64`</li><li>`__open_2`</li><li>`__openat_2`</li><li>`creat64`</li><li>`fopen64`</li><li>`freopen64`</li><li>`recvmsg`</li><li>`console output`</li><li>`console input`</li></ul> |

<hr/>

### fs.read [^](#schema-reference) {#eventfsread}

Structure of the `fs.read` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890091215105",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.455076,
    "source": "fs.read",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "_metric": "fs.read",
      "_metric_type": "histogram",
      "_value": 4096,
      "proc": "accept01",
      "pid": 1933,
      "fd": 3,
      "op": "fgets_unlocked",
      "file": "/etc/passwd",
      "numops": 1,
      "unit": "byte"
    }
  }
}
```

#### `fs.read` properties {#eventfsreadprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsreadbody)._ |

#### `fs.read.body` properties {#eventfsreadbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Read<br/><br/>Value must be `fs.read`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsreadbodydata)._ |

#### `fs.read.body.data` properties {#eventfsreadbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - File Read<br/><br/>Value must be `fs.read`. |
| `_metric_type` (`string`) | histogram<br/><br/>Value must be `histogram`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `op` (`string`) | op_fs_read<br/><br/>**Possible values:**<ul><li>`go_read`</li><li>`readdir`</li><li>`pread64`</li><li>`preadv`</li><li>`preadv2`</li><li>`preadv64v2`</li><li>`__pread_chk`</li><li>`__read_chk`</li><li>`__fread_unlocked_chk`</li><li>`read`</li><li>`readv`</li><li>`pread`</li><li>`fread`</li><li>`__fread_chk`</li><li>`fread_unlocked`</li><li>`fgets`</li><li>`__fgets_chk`</li><li>`fgets_unlocked`</li><li>`__fgetws_chk`</li><li>`fgetws`</li><li>`fgetwc`</li><li>`fgetc`</li><li>`fscanf`</li><li>`getline`</li><li>`getdelim`</li><li>`__getdelim`</li></ul> |
| `file` (`string`) | file |
| `numops` (`number`) | numops |
| `unit` (`string`) | Unit - byte<br/><br/>Value must be `byte`. |

<hr/>

### fs.seek [^](#schema-reference) {#eventfsseek}

Structure of the `fs.seek` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-sh-/bin/sh ./file_x",
  "_channel": "5891441789884",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735836.805196,
    "source": "fs.seek",
    "host": "8bc1398c19f3",
    "proc": "sh",
    "cmd": "/bin/sh ./file_x",
    "pid": 2061,
    "data": {
      "_metric": "fs.seek",
      "_metric_type": "counter",
      "_value": 1,
      "proc": "sh",
      "pid": 2061,
      "fd": 3,
      "op": "lseek",
      "file": "./file_x",
      "unit": "operation"
    }
  }
}
```

#### `fs.seek` properties {#eventfsseekprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsseekbody)._ |

#### `fs.seek.body` properties {#eventfsseekbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Seek<br/><br/>Value must be `fs.seek`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsseekbodydata)._ |

#### `fs.seek.body.data` properties {#eventfsseekbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - File Seek<br/><br/>Value must be `fs.seek`. |
| `_metric_type` (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `op` (`string`) | op_fs_seek<br/><br/>**Possible values:**<ul><li>`lseek64`</li><li>`fseek64`</li><li>`ftello64`</li><li>`fsetpos64`</li><li>`lseek`</li><li>`fseek`</li><li>`fseeko`</li><li>`ftell`</li><li>`ftello`</li><li>`rewind`</li><li>`fsetpos`</li><li>`fgetpos`</li><li>`fgetpos64`</li></ul> |
| `file` (`string`) | file |
| `unit` (`string`) | Unit - operation<br/><br/>Value must be `operation`. |

<hr/>

### fs.stat [^](#schema-reference) {#eventfsstat}

Structure of the `fs.stat` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890091777333",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.454905,
    "source": "fs.stat",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "_metric": "fs.stat",
      "_metric_type": "counter",
      "_value": 1,
      "proc": "accept01",
      "pid": 1933,
      "op": "access",
      "file": "/dev/shm",
      "unit": "operation"
    }
  }
}
```

#### `fs.stat` properties {#eventfsstatprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsstatbody)._ |

#### `fs.stat.body` properties {#eventfsstatbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Stat<br/><br/>Value must be `fs.stat`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsstatbodydata)._ |

#### `fs.stat.body.data` properties {#eventfsstatbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - File Stat<br/><br/>Value must be `fs.stat`. |
| `_metric_type` (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `op` (`string`) | op_fs_stat<br/><br/>**Possible values:**<ul><li>`statfs64`</li><li>`__xstat`</li><li>`__xstat64`</li><li>`__lxstat`</li><li>`__lxstat64`</li><li>`__fxstat`</li><li>`__fxstatat`</li><li>`__fxstatat64`</li><li>`statx`</li><li>`statfs`</li><li>`statvfs`</li><li>`statvfs64`</li><li>`access`</li><li>`faccessat`</li><li>`stat`</li><li>`lstat`</li><li>`fstatfs64`</li><li>`__fxstat`</li><li>`__fxstat64`</li><li>`fstatfs`</li><li>`fstatvfs`</li><li>`fstatvfs64`</li><li>`fstat`</li><li>`fstatat`</li></ul> |
| `file` (`string`) | file |
| `unit` (`string`) | Unit - operation<br/><br/>Value must be `operation`. |

<hr/>

### fs.write [^](#schema-reference) {#eventfswrite}

Structure of the `fs.write` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-access02-/kernel/syscalls/access/access02",
  "_channel": "5891407740765",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735836.773249,
    "source": "fs.write",
    "host": "8bc1398c19f3",
    "proc": "access02",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/access/access02",
    "pid": 2058,
    "data": {
      "_metric": "fs.write",
      "_metric_type": "histogram",
      "_value": 10,
      "proc": "access02",
      "pid": 2058,
      "fd": 3,
      "op": "__write_libc",
      "file": "file_x",
      "numops": 1,
      "unit": "byte"
    }
  }
}
```

#### `fs.write` properties {#eventfswriteprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfswritebody)._ |

#### `fs.write.body` properties {#eventfswritebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Write<br/><br/>Value must be `fs.write`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfswritebodydata)._ |

#### `fs.write.body.data` properties {#eventfswritebodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - File Write<br/><br/>Value must be `fs.write`. |
| `_metric_type` (`string`) | histogram<br/><br/>Value must be `histogram`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `op` (`string`) | op_fs_write<br/><br/>**Possible values:**<ul><li>`go_write`</li><li>`pwrite64`</li><li>`pwritev`</li><li>`pwritev64`</li><li>`pwritev2`</li><li>`pwritev64v2`</li><li>`__overflow`</li><li>`__write_libc`</li><li>`__write_pthread`</li><li>`fwrite_unlocked`</li><li>`__stdio_write`</li><li>`write`</li><li>`pwrite`</li><li>`writev`</li><li>`fwrite`</li><li>`puts`</li><li>`putchar`</li><li>`fputs`</li><li>`fputs_unlocked`</li><li>`fputc`</li><li>`fputc_unlocked`</li><li>`putwc`</li><li>`fputwc`</li></ul> |
| `file` (`string`) | file |
| `numops` (`number`) | numops |
| `unit` (`string`) | Unit - byte<br/><br/>Value must be `byte`. |

<hr/>

### http.req [^](#schema-reference) {#eventhttpreq}

Structure of the `http.req` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-firefox-/usr/lib/firefox/firefox",
  "_channel": "13470846442500",
  "body": {
    "sourcetype": "http",
    "_time": 1643735942.588626,
    "source": "http.req",
    "host": "ubuntu",
    "proc": "firefox",
    "cmd": "/usr/lib/firefox/firefox",
    "pid": 6544,
    "data": {
      "http_method": "GET",
      "http_target": "/canonical.html",
      "http_flavor": "1.1",
      "http_scheme": "http",
      "http_host": "detectportal.firefox.com",
      "http_user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0",
      "net_transport": "IP.TCP",
      "net_peer_ip": "34.107.221.82",
      "net_peer_port": 80,
      "net_host_ip": "172.16.198.210",
      "net_host_port": 33712
    }
  }
}
```

#### `http.req` properties {#eventhttpreqprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventhttpreqbody)._ |

#### `http.req.body` properties {#eventhttpreqbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - http<br/><br/>Value must be `http`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - HTTP request<br/><br/>Value must be `http.req`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventhttpreqbodydata)._ |

#### `http.req.body.data` properties {#eventhttpreqbodydata}

| Property | Description |
|---|---|
| `http_method` (`string`) | http_method |
| `http_frame` (`string`) | http_frame<br/><br/>**Possible values:**<ul><li>`HEADERS`</li><li>`PUSH_PROMISE`</li></ul> |
| `http_target` (`string`) | http_target |
| `http_flavor` (`string`) | http_flavor |
| `http_stream` (`integer`) | http_stream |
| `http_scheme` (`string`) | http_scheme<br/><br/>**Possible values:**<ul><li>`http`</li><li>`https`</li></ul> |
| `http_host` (`string`) | http_host |
| `http_user_agent` (`string`) | http_user_agent |
| `http_client_ip` (`string`) | http_client_ip |
| `net_transport` (`string`) | net_transport<br/><br/>**Possible values:**<ul><li>`IP.TCP`</li><li>`IP.UDP`</li><li>`IP.RAW`</li><li>`IP.RDM`</li><li>`IP.SEQPACKET`</li><li>`Unix.TCP`</li><li>`Unix.UDP`</li><li>`Unix.RAW`</li><li>`Unix.RDM`</li><li>`Unix.SEQPACKET`</li></ul> |
| `net_peer_ip` (`string`) | net_peer_ip |
| `net_peer_port` (`integer`) | net_peer_port |
| `net_host_ip` (`string`) | net_host_ip |
| `net_host_port` (`integer`) | net_host_port |
| `x_appscope` (`string`) | x-appscope<br/><br/>Value must be `x-appscope`. |

<hr/>

### http.resp [^](#schema-reference) {#eventhttpresp}

Structure of the `http.resp` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-firefox-/usr/lib/firefox/firefox",
  "_channel": "13470846442500",
  "body": {
    "sourcetype": "http",
    "_time": 1643735942.613892,
    "source": "http.resp",
    "host": "ubuntu",
    "proc": "firefox",
    "cmd": "/usr/lib/firefox/firefox",
    "pid": 6544,
    "data": {
      "http_method": "GET",
      "http_target": "/canonical.html",
      "http_scheme": "http",
      "http_flavor": "1.1",
      "http_status_code": 200,
      "http_status_text": "OK",
      "http_server_duration": 26,
      "http_host": "detectportal.firefox.com",
      "http_user_agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0",
      "net_transport": "IP.TCP",
      "net_peer_ip": "34.107.221.82",
      "net_peer_port": 80,
      "net_host_ip": "172.16.198.210",
      "net_host_port": 33712,
      "http_response_content_length": 90
    }
  }
}
```

#### `http.resp` properties {#eventhttprespprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventhttprespbody)._ |

#### `http.resp.body` properties {#eventhttprespbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - http<br/><br/>Value must be `http`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - HTTP response<br/><br/>Value must be `http.resp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventhttprespbodydata)._ |

#### `http.resp.body.data` properties {#eventhttprespbodydata}

| Property | Description |
|---|---|
| `http_method` (`string`) | http_method |
| `http_target` (`string`) | http_target |
| `http_stream` (`integer`) | http_stream |
| `http_scheme` (`string`) | http_scheme<br/><br/>**Possible values:**<ul><li>`http`</li><li>`https`</li></ul> |
| `http_flavor` (`string`) | http_flavor |
| `http_status_code` (`integer`) | http_status_code<br/><br/>**Possible values:**<ul><li>`100`</li><li>`101`</li><li>`102`</li><li>`200`</li><li>`201`</li><li>`202`</li><li>`203`</li><li>`204`</li><li>`205`</li><li>`206`</li><li>`207`</li><li>`208`</li><li>`226`</li><li>`300`</li><li>`301`</li><li>`302`</li><li>`303`</li><li>`304`</li><li>`305`</li><li>`307`</li><li>`400`</li><li>`401`</li><li>`402`</li><li>`403`</li><li>`404`</li><li>`405`</li><li>`406`</li><li>`407`</li><li>`408`</li><li>`409`</li><li>`410`</li><li>`411`</li><li>`412`</li><li>`413`</li><li>`414`</li><li>`415`</li><li>`416`</li><li>`417`</li><li>`418`</li><li>`421`</li><li>`422`</li><li>`423`</li><li>`424`</li><li>`426`</li><li>`428`</li><li>`429`</li><li>`431`</li><li>`444`</li><li>`451`</li><li>`499`</li><li>`500`</li><li>`501`</li><li>`502`</li><li>`503`</li><li>`504`</li><li>`505`</li><li>`506`</li><li>`507`</li></ul> |
| `http_status_text` (`string`) | http_status_text<br/><br/>**Possible values:**<ul><li>`Continue`</li><li>`Switching Protocols`</li><li>`Processing`</li><li>`OK`</li><li>`Created`</li><li>`Accepted`</li><li>`Non-authoritative Information`</li><li>`No Content`</li><li>`Reset Content`</li><li>`Partial Content`</li><li>`Multi-Status`</li><li>`Already Reported`</li><li>`IM Used`</li><li>`Multiple Choices`</li><li>`Moved Permanently`</li><li>`Found`</li><li>`See Other`</li><li>`Not Modified`</li><li>`Use Proxy`</li><li>`Temporary Redirect`</li><li>`Permanent Redirect`</li><li>`Bad Request`</li><li>`Unauthorized`</li><li>`Payment Required`</li><li>`Forbidden`</li><li>`Not Found`</li><li>`Method Not Allowed`</li><li>`Not Acceptable`</li><li>`Proxy Authentication Required`</li><li>`Request Timeout`</li><li>`Conflict`</li><li>`Gone`</li><li>`Length Required`</li><li>`Precondition Failed`</li><li>`Payload Too Large`</li><li>`Request-URI Too Long`</li><li>`Unsupported Media Type`</li><li>`Requested Range Not Satisfiable`</li><li>`Expectation Failed`</li><li>`I'm a teapot`</li><li>`Misdirected Request`</li><li>`Unprocessable Entity`</li><li>`Locked`</li><li>`Failed Dependency`</li><li>`Upgrade Required`</li><li>`Precondition Required`</li><li>`Too Many Requests`</li><li>`Request Header Fields Too Large`</li><li>`Connection Closed Without Response`</li><li>`Unavailable For Legal Reasons`</li><li>`Client Closed Request`</li><li>`Internal Server Error`</li><li>`Not Implemented`</li><li>`Bad Gateway`</li><li>`Service Unavailable`</li><li>`Gateway Timeout`</li><li>`HTTP Version Not Supported`</li><li>`Variant Also Negotiates`</li><li>`Insufficient Storage`</li></ul> |
| `http_client_duration` (`number`) | http_client_duration |
| `http_server_duration` (`number`) | http_server_duration |
| `http_host` (`string`) | http_host |
| `http_user_agent` (`string`) | http_user_agent |
| `net_transport` (`string`) | net_transport<br/><br/>**Possible values:**<ul><li>`IP.TCP`</li><li>`IP.UDP`</li><li>`IP.RAW`</li><li>`IP.RDM`</li><li>`IP.SEQPACKET`</li><li>`Unix.TCP`</li><li>`Unix.UDP`</li><li>`Unix.RAW`</li><li>`Unix.RDM`</li><li>`Unix.SEQPACKET`</li></ul> |
| `net_peer_ip` (`string`) | net_peer_ip |
| `net_peer_port` (`integer`) | net_peer_port |
| `net_host_ip` (`string`) | net_host_ip |
| `net_host_port` (`integer`) | net_host_port |
| `http_response_content_length` (`number`) | http_response_content_length |

<hr/>

### net.app [^](#schema-reference) {#eventnetapp}

Structure of the `net.app` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-firefox-/usr/lib/firefox/firefox",
  "_channel": "13470846442500",
  "body": {
    "sourcetype": "net",
    "_time": 1643735942.588594,
    "source": "net.app",
    "host": "ubuntu",
    "proc": "firefox",
    "cmd": "/usr/lib/firefox/firefox",
    "pid": 6544,
    "data": {
      "proc": "firefox",
      "pid": 6544,
      "fd": 91,
      "host": "ubuntu",
      "protocol": "HTTP"
    }
  }
}
```

#### `net.app` properties {#eventnetappprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetappbody)._ |

#### `net.app.body` properties {#eventnetappbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - net<br/><br/>Value must be `net`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net App<br/><br/>Value must be `net.app`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetappbodydata)._ |

#### `net.app.body.data` properties {#eventnetappbodydata}

| Property | Description |
|---|---|
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` (`string`) | host |
| `protocol` (`string`) | protocol<br/><br/>**Possible values:**<ul><li>`HTTP`</li></ul> |

<hr/>

### net.close [^](#schema-reference) {#eventnetclose}

Structure of the `net.close` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-recvfrom01-nel/syscalls/recvfrom/recvfrom01",
  "_channel": "5912618970557",
  "body": {
    "sourcetype": "net",
    "_time": 1643735857.983449,
    "source": "net.close",
    "host": "8bc1398c19f3",
    "proc": "recvfrom01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/recvfrom/recvfrom01",
    "pid": 3793,
    "data": {
      "net_transport": "IP.TCP",
      "net_peer_ip": "0.0.0.0",
      "net_peer_port": 35533,
      "net_host_ip": "127.0.0.1",
      "net_host_port": 40184,
      "duration": 0,
      "net_bytes_sent": 0,
      "net_bytes_recv": 6,
      "net_close_reason": "local"
    }
  }
}
```

#### `net.close` properties {#eventnetcloseprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetclosebody)._ |

#### `net.close.body` properties {#eventnetclosebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - net<br/><br/>Value must be `net`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net Close<br/><br/>Value must be `net.close`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetclosebodydata)._ |

#### `net.close.body.data` properties {#eventnetclosebodydata}

| Property | Description |
|---|---|
| `net_transport` (`string`) | net_transport<br/><br/>**Possible values:**<ul><li>`IP.TCP`</li><li>`IP.UDP`</li><li>`IP.RAW`</li><li>`IP.RDM`</li><li>`IP.SEQPACKET`</li><li>`Unix.TCP`</li><li>`Unix.UDP`</li><li>`Unix.RAW`</li><li>`Unix.RDM`</li><li>`Unix.SEQPACKET`</li></ul> |
| `net_peer_ip` (`string`) | net_peer_ip |
| `net_peer_port` (`integer`) | net_peer_port |
| `net_host_ip` (`string`) | net_host_ip |
| `net_host_port` (`integer`) | net_host_port |
| `net_protocol` (`string`) | net_protocol<br/><br/>Value must be `http`. |
| `unix_peer_inode` (`number`) | unix_peer_inode |
| `unix_local_inode` (`number`) | unix_local_inode |
| `duration` (`number`) | duration<br/><br/>**Example:**<br/>`55` |
| `net_bytes_sent` (`number`) | net_bytes_sent |
| `net_bytes_recv` (`number`) | net_bytes_recv |
| `net_close_reason` (`string`) | net_close_reason<br/><br/>**Possible values:**<ul><li>`local`</li><li>`remote`</li></ul> |

<hr/>

### net.duration [^](#schema-reference) {#eventnetduration}

Structure of the `net.duration` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-recvmsg01-ernel/syscalls/recvmsg/recvmsg01",
  "_channel": "5912681876432",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735858.046756,
    "source": "net.duration",
    "host": "8bc1398c19f3",
    "proc": "recvmsg01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/recvmsg/recvmsg01",
    "pid": 3798,
    "data": {
      "_metric": "net.duration",
      "_metric_type": "timer",
      "_value": 1,
      "proc": "recvmsg01",
      "pid": 3798,
      "fd": 4,
      "proto": "TCP",
      "port": 41482,
      "numops": 1,
      "unit": "millisecond"
    }
  }
}
```

#### `net.duration` properties {#eventnetdurationprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetdurationbody)._ |

#### `net.duration.body` properties {#eventnetdurationbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net duration<br/><br/>Value must be `net.duration`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetdurationbodydata)._ |

#### `net.duration.body.data` properties {#eventnetdurationbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net duration<br/><br/>Value must be `net.duration`. |
| `_metric_type` (`string`) | timer<br/><br/>Value must be `timer`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `numops` (`number`) | numops |
| `unit` (`string`) | Unit - millisecond<br/><br/>Value must be `millisecond`. |

<hr/>

### net.error [^](#schema-reference) {#eventneterror}

Structure of the `net.error` event

#### Example

```json
{
  "type": "evt",
  "id": "90aac4bb0722-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "2745569202700291",
  "body": {
    "sourcetype": "metric",
    "_time": 1643972258.00885,
    "source": "net.error",
    "host": "90aac4bb0722",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1934,
    "data": {
      "_metric": "net.error",
      "_metric_type": "counter",
      "_value": 1,
      "proc": "accept01",
      "pid": 1934,
      "op": "accept",
      "class": "connection",
      "unit": "operation"
    }
  }
}
```

#### `net.error` properties {#eventneterrorprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventneterrorbody)._ |

#### `net.error.body` properties {#eventneterrorbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net Error<br/><br/>Value must be `net.error`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventneterrorbodydata)._ |

#### `net.error.body.data` properties {#eventneterrorbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net Error<br/><br/>Value must be `net.error`. |
| `_metric_type` (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `op` (`string`) | op |
| `class` (`string`) | connection<br/><br/>Value must be `connection`. |
| `unit` (`string`) | Unit - operation<br/><br/>Value must be `operation`. |

<hr/>

### net.open [^](#schema-reference) {#eventnetopen}

Structure of the `net.open` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept02-/kernel/syscalls/accept/accept02",
  "_channel": "5890157346952",
  "body": {
    "sourcetype": "net",
    "_time": 1643735835.521928,
    "source": "net.open",
    "host": "8bc1398c19f3",
    "proc": "accept02",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept02",
    "pid": 1936,
    "data": {
      "net_transport": "IP.TCP",
      "net_peer_ip": "127.0.0.1",
      "net_peer_port": 58625,
      "net_host_ip": "0.0.0.0",
      "net_host_port": 0
    }
  }
}
```

#### `net.open` properties {#eventnetopenprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetopenbody)._ |

#### `net.open.body` properties {#eventnetopenbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - net<br/><br/>Value must be `net`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net Open<br/><br/>Value must be `net.open`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetopenbodydata)._ |

#### `net.open.body.data` properties {#eventnetopenbodydata}

| Property | Description |
|---|---|
| `net_transport` (`string`) | net_transport<br/><br/>**Possible values:**<ul><li>`IP.TCP`</li><li>`IP.UDP`</li><li>`IP.RAW`</li><li>`IP.RDM`</li><li>`IP.SEQPACKET`</li><li>`Unix.TCP`</li><li>`Unix.UDP`</li><li>`Unix.RAW`</li><li>`Unix.RDM`</li><li>`Unix.SEQPACKET`</li></ul> |
| `net_peer_ip` (`string`) | net_peer_ip |
| `net_peer_port` (`integer`) | net_peer_port |
| `net_host_ip` (`string`) | net_host_ip |
| `net_host_port` (`integer`) | net_host_port |
| `unix_peer_inode` (`number`) | unix_peer_inode |
| `unix_local_inode` (`number`) | unix_local_inode |
| `net_protocol` (`string`) | net_protocol<br/><br/>Value must be `http`. |

<hr/>

### net.other [^](#schema-reference) {#eventnetother}

Structure of the `net.other` event

#### Example

```json
{
  "type": "evt",
  "id": "test_user-server_seqpacket-./server_seqpacket",
  "_channel": "11977632602680",
  "body": {
    "sourcetype": "metric",
    "_time": 1643886739.820863,
    "source": "net.other",
    "host": "test_user",
    "proc": "server_seqpacket",
    "cmd": "./server_seqpacket",
    "pid": 232570,
    "data": {
      "_metric": "net.other",
      "_metric_type": "gauge",
      "_value": 1,
      "proc": "server_seqpacket",
      "pid": 232570,
      "fd": 3,
      "proto": "SEQPACKET",
      "port": 0,
      "unit": "connection"
    }
  }
}
```

#### `net.other` properties {#eventnetotherprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetotherbody)._ |

#### `net.other.body` properties {#eventnetotherbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net other<br/><br/>Value must be `net.other`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetotherbodydata)._ |

#### `net.other.body.data` properties {#eventnetotherbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net other<br/><br/>Value must be `net.other`. |
| `_metric_type` (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `unit` (`string`) | Unit - connection<br/><br/>Value must be `connection`. |

<hr/>

### net.port [^](#schema-reference) {#eventnetport}

Structure of the `net.port` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890091645261",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.455222,
    "source": "net.port",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "_metric": "net.port",
      "_metric_type": "gauge",
      "_value": 1,
      "proc": "accept01",
      "pid": 1933,
      "fd": 4,
      "proto": "TCP",
      "port": 0,
      "unit": "instance"
    }
  }
}
```

#### `net.port` properties {#eventnetportprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetportbody)._ |

#### `net.port.body` properties {#eventnetportbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net port<br/><br/>Value must be `net.port`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetportbodydata)._ |

#### `net.port.body.data` properties {#eventnetportbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net port<br/><br/>Value must be `net.port`. |
| `_metric_type` (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `unit` (`string`) | Unit - instance<br/><br/>Value must be `instance`. |

<hr/>

### net.rx [^](#schema-reference) {#eventnetrx}

Structure of the `net.rx` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-recvfrom01-nel/syscalls/recvfrom/recvfrom01",
  "_channel": "5912618970557",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735857.983368,
    "source": "net.rx",
    "host": "8bc1398c19f3",
    "proc": "recvfrom01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/recvfrom/recvfrom01",
    "pid": 3793,
    "data": {
      "_metric": "net.rx",
      "_metric_type": "counter",
      "_value": 6,
      "proc": "recvfrom01",
      "pid": 3793,
      "fd": 4,
      "domain": "AF_INET",
      "proto": "TCP",
      "localip": "127.0.0.1",
      "localp": 40184,
      "remoteip": "0.0.0.0",
      "remotep": 35533,
      "data": "clear",
      "numops": 1,
      "unit": "byte"
    }
  }
}
```

#### `net.rx` properties {#eventnetrxprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetrxbody)._ |

#### `net.rx.body` properties {#eventnetrxbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net RX<br/><br/>Value must be `net.rx`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetrxbodydata)._ |

#### `net.rx.body.data` properties {#eventnetrxbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net RX<br/><br/>Value must be `net.rx`. |
| `_metric_type` (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `domain` (`string`) | domain |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `localip` (`string`) | localip<br/><br/>**Example:**<br/>`127.0.0.1` |
| `localp` (`number`) | localp<br/><br/>**Example:**<br/>`9109` |
| `localn` (`number`) | localn |
| `remoteip` (`string`) | remoteip<br/><br/>**Example:**<br/>`192.158.1.38` |
| `remotep` (`number`) | remotep<br/><br/>**Example:**<br/>`9108` |
| `remoten` (`number`) | remoten |
| `data` (`string`) | data<br/><br/>**Possible values:**<ul><li>`ssl`</li><li>`clear`</li></ul> |
| `numops` (`number`) | numops |
| `unit` (`string`) | Unit - byte<br/><br/>Value must be `byte`. |

<hr/>

### net.tcp [^](#schema-reference) {#eventnettcp}

Structure of the `net.tcp` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890091645261",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.455387,
    "source": "net.tcp",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "_metric": "net.tcp",
      "_metric_type": "gauge",
      "_value": 0,
      "proc": "accept01",
      "pid": 1933,
      "fd": 4,
      "proto": "TCP",
      "port": 0,
      "unit": "connection"
    }
  }
}
```

#### `net.tcp` properties {#eventnettcpprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnettcpbody)._ |

#### `net.tcp.body` properties {#eventnettcpbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net tcp<br/><br/>Value must be `net.tcp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnettcpbodydata)._ |

#### `net.tcp.body.data` properties {#eventnettcpbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net tcp<br/><br/>Value must be `net.tcp`. |
| `_metric_type` (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `proto` (`string`) | proto_tcp<br/><br/>Value must be `TCP`. |
| `port` (`number`) | port |
| `unit` (`string`) | Unit - connection<br/><br/>Value must be `connection`. |

<hr/>

### net.tx [^](#schema-reference) {#eventnettx}

Structure of the `net.tx` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-recvfrom01-nel/syscalls/recvfrom/recvfrom01",
  "_channel": "5912618642035",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735857.983059,
    "source": "net.tx",
    "host": "8bc1398c19f3",
    "proc": "recvfrom01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/recvfrom/recvfrom01",
    "pid": 3795,
    "data": {
      "_metric": "net.tx",
      "_metric_type": "counter",
      "_value": 6,
      "proc": "recvfrom01",
      "pid": 3795,
      "fd": 4,
      "domain": "AF_INET",
      "proto": "TCP",
      "localip": "0.0.0.0",
      "localp": 0,
      "remoteip": "127.0.0.1",
      "remotep": 40184,
      "data": "clear",
      "numops": 1,
      "unit": "byte"
    }
  }
}
```

#### `net.tx` properties {#eventnettxprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnettxbody)._ |

#### `net.tx.body` properties {#eventnettxbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net TX<br/><br/>Value must be `net.tx`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnettxbodydata)._ |

#### `net.tx.body.data` properties {#eventnettxbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net TX<br/><br/>Value must be `net.tx`. |
| `_metric_type` (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `domain` (`string`) | domain |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `localip` (`string`) | localip<br/><br/>**Example:**<br/>`127.0.0.1` |
| `localp` (`number`) | localp<br/><br/>**Example:**<br/>`9109` |
| `localn` (`number`) | localn |
| `remoteip` (`string`) | remoteip<br/><br/>**Example:**<br/>`192.158.1.38` |
| `remotep` (`number`) | remotep<br/><br/>**Example:**<br/>`9108` |
| `remoten` (`number`) | remoten |
| `data` (`string`) | data<br/><br/>**Possible values:**<ul><li>`ssl`</li><li>`clear`</li></ul> |
| `numops` (`number`) | numops |
| `unit` (`string`) | Unit - byte<br/><br/>Value must be `byte`. |

<hr/>

### net.udp [^](#schema-reference) {#eventnetudp}

Structure of the `net.udp` event

#### Example

```json
{
  "type": "evt",
  "id": "8bc1398c19f3-accept01-/kernel/syscalls/accept/accept01",
  "_channel": "5890091656419",
  "body": {
    "sourcetype": "metric",
    "_time": 1643735835.455419,
    "source": "net.udp",
    "host": "8bc1398c19f3",
    "proc": "accept01",
    "cmd": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "pid": 1933,
    "data": {
      "_metric": "net.udp",
      "_metric_type": "gauge",
      "_value": 0,
      "proc": "accept01",
      "pid": 1933,
      "fd": 5,
      "proto": "UDP",
      "port": 0,
      "unit": "connection"
    }
  }
}
```

#### `net.udp` properties {#eventnetudpprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetudpbody)._ |

#### `net.udp.body` properties {#eventnetudpbody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net udp<br/><br/>Value must be `net.udp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetudpbodydata)._ |

#### `net.udp.body.data` properties {#eventnetudpbodydata}

| Property | Description |
|---|---|
| `_metric` (`string`) | Source - Net udp<br/><br/>Value must be `net.udp`. |
| `_metric_type` (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `proto` (`string`) | proto_udp<br/><br/>Value must be `UDP`. |
| `port` (`number`) | port |
| `unit` (`string`) | Unit - connection<br/><br/>Value must be `connection`. |

<hr/>

### notice [^](#schema-reference) {#eventnotice}

Structure of the `notice` event

#### Example

```json
{
  "type": "evt",
  "id": "9a721a6ad0be-htop-htop",
  "_channel": "13544129471303",
  "body": {
    "sourcetype": "metric",
    "_time": 1643888296.317304,
    "source": "notice",
    "host": "9a721a6ad0be",
    "proc": "htop",
    "cmd": "htop",
    "pid": 302,
    "data": "Truncated metrics. Your rate exceeded 10000 metrics per second"
  }
}
```

#### `notice` properties {#eventnoticeprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnoticebody)._ |

#### `notice.body` properties {#eventnoticebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - notice<br/><br/>Value must be `notice`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`string`) | data |

<hr/>

### console [^](#schema-reference) {#eventconsole}

Structure of the `console` event

#### Example

```json
{
  "type": "evt",
  "id": "eaf4d0598443-a.out-./a.out",
  "_channel": "8499188821284",
  "body": {
    "sourcetype": "console",
    "_time": 1643883251.376672,
    "source": "stderr",
    "host": "eaf4d0598443",
    "proc": "a.out",
    "cmd": "./a.out",
    "pid": 986,
    "data": {
      "message": "stderr hello world"
    }
  }
}
```

#### `console` properties {#eventconsoleprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventconsolebody)._ |

#### `console.body` properties {#eventconsolebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - console<br/><br/>Value must be `console`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Specifies whether AppScope is capturing either `stderr` or `file` from console.<br/><br/>Value must be `stderr` or `file`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventconsolebodydata)._ |

#### `console.body.data` properties {#eventconsolebodydata}

| Property | Description |
|---|---|
| `message` (`string`) | message |

<hr/>

### file [^](#schema-reference) {#eventfile}

Structure of the `file` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-sh- /usr/bin/which /usr/bin/firefox",
  "_channel": "13468365092424",
  "body": {
    "sourcetype": "file",
    "_time": 1643735941.602952,
    "source": "/var/log/firefox.log",
    "host": "ubuntu",
    "proc": "sh",
    "cmd": "/bin/sh /usr/bin/which /usr/bin/firefox",
    "pid": 6545,
    "data": {
      "message": "/usr/bin/firefox\n"
    }
  }
}
```

#### `file` properties {#eventfileprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfilebody)._ |

#### `file.body` properties {#eventfilebody}

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - file<br/><br/>Value must be `file`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | String that describes a file path. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfilebodydata)._ |

#### `file.body.data` properties {#eventfilebodydata}

| Property | Description |
|---|---|
| `message` (`string`) | message |

<hr/>

### fs.close [^](#schema-reference) {#metricfsclose}

Structure of the `fs.close` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.close",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 13687,
    "host": "1f0ec6c8a7bc",
    "unit": "operation",
    "summary": "true",
    "_time": 1643826403.121424
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.close",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 9871,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "op": "close",
    "file": "/dev/shm/ltp_accept01_9870",
    "unit": "operation",
    "_time": 1643826292.07658
  }
}
```

#### `fs.close` properties {#metricfscloseprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsclosebody)._ |

#### `fs.close.body` properties {#metricfsclosebody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Close<br/><br/>Value must be `fs.close`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op_fs_close<br/><br/>**Possible values:**<ul><li>`go_close`</li><li>`closedir`</li><li>`freopen`</li><li>`freopen64`</li><li>`close`</li><li>`fclose`</li><li>`close$NOCANCEL`</li><li>`guarded_close_np`</li><li>`close_nocancel`</li></ul> |
| `file` (`string`) | file |
| `unit` _required_ (`string`) | Unit - operation<br/><br/>Value must be `operation`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.duration [^](#schema-reference) {#metricfsduration}

Structure of the `fs.duration` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.duration",
    "_metric_type": "histogram",
    "_value": 1,
    "proc": "access01",
    "pid": 13697,
    "host": "1f0ec6c8a7bc",
    "unit": "microsecond",
    "summary": "true",
    "_time": 1643826404.006442
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.duration",
    "_metric_type": "histogram",
    "_value": 16,
    "proc": "accept01",
    "pid": 9871,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "op": "fgets_unlocked",
    "file": "/etc/passwd",
    "numops": 1,
    "unit": "microsecond",
    "_time": 1643826292.076675
  }
}
```

#### `fs.duration` properties {#metricfsdurationprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsdurationbody)._ |

#### `fs.duration.body` properties {#metricfsdurationbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Duration<br/><br/>Value must be `fs.duration`. |
| `_metric_type` _required_ (`string`) | histogram<br/><br/>Value must be `histogram`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op |
| `file` (`string`) | file |
| `numops` (`number`) | numops |
| `unit` _required_ (`string`) | Unit - microsecond<br/><br/>Value must be `microsecond`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.error [^](#schema-reference) {#metricfserror}

Structure of the `fs.error` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.error",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 13686,
    "host": "1f0ec6c8a7bc",
    "class": "stat",
    "unit": "operation",
    "summary": "true",
    "_time": 1643826403.123802
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.error",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept02",
    "pid": 9872,
    "host": "1f0ec6c8a7bc",
    "op": "readdir",
    "file": "/tmp/QxbCjC",
    "class": "read_write",
    "unit": "operation",
    "_time": 1643826292.14466
  }
}
```

#### `fs.error` properties {#metricfserrorprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfserrorbody)._ |

#### `fs.error.body` properties {#metricfserrorbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Error<br/><br/>Value must be `fs.error`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op |
| `file` (`string`) | file |
| `class` _required_ (`string`) | class fs.error<br/><br/>**Possible values:**<ul><li>`open_close`</li><li>`read_write`</li><li>`stat`</li></ul> |
| `unit` _required_ (`string`) | Unit - operation<br/><br/>Value must be `operation`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.open [^](#schema-reference) {#metricfsopen}

Structure of the `fs.open` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.open",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 13687,
    "host": "1f0ec6c8a7bc",
    "unit": "operation",
    "summary": "true",
    "_time": 1643826403.121411
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.open",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 9871,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "op": "open",
    "file": "/dev/shm/ltp_accept01_9870",
    "unit": "operation",
    "_time": 1643826292.076503
  }
}
```

#### `fs.open` properties {#metricfsopenprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsopenbody)._ |

#### `fs.open.body` properties {#metricfsopenbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File open<br/><br/>Value must be `fs.open`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op_fs_open<br/><br/>**Possible values:**<ul><li>`open`</li><li>`openat`</li><li>`opendir`</li><li>`creat`</li><li>`fopen`</li><li>`freopen`</li><li>`open64`</li><li>`openat64`</li><li>`__open_2`</li><li>`__openat_2`</li><li>`creat64`</li><li>`fopen64`</li><li>`freopen64`</li><li>`recvmsg`</li><li>`console output`</li><li>`console input`</li></ul> |
| `file` (`string`) | file |
| `unit` _required_ (`string`) | Unit - operation<br/><br/>Value must be `operation`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.read [^](#schema-reference) {#metricfsread}

Structure of the `fs.read` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.read",
    "_metric_type": "counter",
    "_value": 13312,
    "proc": "access01",
    "pid": 13697,
    "host": "1f0ec6c8a7bc",
    "unit": "byte",
    "summary": "true",
    "_time": 1643826404.006381
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.read",
    "_metric_type": "counter",
    "_value": 4096,
    "proc": "accept01",
    "pid": 9871,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "op": "fgets_unlocked",
    "file": "/etc/passwd",
    "numops": 1,
    "unit": "byte",
    "_time": 1643826292.076709
  }
}
```

#### `fs.read` properties {#metricfsreadprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsreadbody)._ |

#### `fs.read.body` properties {#metricfsreadbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Read<br/><br/>Value must be `fs.read`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op_fs_read<br/><br/>**Possible values:**<ul><li>`go_read`</li><li>`readdir`</li><li>`pread64`</li><li>`preadv`</li><li>`preadv2`</li><li>`preadv64v2`</li><li>`__pread_chk`</li><li>`__read_chk`</li><li>`__fread_unlocked_chk`</li><li>`read`</li><li>`readv`</li><li>`pread`</li><li>`fread`</li><li>`__fread_chk`</li><li>`fread_unlocked`</li><li>`fgets`</li><li>`__fgets_chk`</li><li>`fgets_unlocked`</li><li>`__fgetws_chk`</li><li>`fgetws`</li><li>`fgetwc`</li><li>`fgetc`</li><li>`fscanf`</li><li>`getline`</li><li>`getdelim`</li><li>`__getdelim`</li></ul> |
| `file` (`string`) | file |
| `numops` (`number`) | numops |
| `unit` _required_ (`string`) | Unit - byte<br/><br/>Value must be `byte`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.seek [^](#schema-reference) {#metricfsseek}

Structure of the `fs.seek` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.seek",
    "_metric_type": "counter",
    "_value": 3,
    "proc": "sh",
    "pid": 13810,
    "host": "1f0ec6c8a7bc",
    "unit": "operation",
    "summary": "true",
    "_time": 1643826404.175738
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.seek",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "sh",
    "pid": 9994,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "op": "lseek",
    "file": "./file_x",
    "unit": "operation",
    "_time": 1643826293.407508
  }
}
```

#### `fs.seek` properties {#metricfsseekprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsseekbody)._ |

#### `fs.seek.body` properties {#metricfsseekbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Seek<br/><br/>Value must be `fs.seek`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op |
| `file` (`string`) | file |
| `unit` _required_ (`string`) | Unit - operation<br/><br/>Value must be `operation`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.stat [^](#schema-reference) {#metricfsstat}

Structure of the `fs.stat` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.stat",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 13686,
    "host": "1f0ec6c8a7bc",
    "unit": "operation",
    "summary": "true",
    "_time": 1643826403.123752
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.stat",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 9871,
    "host": "1f0ec6c8a7bc",
    "op": "access",
    "file": "/dev/shm",
    "unit": "operation",
    "_time": 1643826292.076446
  }
}
```

#### `fs.stat` properties {#metricfsstatprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsstatbody)._ |

#### `fs.stat.body` properties {#metricfsstatbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Stat<br/><br/>Value must be `fs.stat`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op |
| `file` (`string`) | file |
| `unit` _required_ (`string`) | Unit - operation<br/><br/>Value must be `operation`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### fs.write [^](#schema-reference) {#metricfswrite}

Structure of the `fs.write` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.write",
    "_metric_type": "counter",
    "_value": 10,
    "proc": "access02",
    "pid": 13806,
    "host": "1f0ec6c8a7bc",
    "unit": "byte",
    "summary": "true",
    "_time": 1643826404.234963
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "fs.write",
    "_metric_type": "counter",
    "_value": 10,
    "proc": "access02",
    "pid": 9991,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "op": "__write_libc",
    "file": "file_x",
    "numops": 1,
    "unit": "byte",
    "_time": 1643826293.385378
  }
}
```

#### `fs.write` properties {#metricfswriteprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfswritebody)._ |

#### `fs.write.body` properties {#metricfswritebody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - File Write<br/><br/>Value must be `fs.write`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op_fs_write<br/><br/>**Possible values:**<ul><li>`go_write`</li><li>`pwrite64`</li><li>`pwritev`</li><li>`pwritev64`</li><li>`pwritev2`</li><li>`pwritev64v2`</li><li>`__overflow`</li><li>`__write_libc`</li><li>`__write_pthread`</li><li>`fwrite_unlocked`</li><li>`__stdio_write`</li><li>`write`</li><li>`pwrite`</li><li>`writev`</li><li>`fwrite`</li><li>`puts`</li><li>`putchar`</li><li>`fputs`</li><li>`fputs_unlocked`</li><li>`fputc`</li><li>`fputc_unlocked`</li><li>`putwc`</li><li>`fputwc`</li></ul> |
| `file` (`string`) | file |
| `numops` (`number`) | numops |
| `unit` _required_ (`string`) | Unit - byte<br/><br/>Value must be `byte`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### http.duration.client [^](#schema-reference) {#metrichttpdurationclient}

Structure of the `http.duration.client` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.duration.client",
    "_metric_type": "timer",
    "_value": 6,
    "http_target": "/",
    "numops": 1,
    "proc": "lt-curl",
    "pid": 788,
    "host": "c067d78736db",
    "unit": "millisecond",
    "summary": "true",
    "_time": 1643924553.681483
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.duration.client",
    "_metric_type": "timer",
    "_value": 7,
    "http_target": "/",
    "numops": 1,
    "proc": "lt-curl",
    "pid": 30,
    "host": "c067d78736db",
    "unit": "millisecond",
    "summary": "true",
    "_time": 1643924472.648148
  }
}
```

#### `http.duration.client` properties {#metrichttpdurationclientprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpdurationclientbody)._ |

#### `http.duration.client.body` properties {#metrichttpdurationclientbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP client duration<br/><br/>Value must be `http.duration.client`. |
| `_metric_type` _required_ (`string`) | timer<br/><br/>Value must be `timer`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `http_target` _required_ (`string`) | http_target |
| `numops` _required_ (`number`) | numops |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - millisecond<br/><br/>Value must be `millisecond`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### http.request.content_length [^](#schema-reference) {#metrichttpreqcontentlength}

Structure of the `http.req.content_length` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.req.content_length",
    "_metric_type": "counter",
    "_value": 38,
    "http_target": "/echo/post/json",
    "numops": 1,
    "proc": "curl",
    "pid": 525,
    "host": "272cc69a120a",
    "unit": "byte",
    "summary": "true",
    "_time": 1644230452.63037
  }
}
```

#### `http.request.content_length` properties {#metrichttpreqcontentlengthprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpreqcontentlengthbody)._ |

#### `http.request.content_length.body` properties {#metrichttpreqcontentlengthbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP request content length<br/><br/>Value must be `http.req.content_length`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `http_target` _required_ (`string`) | http_target |
| `numops` _required_ (`number`) | numops |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - byte<br/><br/>Value must be `byte`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### http.req [^](#schema-reference) {#metrichttpreq}

Structure of the `http.req` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.req",
    "_metric_type": "counter",
    "_value": 1,
    "http_target": "/",
    "http_status_code": 200,
    "proc": "lt-curl",
    "pid": 788,
    "host": "c067d78736db",
    "unit": "request",
    "summary": "true",
    "_time": 1643924553.681441
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.req",
    "_metric_type": "counter",
    "_value": 1,
    "http_target": "/",
    "http_status_code": 200,
    "proc": "lt-curl",
    "pid": 30,
    "host": "c067d78736db",
    "unit": "request",
    "summary": "true",
    "_time": 1643924472.64811
  }
}
```

#### `http.req` properties {#metrichttpreqprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpreqbody)._ |

#### `http.req.body` properties {#metrichttpreqbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP requests<br/><br/>Value must be `http.req`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `http_target` _required_ (`string`) | http_target |
| `http_status_code` _required_ (`integer`) | http_status_code<br/><br/>**Possible values:**<ul><li>`100`</li><li>`101`</li><li>`102`</li><li>`200`</li><li>`201`</li><li>`202`</li><li>`203`</li><li>`204`</li><li>`205`</li><li>`206`</li><li>`207`</li><li>`208`</li><li>`226`</li><li>`300`</li><li>`301`</li><li>`302`</li><li>`303`</li><li>`304`</li><li>`305`</li><li>`307`</li><li>`400`</li><li>`401`</li><li>`402`</li><li>`403`</li><li>`404`</li><li>`405`</li><li>`406`</li><li>`407`</li><li>`408`</li><li>`409`</li><li>`410`</li><li>`411`</li><li>`412`</li><li>`413`</li><li>`414`</li><li>`415`</li><li>`416`</li><li>`417`</li><li>`418`</li><li>`421`</li><li>`422`</li><li>`423`</li><li>`424`</li><li>`426`</li><li>`428`</li><li>`429`</li><li>`431`</li><li>`444`</li><li>`451`</li><li>`499`</li><li>`500`</li><li>`501`</li><li>`502`</li><li>`503`</li><li>`504`</li><li>`505`</li><li>`506`</li><li>`507`</li></ul> |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - request<br/><br/>Value must be `request`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### http.response.content_length [^](#schema-reference) {#metrichttprespcontentlength}

Structure of the `http.resp.content_length` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.resp.content_length",
    "_metric_type": "counter",
    "_value": 58896,
    "http_target": "/",
    "numops": 1,
    "proc": "lt-curl",
    "pid": 788,
    "host": "c067d78736db",
    "unit": "byte",
    "summary": "true",
    "_time": 1643924553.6815
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.resp.content_length",
    "_metric_type": "counter",
    "_value": 58896,
    "http_target": "/",
    "numops": 1,
    "proc": "lt-curl",
    "pid": 30,
    "host": "c067d78736db",
    "unit": "byte",
    "summary": "true",
    "_time": 1643924472.648165
  }
}
```

#### `http.response.content_length` properties {#metrichttprespcontentlengthprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttprespcontentlengthbody)._ |

#### `http.response.content_length.body` properties {#metrichttprespcontentlengthbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP response content length<br/><br/>Value must be `http.resp.content_length`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `http_target` _required_ (`string`) | http_target |
| `numops` _required_ (`number`) | numops |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - byte<br/><br/>Value must be `byte`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### http.duration.server [^](#schema-reference) {#metrichttpdurationserver}

Structure of the `http.duration.server` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.duration.server",
    "_metric_type": "timer",
    "_value": 0,
    "http_target": "/",
    "numops": 1,
    "proc": "httpd",
    "pid": 2260,
    "host": "c067d78736db",
    "unit": "millisecond",
    "summary": "true",
    "_time": 1643924563.450939
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.duration.server",
    "_metric_type": "timer",
    "_value": 1,
    "http_target": "/",
    "numops": 1,
    "proc": "httpd",
    "pid": 648,
    "host": "c067d78736db",
    "unit": "millisecond",
    "summary": "true",
    "_time": 1643924498.350866
  }
}
```

#### `http.duration.server` properties {#metrichttpdurationserverprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpdurationserverbody)._ |

#### `http.duration.server.body` properties {#metrichttpdurationserverbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP server duration<br/><br/>Value must be `http.duration.server`. |
| `_metric_type` _required_ (`string`) | timer<br/><br/>Value must be `timer`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `http_target` _required_ (`string`) | http_target |
| `numops` _required_ (`number`) | numops |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - millisecond<br/><br/>Value must be `millisecond`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.close [^](#schema-reference) {#metricnetclose}

Structure of the `net.close` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.close",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 13687,
    "host": "1f0ec6c8a7bc",
    "unit": "connection",
    "summary": "true",
    "_time": 1643826403.12145
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.close",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 9871,
    "fd": 5,
    "host": "1f0ec6c8a7bc",
    "proto": "UDP",
    "port": 0,
    "unit": "connection",
    "_time": 1643826292.077388
  }
}
```

#### `net.close` properties {#metricnetcloseprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetclosebody)._ |

#### `net.close.body` properties {#metricnetclosebody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net Close<br/><br/>Value must be `net.close`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `unit` _required_ (`string`) | Unit - connection<br/><br/>Value must be `connection`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### dns.req [^](#schema-reference) {#metricdnsreq}

Structure of the `dns.req` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "dns.req",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "lt-curl",
    "pid": 31,
    "host": "2a6bc132b07a",
    "unit": "request",
    "summary": "true",
    "_time": 1643832467.795134
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "dns.req",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "lt-curl",
    "pid": 2485,
    "host": "2a6bc132b07a",
    "domain": "cribl.io",
    "duration": 0,
    "unit": "request",
    "_time": 1643832569.764219
  }
}
```

#### `dns.req` properties {#metricdnsreqprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricdnsreqbody)._ |

#### `dns.req.body` properties {#metricdnsreqbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net DNS<br/><br/>Value must be `dns.req`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `domain` (`string`) | domain |
| `duration` (`number`) | duration<br/><br/>**Example:**<br/>`55` |
| `unit` _required_ (`string`) | Unit - request<br/><br/>Value must be `request`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.duration [^](#schema-reference) {#metricnetduration}

Structure of the `net.duration` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.duration",
    "_metric_type": "timer",
    "_value": 1,
    "proc": "sendfile06_64",
    "pid": 15385,
    "host": "1f0ec6c8a7bc",
    "unit": "millisecond",
    "summary": "true",
    "_time": 1643826428.960074
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.duration",
    "_metric_type": "timer",
    "_value": 53,
    "proc": "send02",
    "pid": 11555,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "proto": "UDP",
    "port": 0,
    "numops": 1,
    "unit": "millisecond",
    "_time": 1643826318.65727
  }
}
```

#### `net.duration` properties {#metricnetdurationprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetdurationbody)._ |

#### `net.duration.body` properties {#metricnetdurationbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net duration<br/><br/>Value must be `net.duration`. |
| `_metric_type` _required_ (`string`) | timer<br/><br/>Value must be `timer`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `numops` (`number`) | numops |
| `unit` _required_ (`string`) | Unit - millisecond<br/><br/>Value must be `millisecond`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.error [^](#schema-reference) {#metricneterror}

Structure of the `net.error` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.error",
    "_metric_type": "counter",
    "_value": 6,
    "proc": "accept01",
    "pid": 5920,
    "host": "7cb66c7f77dd",
    "op": "summary",
    "class": "connection",
    "unit": "operation",
    "_time": 1643749774.573214
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.error",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "recv01",
    "pid": 3593,
    "host": "7cb66c7f77dd",
    "op": "recv",
    "class": "rx_tx",
    "unit": "operation",
    "_time": 1643749590.518109
  }
}
```

#### `net.error` properties {#metricneterrorprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricneterrorbody)._ |

#### `net.error.body` properties {#metricneterrorbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net Error<br/><br/>Value must be `net.error`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `op` (`string`) | op |
| `class` _required_ (`string`) | class net.error<br/><br/>**Possible values:**<ul><li>`connection`</li><li>`rx_tx`</li></ul> |
| `unit` _required_ (`string`) | Unit - operation<br/><br/>Value must be `operation`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.open [^](#schema-reference) {#metricnetopen}

Structure of the `net.open` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.open",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 13687,
    "host": "1f0ec6c8a7bc",
    "unit": "connection",
    "summary": "true",
    "_time": 1643826403.121437
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.open",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "lt-curl",
    "pid": 2485,
    "fd": 7,
    "host": "2a6bc132b07a",
    "proto": "UDP",
    "port": 0,
    "unit": "connection",
    "_time": 1643832569.764144
  }
}
```

#### `net.open` properties {#metricnetopenprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetopenbody)._ |

#### `net.open.body` properties {#metricnetopenbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net Open<br/><br/>Value must be `net.open`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `unit` _required_ (`string`) | Unit - connection<br/><br/>Value must be `connection`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.other [^](#schema-reference) {#metricnetother}

Structure of the `net.other` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.other",
    "_metric_type": "gauge",
    "_value": 1,
    "proc": "server_seqpacket",
    "pid": 234979,
    "host": "test_user",
    "unit": "connection",
    "summary": "true",
    "_time": 1643887036.00144
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.other",
    "_metric_type": "gauge",
    "_value": 1,
    "proc": "server_seqpacket",
    "pid": 235293,
    "fd": 4,
    "host": "test_user",
    "proto": "SEQPACKET",
    "port": 0,
    "unit": "connection",
    "_time": 1643887122.646226
  }
}
```

#### `net.other` properties {#metricnetotherprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetotherbody)._ |

#### `net.other.body` properties {#metricnetotherbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net other<br/><br/>Value must be `net.other`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `unit` _required_ (`string`) | Unit - connection<br/><br/>Value must be `connection`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.port [^](#schema-reference) {#metricnetport}

Structure of the `net.port` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.port",
    "_metric_type": "gauge",
    "_value": 2,
    "proc": "accept02",
    "pid": 13689,
    "host": "1f0ec6c8a7bc",
    "unit": "instance",
    "summary": "true",
    "_time": 1643826403.184484
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.port",
    "_metric_type": "gauge",
    "_value": 1,
    "proc": "accept01",
    "pid": 9871,
    "fd": 4,
    "host": "1f0ec6c8a7bc",
    "proto": "TCP",
    "port": 0,
    "unit": "instance",
    "_time": 1643826292.076967
  }
}
```

#### `net.port` properties {#metricnetportprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetportbody)._ |

#### `net.port.body` properties {#metricnetportbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net port<br/><br/>Value must be `net.port`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `port` (`number`) | port |
| `unit` _required_ (`string`) | Unit - instance<br/><br/>Value must be `instance`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.rx [^](#schema-reference) {#metricnetrx}

Structure of the `net.rx` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.rx",
    "_metric_type": "counter",
    "_value": 99000,
    "proc": "send02",
    "pid": 15371,
    "host": "1f0ec6c8a7bc",
    "unit": "byte",
    "class": "inet_udp",
    "summary": "true",
    "_time": 1643826428.564141
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.rx",
    "_metric_type": "counter",
    "_value": 6,
    "proc": "recvfrom01",
    "pid": 11544,
    "fd": 4,
    "host": "1f0ec6c8a7bc",
    "domain": "AF_INET",
    "proto": "TCP",
    "localip": "127.0.0.1",
    "localp": 37432,
    "remoteip": "0.0.0.0",
    "remotep": 40765,
    "data": "clear",
    "numops": 1,
    "unit": "byte",
    "_time": 1643826317.098972
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.rx",
    "_metric_type": "counter",
    "_value": 16,
    "proc": "send02",
    "pid": 11555,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "domain": "AF_INET",
    "proto": "UDP",
    "localip": "127.0.0.1",
    "localp": 0,
    "remoteip": " ",
    "remotep": 0,
    "data": "clear",
    "numops": 1,
    "unit": "byte",
    "_time": 1643826318.241899
  }
}
```

#### `net.rx` properties {#metricnetrxprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetrxbody)._ |

#### `net.rx.body` properties {#metricnetrxbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net RX<br/><br/>Value must be `net.rx`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `domain` (`string`) | domain |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `localn` (`number`) | localn |
| `localip` (`string`) | localip<br/><br/>**Example:**<br/>`127.0.0.1` |
| `localp` (`number`) | localp<br/><br/>**Example:**<br/>`9109` |
| `remoten` (`number`) | remoten |
| `remoteip` (`string`) | remoteip<br/><br/>**Example:**<br/>`192.158.1.38` |
| `remotep` (`number`) | remotep<br/><br/>**Example:**<br/>`9108` |
| `data` (`string`) | data |
| `numops` (`number`) | numops |
| `unit` _required_ (`string`) | Unit - byte<br/><br/>Value must be `byte`. |
| `class` (`string`) | class net.rxrx<br/><br/>**Possible values:**<ul><li>`inet_tcp`</li><li>`inet_udp`</li><li>`unix_tcp`</li><li>`unix_udp`</li><li>`other`</li></ul> |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.tcp [^](#schema-reference) {#metricnettcp}

Structure of the `net.tcp` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.tcp",
    "_metric_type": "gauge",
    "_value": 1,
    "proc": "accept02",
    "pid": 13689,
    "host": "1f0ec6c8a7bc",
    "unit": "connection",
    "summary": "true",
    "_time": 1643826403.184497
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.tcp",
    "_metric_type": "gauge",
    "_value": 0,
    "proc": "accept01",
    "pid": 9871,
    "fd": 4,
    "host": "1f0ec6c8a7bc",
    "proto": "TCP",
    "port": 0,
    "unit": "connection",
    "_time": 1643826292.07731
  }
}
```

#### `net.tcp` properties {#metricnettcpprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnettcpbody)._ |

#### `net.tcp.body` properties {#metricnettcpbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net tcp<br/><br/>Value must be `net.tcp`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `proto` (`string`) | proto_tcp<br/><br/>Value must be `TCP`. |
| `port` (`number`) | port |
| `unit` _required_ (`string`) | Unit - connection<br/><br/>Value must be `connection`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.tx [^](#schema-reference) {#metricnettx}

Structure of the `net.tx` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.tx",
    "_metric_type": "counter",
    "_value": 3,
    "proc": "recvmsg01",
    "pid": 15364,
    "host": "1f0ec6c8a7bc",
    "unit": "byte",
    "class": "unix_tcp",
    "summary": "true",
    "_time": 1643826427.279136
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.tx",
    "_metric_type": "counter",
    "_value": 16,
    "proc": "send02",
    "pid": 11555,
    "fd": 4,
    "host": "1f0ec6c8a7bc",
    "domain": "AF_INET",
    "proto": "UDP",
    "localip": "0.0.0.0",
    "localp": 0,
    "remoteip": "127.0.0.1",
    "remotep": 38725,
    "data": "clear",
    "numops": 1,
    "unit": "byte",
    "_time": 1643826318.241855
  }
}
```

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.tx",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "recvmsg01",
    "pid": 11548,
    "fd": 3,
    "host": "1f0ec6c8a7bc",
    "domain": "UNIX",
    "proto": "TCP",
    "localn": 48335,
    "remoten": 46396,
    "data": "clear",
    "numops": 1,
    "unit": "byte",
    "_time": 1643826317.162209
  }
}
```

#### `net.tx` properties {#metricnettxprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnettxbody)._ |

#### `net.tx.body` properties {#metricnettxbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net TX<br/><br/>Value must be `net.tx`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `domain` (`string`) | domain |
| `proto` (`string`) | proto<br/><br/>**Possible values:**<ul><li>`TCP`</li><li>`UDP`</li><li>`RAW`</li><li>`RDM`</li><li>`SEQPACKET`</li><li>`OTHER`</li></ul> |
| `localn` (`number`) | localn |
| `localip` (`string`) | localip<br/><br/>**Example:**<br/>`127.0.0.1` |
| `localp` (`number`) | localp<br/><br/>**Example:**<br/>`9109` |
| `remoten` (`number`) | remoten |
| `remoteip` (`string`) | remoteip<br/><br/>**Example:**<br/>`192.158.1.38` |
| `remotep` (`number`) | remotep<br/><br/>**Example:**<br/>`9108` |
| `data` (`string`) | data |
| `numops` (`number`) | numops |
| `unit` _required_ (`string`) | Unit - byte<br/><br/>Value must be `byte`. |
| `class` (`string`) | class net.rxrx<br/><br/>**Possible values:**<ul><li>`inet_tcp`</li><li>`inet_udp`</li><li>`unix_tcp`</li><li>`unix_udp`</li><li>`other`</li></ul> |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### net.udp [^](#schema-reference) {#metricnetudp}

Structure of the `net.udp` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.udp",
    "_metric_type": "gauge",
    "_value": 0,
    "proc": "accept01",
    "pid": 9871,
    "fd": 5,
    "host": "1f0ec6c8a7bc",
    "proto": "UDP",
    "port": 0,
    "unit": "connection",
    "_time": 1643826292.077372
  }
}
```

#### `net.udp` properties {#metricnetudpprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetudpbody)._ |

#### `net.udp.body` properties {#metricnetudpbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net udp<br/><br/>Value must be `net.udp`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` _required_ (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` _required_ (`string`) | host |
| `proto` _required_ (`string`) | proto_udp<br/><br/>Value must be `UDP`. |
| `port` _required_ (`number`) | port |
| `unit` _required_ (`string`) | Unit - connection<br/><br/>Value must be `connection`. |
| `summary` (`string`) | summary<br/><br/>Value must be `true`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.child [^](#schema-reference) {#metricprocchild}

Structure of the `proc.child` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.child",
    "_metric_type": "gauge",
    "_value": 0,
    "proc": "accept01",
    "pid": 1946,
    "host": "7cb66c7f77dd",
    "unit": "process",
    "_time": 1643749566.030543
  }
}
```

#### `proc.child` properties {#metricprocchildprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocchildbody)._ |

#### `proc.child.body` properties {#metricprocchildbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc child<br/><br/>Value must be `proc.child`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - process<br/><br/>Value must be `process`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.cpu [^](#schema-reference) {#metricproccpu}

Structure of the `proc.cpu` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.cpu",
    "_metric_type": "counter",
    "_value": 2107,
    "proc": "accept01",
    "pid": 1946,
    "host": "7cb66c7f77dd",
    "unit": "microsecond",
    "_time": 1643749566.030295
  }
}
```

#### `proc.cpu` properties {#metricproccpuprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricproccpubody)._ |

#### `proc.cpu.body` properties {#metricproccpubody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc cpu<br/><br/>Value must be `proc.cpu`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - microsecond<br/><br/>Value must be `microsecond`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.cpu.perc [^](#schema-reference) {#metricproccpuperc}

Structure of the `proc.cpu_perc` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.cpu_perc",
    "_metric_type": "gauge",
    "_value": 0.02107,
    "proc": "accept01",
    "pid": 1946,
    "host": "7cb66c7f77dd",
    "unit": "percent",
    "_time": 1643749566.030327
  }
}
```

#### `proc.cpu.perc` properties {#metricproccpupercprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricproccpupercbody)._ |

#### `proc.cpu.perc.body` properties {#metricproccpupercbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc cpu_perc<br/><br/>Value must be `proc.cpu_perc`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - percent<br/><br/>Value must be `percent`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.fd [^](#schema-reference) {#metricprocfd}

Structure of the `proc.fd` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.fd",
    "_metric_type": "gauge",
    "_value": 5,
    "proc": "accept01",
    "pid": 1946,
    "host": "7cb66c7f77dd",
    "unit": "file",
    "_time": 1643749566.030497
  }
}
```

#### `proc.fd` properties {#metricprocfdprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocfdbody)._ |

#### `proc.fd.body` properties {#metricprocfdbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc fd<br/><br/>Value must be `proc.fd`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - file<br/><br/>Value must be `file`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.mem [^](#schema-reference) {#metricprocmem}

Structure of the `proc.mem` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.mem",
    "_metric_type": "gauge",
    "_value": 31284,
    "proc": "accept01",
    "pid": 1946,
    "host": "7cb66c7f77dd",
    "unit": "kibibyte",
    "_time": 1643749566.030388
  }
}
```

#### `proc.mem` properties {#metricprocmemprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocmembody)._ |

#### `proc.mem.body` properties {#metricprocmembody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc memory<br/><br/>Value must be `proc.mem`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - kibibyte<br/><br/>Value must be `kibibyte`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.start [^](#schema-reference) {#metricprocstart}

Structure of the `proc.start` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.start",
    "_metric_type": "counter",
    "_value": 1,
    "proc": "accept01",
    "pid": 1945,
    "gid": 0,
    "groupname": "root",
    "uid": 0,
    "username": "root",
    "host": "7cb66c7f77dd",
    "args": "/opt/test/ltp/testcases/kernel/syscalls/accept/accept01",
    "unit": "process",
    "_time": 1643749566.026885
  }
}
```

#### `proc.start` properties {#metricprocstartprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocstartbody)._ |

#### `proc.start.body` properties {#metricprocstartbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc start<br/><br/>Value must be `proc.start`. |
| `_metric_type` _required_ (`string`) | counter<br/><br/>Value must be `counter`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `gid` _required_ (`integer`) | gid<br/><br/>**Example:**<br/>`0` |
| `groupname` _required_ (`string`) | groupname<br/><br/>**Example:**<br/>`root` |
| `uid` _required_ (`integer`) | uid<br/><br/>**Example:**<br/>`0` |
| `username` _required_ (`string`) | username<br/><br/>**Example:**<br/>`root` |
| `host` _required_ (`string`) | host |
| `args` _required_ (`string`) | args |
| `unit` _required_ (`string`) | Unit - process<br/><br/>Value must be `process`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |

<hr/>

### proc.thread [^](#schema-reference) {#metricprocthread}

Structure of the `proc.thread` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "proc.thread",
    "_metric_type": "gauge",
    "_value": 1,
    "proc": "accept01",
    "pid": 1946,
    "host": "7cb66c7f77dd",
    "unit": "thread",
    "_time": 1643749566.030435
  }
}
```

#### `proc.thread` properties {#metricprocthreadprops}

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocthreadbody)._ |

#### `proc.thread.body` properties {#metricprocthreadbody}

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - proc thread<br/><br/>Value must be `proc.thread`. |
| `_metric_type` _required_ (`string`) | gauge<br/><br/>Value must be `gauge`. |
| `_value` _required_ (`number`) | _value<br/><br/>**Example:**<br/>`1` |
| `proc` _required_ (`string`) | proc |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` _required_ (`string`) | host |
| `unit` _required_ (`string`) | Unit - thread<br/><br/>Value must be `thread`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
