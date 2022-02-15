---
title: Schema Reference
---

# Schema Reference

1. [Events](#events)
1. [Metrics](#metrics)

## Events

**DNS**

1. [dns.req](#eventdnsreq)
1. [dns.resp](#eventdnsresp)

**File System**

1. [fs.open](#eventfsopen)
1. [fs.close](#eventfsclose)
1. [fs.duration](#eventfsduration)
1. [fs.error](#eventfserror)
1. [fs.read](#eventfsread)
1. [fs.write](#eventfswrite)
1. [fs.delete](#eventfsdelete)
1. [fs.seek](#eventfsseek)
1. [fs.stat](#eventfsstat)

**HTTP**

1. [http.req](#eventhttpreq)
1. [http.resp](#eventhttpresp)

**Network**

1. [net.open](#eventnetopen)
1. [net.close](#eventnetclose)
1. [net.duration](#eventnetduration)
1. [net.error](#eventneterror)
1. [net.rx](#eventnetrx)
1. [net.tx](#eventnettx)
1. [net.app](#eventnetapp)
1. [net.port](#eventnetport)
1. [net.tcp](#eventnettcp)
1. [net.udp](#eventnetudp)
1. [net.other](#eventnetother)

**System Notification**

1. [notice](#eventnotice)

**stderr/stdout**

1. [event.stderr](#eventstderr)
1. [event.stdout](#eventstdout)

<span id="eventdnsreq"> </span>

### dns.req [^](#schema-reference)

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

#### `dns.req` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventdnsreqbody-properties)._ |

#### `dns.req.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - dns<br/><br/>Value must be `dns`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - DNS request<br/><br/>Value must be `dns.req`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventdnsreqbodydata-properties)._ |

#### `dns.req.body.data` properties

| Property | Description |
|---|---|
| `domain` _required_ (`string`) | domain |

<hr/>

<span id="eventdnsresp"> </span>

### dns.resp[^](#schema-reference)

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

#### `dns.resp` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventdnsrespbody-properties)._ |

#### `dns.resp.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - dns<br/><br/>Value must be `dns`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - DNS response<br/><br/>Value must be `dns.resp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventdnsrespbodydata-properties)._ |

#### `dns.resp.body.data` properties

| Property | Description |
|---|---|
| `duration` (`number`) | duration<br/><br/>**Example:**<br/>`55` |
| `domain` (`string`) | domain |
| `addrs` (`array`) | addrs |

<hr/>

<span id="eventfsclose"> </span>

### fs.close [^](#schema-reference)

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

#### `fs.close` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsclosebody-properties)._ |

#### `fs.close.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - fs<br/><br/>Value must be `fs`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Close<br/><br/>Value must be `fs.close`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsclosebodydata-properties)._ |

#### `fs.close.body.data` properties

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

<span id="eventfsdelete"> </span>

### fs.delete [^](#schema-reference)

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

#### `fs.delete` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsdeletebody-properties)._ |

#### `fs.delete.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - fs<br/><br/>Value must be `fs`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Delete<br/><br/>Value must be `fs.delete`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsdeletebodydata-properties)._ |

#### `fs.delete.body.data` properties

| Property | Description |
|---|---|
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `host` (`string`) | host |
| `op` (`string`) | op_fs_delete<br/><br/>**Possible values:**<ul><li>`unlink`</li><li>`unlinkat`</li></ul> |
| `file` (`string`) | file |
| `unit` (`string`) | Unit - operation<br/><br/>Value must be `operation`. |

<hr/>

<span id="eventfsduration"> </span>

### fs.duration [^](#schema-reference)

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

#### `fs.duration` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsdurationbody-properties)._ |

#### `fs.duration.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Duration<br/><br/>Value must be `fs.duration`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsdurationbodydata-properties)._ |

#### `fs.duration.body.data` properties

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

<span id="eventfserror"> </span>

### fs.error [^](#schema-reference)

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

#### `fs.error` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfserrorbody-properties)._ |

#### `fs.error.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Error<br/><br/>Value must be `fs.error`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfserrorbodydata-properties)._ |

#### `fs.error.body.data` properties

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

<span id="eventfsopen"> </span>

### fs.open [^](#schema-reference)

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

#### `fs.open` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsopenbody-properties)._ |

#### `fs.open.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - fs<br/><br/>Value must be `fs`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File open<br/><br/>Value must be `fs.open`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsopenbodydata-properties)._ |

#### `fs.open.body.data` properties

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

<span id="eventfsread"> </span>

### fs.read [^](#schema-reference)

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

#### `fs.read` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsreadbody-properties)._ |

#### `fs.read.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Read<br/><br/>Value must be `fs.read`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsreadbodydata-properties)._ |

#### `fs.read.body.data` properties

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

<span id="eventfsseek"> </span>

### fs.seek [^](#schema-reference)

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

#### `fs.seek` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsseekbody-properties)._ |

#### `fs.seek.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Seek<br/><br/>Value must be `fs.seek`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsseekbodydata-properties)._ |

#### `fs.seek.body.data` properties

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

<span id="eventfsstat"> </span>

### fs.stat [^](#schema-reference)

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

#### `fs.stat` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfsstatbody-properties)._ |

#### `fs.stat.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Stat<br/><br/>Value must be `fs.stat`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfsstatbodydata-properties)._ |

#### `fs.stat.body.data` properties

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

<span id="eventfswrite"> </span>

### fs.write [^](#schema-reference)

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

#### `fs.write` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventfswritebody-properties)._ |

#### `fs.write.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - File Write<br/><br/>Value must be `fs.write`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventfswritebodydata-properties)._ |

#### `fs.write.body.data` properties

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

<span id="eventhttpreq"> </span>

### http.req [^](#schema-reference)

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

#### `http.req` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventhttpreqbody-properties)._ |

#### `http.req.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - http<br/><br/>Value must be `http`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - HTTP request<br/><br/>Value must be `http.req`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventhttpreqbodydata-properties)._ |

#### `http.req.body.data` properties

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

<span id="eventhttpresp"> </span>

### http.resp [^](#schema-reference)

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

#### `http.resp` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventhttprespbody-properties)._ |

#### `http.resp.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - http<br/><br/>Value must be `http`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - HTTP response<br/><br/>Value must be `http.resp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventhttprespbodydata-properties)._ |

#### `http.resp.body.data` properties

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

<span id="eventnetapp"> </span>

### net.app [^](#schema-reference)

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

#### `net.app` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetappbody-properties)._ |

#### `net.app.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - net<br/><br/>Value must be `net`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net App<br/><br/>Value must be `net.app`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetappbodydata-properties)._ |

#### `net.app.body.data` properties

| Property | Description |
|---|---|
| `proc` (`string`) | proc |
| `pid` (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `fd` (`integer`) | fd<br/><br/>**Example:**<br/>`4` |
| `host` (`string`) | host |
| `protocol` (`string`) | protocol<br/><br/>**Possible values:**<ul><li>`HTTP`</li></ul> |

<hr/>

<span id="eventnetclose"> </span>

### net.close [^](#schema-reference)

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

#### `net.close` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetclosebody-properties)._ |

#### `net.close.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - net<br/><br/>Value must be `net`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net Close<br/><br/>Value must be `net.close`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetclosebodydata-properties)._ |

#### `net.close.body.data` properties

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

<span id="eventnetduration"> </span>

### net.duration [^](#schema-reference)

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

#### `net.duration` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetdurationbody-properties)._ |

#### `net.duration.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net duration<br/><br/>Value must be `net.duration`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetdurationbodydata-properties)._ |

#### `net.duration.body.data` properties

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

<span id="eventneterror"> </span>

### net.error [^](#schema-reference)

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

#### `net.error` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventneterrorbody-properties)._ |

#### `net.error.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net Error<br/><br/>Value must be `net.error`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventneterrorbodydata-properties)._ |

#### `net.error.body.data` properties

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

<span id="eventnetopen"> </span>

### net.open [^](#schema-reference)

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

#### `net.open` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetopenbody-properties)._ |

#### `net.open.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - net<br/><br/>Value must be `net`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net Open<br/><br/>Value must be `net.open`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetopenbodydata-properties)._ |

#### `net.open.body.data` properties

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

<span id="eventnetother"> </span>

### net.other [^](#schema-reference)

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

#### `net.other` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetotherbody-properties)._ |

#### `net.other.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net other<br/><br/>Value must be `net.other`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetotherbodydata-properties)._ |

#### `net.other.body.data` properties

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

<span id="eventnetport"> </span>

### net.port [^](#schema-reference)

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

#### `net.port` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetportbody-properties)._ |

#### `net.port.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net port<br/><br/>Value must be `net.port`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetportbodydata-properties)._ |

#### `net.port.body.data` properties

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

<span id="eventnetrx"> </span>

### net.rx [^](#schema-reference)

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

#### `net.rx` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetrxbody-properties)._ |

#### `net.rx.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net RX<br/><br/>Value must be `net.rx`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetrxbodydata-properties)._ |

#### `net.rx.body.data` properties

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

<span id="eventnettcp"> </span>

### net.tcp [^](#schema-reference)

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

#### `net.tcp` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnettcpbody-properties)._ |

#### `net.tcp.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net tcp<br/><br/>Value must be `net.tcp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnettcpbodydata-properties)._ |

#### `net.tcp.body.data` properties

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

<span id="eventnettx"> </span>

### net.tx [^](#schema-reference)

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

#### `net.tx` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnettxbody-properties)._ |

#### `net.tx.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net TX<br/><br/>Value must be `net.tx`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnettxbodydata-properties)._ |

#### `net.tx.body.data` properties

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

<span id="eventnetudp"> </span>

### net.udp [^](#schema-reference)

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

#### `net.udp` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnetudpbody-properties)._ |

#### `net.udp.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - metric<br/><br/>Value must be `metric`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - Net udp<br/><br/>Value must be `net.udp`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventnetudpbodydata-properties)._ |

#### `net.udp.body.data` properties

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

<span id="eventnotice"> </span>

### event.notice [^](#schema-reference)

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

#### `event.notice` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventnoticebody-properties)._ |

#### `notice.body` properties

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

<span id="eventstderr"> </span>

### event.stderr [^](#schema-reference)

Structure of the console `stderr` event

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

#### `event.stderr` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventstderrbody-properties)._ |

#### `stderr.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - console<br/><br/>Value must be `console`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - console stderr<br/><br/>Value must be `stderr`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventstderrbodydata-properties)._ |

#### `stderr.body.data` properties

| Property | Description |
|---|---|
| `message` (`string`) | message |

<hr/>

<span id="eventstdout"> </span>

### event.stdout [^](#schema-reference)

Structure of the console `stdout` event

#### Example

```json
{
  "type": "evt",
  "id": "ubuntu-sh- /usr/bin/which /usr/bin/firefox",
  "_channel": "13468365092424",
  "body": {
    "sourcetype": "console",
    "_time": 1643735941.602952,
    "source": "stdout",
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

#### `event.stdout` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes events from metrics.<br/><br/>Value must be `evt`. |
| `id` _required_ (`string`) | Identifies the application that the process is associated with. |
| `_channel` _required_ (`string`) | Identifies the operation during whose lifetime the event or metric is emitted. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#eventstdoutbody-properties)._ |

#### `stdout.body` properties

| Property | Description |
|---|---|
| `sourcetype` _required_ (`string`) | Sourcetype - console<br/><br/>Value must be `console`. |
| `_time` _required_ (`number`) | _time<br/><br/>**Example:**<br/>`1643662126.91777` |
| `source` _required_ (`string`) | Source - console stdout<br/><br/>Value must be `stdout`. |
| `host` _required_ (`string`) | host |
| `proc` _required_ (`string`) | proc |
| `cmd` _required_ (`string`) | cmd<br/><br/>**Example:**<br/>`top` |
| `pid` _required_ (`integer`) | pid<br/><br/>**Example:**<br/>`1000` |
| `data` _required_ (`object`) | data<br/><br/>_Details [below](#eventstdoutbodydata-properties)._ |

#### `stdout.body.data` properties

| Property | Description |
|---|---|
| `message` (`string`) | message |

<span id="metrics"> </span>

## Metrics [^](#schema-reference)

**File System**

1. [fs.open](#metricfsopen)
1. [fs.close](#metricfsclose)
1. [fs.duration](#metricfsduration)
1. [fs.error](#metricfserror)
1. [fs.read](#metricfsread)
1. [fs.write](#metricfswrite)
1. [fs.seek](#metricfsseek)
1. [fs.stat](#metricfsstat)

**HTTP**

1. [http.requests](#metrichttprequests)
1. [http.request.content.length](#metrichttprequestcontentlength)
1. [http.response.content.length](#metrichttpresponsecontentlength)
1. [http.client.duration](#metrichttpclientduration)
1. [http.server.duration](#metrichttpserverduration)

**Network**

1. [net.open](#metricnetopen)
1. [net.close](#metricnetclose)
1. [net.duration](#metricnetduration)
1. [net.error](#metricneterror)
1. [net.rx](#metricnetrx)
1. [net.tx](#metricnettx)
1. [net.dns](#metricnetdns)
1. [net.port](#metricnetport)
1. [net.tcp](#metricnettcp)
1. [net.udp](#metricnetudp)
1. [net.other](#metricnetother)

**Process**

1. [proc.fd](#metricprocfd)
1. [proc.thread](#metricprocthread)
1. [proc.start](#metricprocstart)
1. [proc.child](#metricprocchild)
1. [proc.cpu](#metricproccpu)
1. [proc.cpu.perc](#metricproccpuperc)
1. [proc.mem](#metricprocmem)

<hr/>

<span id="metricfsclose"> </span>

### fs.close [^](#schema-reference)

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

#### `fs.close` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsclosebody-properties)._ |

#### `fs.close.body` properties

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

<span id="metricfsduration"> </span>

### fs.duration [^](#schema-reference)

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

#### `fs.duration` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsdurationbody-properties)._ |

#### `fs.duration.body` properties

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

<span id="metricfserror"> </span>

### fs.error [^](#schema-reference)

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

#### `fs.error` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfserrorbody-properties)._ |

#### `fs.error.body` properties

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

<span id="metricfsopen"> </span>

### fs.open [^](#schema-reference)

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

#### `fs.open` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsopenbody-properties)._ |

#### `fs.open.body` properties

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

<span id="metricfsread"> </span>

### fs.read [^](#schema-reference)

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

#### `fs.read` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsreadbody-properties)._ |

#### `fs.read.body` properties

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

<span id="metricfsseek"> </span>

### fs.seek [^](#schema-reference)

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

#### `fs.seek` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsseekbody-properties)._ |

#### `fs.seek.body` properties

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

<span id="metricfsstat"> </span>

### fs.stat [^](#schema-reference)

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

#### `fs.stat` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfsstatbody-properties)._ |

#### `fs.stat.body` properties

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

<span id="metricfswrite"> </span>

### fs.write [^](#schema-reference)

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

#### `fs.write` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricfswritebody-properties)._ |

#### `fs.write.body` properties

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

<span id="metrichttpclientduration"> </span>

### http.client.duration [^](#schema-reference)

Structure of the `http.client.duration` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.client.duration",
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
    "_metric": "http.client.duration",
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

#### `http.client.duration` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpclientdurationbody-properties)._ |

#### `http.client.duration.body` properties

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP client duration<br/><br/>Value must be `http.client.duration`. |
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

<span id="metrichttprequestcontentlength"> </span>

### http.request.content.length [^](#schema-reference)

Structure of the `http.request.content_length` metric

#### Example

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.request.content_length",
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

#### `http.request.content.length` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttprequestcontentlengthbody-properties)._ |

#### `http.request.content.length.body` properties

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP request content length<br/><br/>Value must be `http.request.content_length`. |
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

<span id="metrichttprequests"> </span>

### http.requests [^](#schema-reference)

Structure of the `http.requests` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.requests",
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
    "_metric": "http.requests",
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

#### `http.requests` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttprequestsbody-properties)._ |

#### `http.requests.body` properties

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP requests<br/><br/>Value must be `http.requests`. |
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

<span id="metrichttpresponsecontentlength"> </span>

### http.response.content.length [^](#schema-reference)

Structure of the `http.response.content_length` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.response.content_length",
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
    "_metric": "http.response.content_length",
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

#### `http.response.content.length` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpresponsecontentlengthbody-properties)._ |

#### `http.response.content.length.body` properties

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP response content length<br/><br/>Value must be `http.response.content_length`. |
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

<span id="metrichttpserverduration"> </span>

### http.server.duration [^](#schema-reference)

Structure of the `http.server.duration` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "http.server.duration",
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
    "_metric": "http.server.duration",
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

#### `http.server.duration` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metrichttpserverdurationbody-properties)._ |

#### `http.server.duration.body` properties

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - HTTP server duration<br/><br/>Value must be `http.server.duration`. |
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

<span id="metricnetclose"> </span>

### net.close [^](#schema-reference)

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

#### `net.close` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetclosebody-properties)._ |

#### `net.close.body` properties

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

<span id="metricnetdns"> </span>

### net.dns [^](#schema-reference)

Structure of the `net.dns` metric

#### Examples

```json
{
  "type": "metric",
  "body": {
    "_metric": "net.dns",
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
    "_metric": "net.dns",
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

#### `net.dns` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetdnsbody-properties)._ |

#### `net.dns.body` properties

| Property | Description |
|---|---|
| `_metric` _required_ (`string`) | Source - Net DNS<br/><br/>Value must be `net.dns`. |
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

<span id="metricnetduration"> </span>

### net.duration [^](#schema-reference)

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

#### `net.duration` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetdurationbody-properties)._ |

#### `net.duration.body` properties

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

<span id="metricneterror"> </span>

### net.error [^](#schema-reference)

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

#### `net.error` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricneterrorbody-properties)._ |

#### `net.error.body` properties

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

<span id="metricnetopen"> </span>

### net.open [^](#schema-reference)

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

#### `net.open` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetopenbody-properties)._ |

#### `net.open.body` properties

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

<span id="metricnetother"> </span>

### net.other [^](#schema-reference)

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

#### `net.other` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetotherbody-properties)._ |

#### `net.other.body` properties

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

<span id="metricnetport"> </span>

### net.port [^](#schema-reference)

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

#### `net.port` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetportbody-properties)._ |

#### `net.port.body` properties

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

<span id="metricnetrx"> </span>

### net.rx [^](#schema-reference)

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

#### `net.rx` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetrxbody-properties)._ |

#### `net.rx.body` properties

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

<span id="metricnettcp"> </span>

### net.tcp [^](#schema-reference)

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

#### `net.tcp` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnettcpbody-properties)._ |

#### `net.tcp.body` properties

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

<span id="metricnettx"> </span>

### net.tx [^](#schema-reference)

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

#### `net.tx` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnettxbody-properties)._ |

#### `net.tx.body` properties

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

<span id="metricnetudp"> </span>

### net.udp [^](#schema-reference)

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

#### `net.udp` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricnetudpbody-properties)._ |

#### `net.udp.body` properties

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

<span id="metricprocchild"> </span>

### proc.child [^](#schema-reference)

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

#### `proc.child` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocchildbody-properties)._ |

#### `proc.child.body` properties

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

<span id="metricproccpu"> </span>

### proc.cpu [^](#schema-reference)

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

#### `proc.cpu` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricproccpubody-properties)._ |

#### `proc.cpu.body` properties

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

<span id="metricproccpuperc"> </span>

### proc.cpu.perc [^](#schema-reference)

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

#### `proc.cpu.perc` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricproccpupercbody-properties)._ |

#### `proc.cpu.perc.body` properties

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

<span id="metricprocfd"> </span>

### proc.fd [^](#schema-reference)

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

#### `proc.fd` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocfdbody-properties)._ |

#### `proc.fd.body` properties

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

<span id="metricprocmem"> </span>

### proc.mem [^](#schema-reference)

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

#### `proc.mem` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocmembody-properties)._ |

#### `proc.mem.body` properties

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

<span id="metricprocstart"> </span>

### proc.start [^](#schema-reference)

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

#### `proc.start` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocstartbody-properties)._ |

#### `proc.start.body` properties

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

<span id="metricprocthread"> </span>

### proc.thread [^](#schema-reference)

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

#### `proc.thread` properties

| Property | Description |
|---|---|
| `type` _required_ (`string`) | Distinguishes metrics from events.<br/><br/>Value must be `metric`. |
| `body` _required_ (`object`) | body<br/><br/>_Details [below](#metricprocthreadbody-properties)._ |

#### `proc.thread.body` properties

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
