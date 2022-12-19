OVERVIEW:
The Scope library supports extraction of data from within applications.
As a general rule, applications consist of one or more processes.
The Scope library can be loaded into any process as the
process starts.
The primary way to define which processes include the Scope library
is by exporting the environment variable LD_PRELOAD, which is set to point
to the path name of the Scope library. E.g.: 
export LD_PRELOAD=./libscope.so

Scope emits data as metrics and/or events.
Scope is fully configurable by means of a configuration file (scope.yml)
and/or environment variables.

Metrics are emitted in StatsD format, over a configurable link. By default,
metrics are sent over a UDP socket using localhost and port 8125.

Events are emitted in JSON format over a configurable link. By default,
events are sent over a TCP socket using localhost and port 9109.

Scope logs to a configurable destination, at a configurable
verbosity level. The default verbosity setting is level 4, and the
default destination is the file `/tmp/scope.log`.

CONFIGURATION:
Configuration File:
    A YAML config file (named scope.yml) enables control of all available
    settings. The config file is optional. Environment variables take
    precedence over settings in a config file.

Config File Resolution
    If the SCOPE_CONF_PATH env variable is defined, and points to a
    file that can be opened, it will use this as the config file.
    Otherwise, AppScope searches for the config file in this priority
    order, using the first one it finds:

        $SCOPE_HOME/conf/scope.yml
        $SCOPE_HOME/scope.yml
        /etc/scope/scope.yml
        ~/conf/scope.yml
        ~/scope.yml
        ./conf/scope.yml
        ./scope.yml

Environment Variables:
    SCOPE_CONF_PATH
        Directly specify config file's location and name.
        Used only at start time.
    SCOPE_HOME
        Specify a directory from which conf/scope.yml or ./scope.yml can
        be found. Used only at start time, and only if SCOPE_CONF_PATH
        does not exist. For more info, see Config File Resolution below.
    SCOPE_METRIC_ENABLE
        Single flag to make it possible to disable all metric output.
        true,false  Default is true.
    SCOPE_METRIC_VERBOSITY
        0-9 are valid values. Default is 4.
        For more info, see Metric Verbosity below.
    SCOPE_METRIC_FS
        Create metrics describing file connectivity.
        true, false  Default is true.
    SCOPE_METRIC_NET
        Create metrics describing network connectivity.
        true, false  Default is true.
    SCOPE_METRIC_HTTP
        Create metrics describing HTTP communication.
        true, false  Default is true.
    SCOPE_METRIC_DNS
        Create metrics describing DNS activity.
        true, false  Default is true.
    SCOPE_METRIC_PROC
        Create metrics describing process.
        true, false  Default is true.
    SCOPE_METRIC_STATSD
        When true, statsd metrics sent or received by this application
        will be handled as appscope-native metrics.
        true, false  Default is true.
    SCOPE_METRIC_DEST
        Default is udp://localhost:8125
        Format is one of:
            file:///tmp/output.log
                Output to a file. Use file://stdout, file://stderr for
                STDOUT or STDERR
            udp://host:port
            tcp://host:port
                Send to a TCP or UDP server. \"host\" is the hostname or
                IP address and \"port\" is the port number of service name.
            unix://socketpath
                Output to a unix domain server using TCP.
                Use unix://@abstractname, unix:///var/run/mysock for
                abstract address or filesystem address.
    SCOPE_METRIC_TLS_ENABLE
        Flag to enable Transport Layer Security (TLS). Only affects
        tcp:// destinations. true,false  Default is false.
    SCOPE_METRIC_TLS_VALIDATE_SERVER
        false allows insecure (untrusted) TLS connections, true uses
        certificate validation to ensure the server is trusted.
        Default is true.
    SCOPE_METRIC_TLS_CA_CERT_PATH
        Path on the local filesystem which contains CA certificates in
        PEM format. Default is an empty string. For a description of what
        this means, see Certificate Authority Resolution below.
    SCOPE_METRIC_FORMAT
        statsd, ndjson
        Default is statsd.
    SCOPE_STATSD_PREFIX
        Specify a string to be prepended to every scope metric.
    SCOPE_STATSD_MAXLEN
        Default is 512.
    SCOPE_SUMMARY_PERIOD
        Number of seconds between output summarizations. Default is 10.
    SCOPE_EVENT_ENABLE
        Single flag to make it possible to disable all event output.
        true,false  Default is true.
    SCOPE_EVENT_DEST
        Same format as SCOPE_METRIC_DEST above.
        Default is tcp://localhost:9109
    SCOPE_EVENT_TLS_ENABLE
        Flag to enable Transport Layer Security (TLS). Only affects
        tcp:// destinations. true,false  Default is false.
    SCOPE_EVENT_TLS_VALIDATE_SERVER
        false allows insecure (untrusted) TLS connections, true uses
        certificate validation to ensure the server is trusted.
        Default is true.
    SCOPE_EVENT_TLS_CA_CERT_PATH
        Path on the local filesystem which contains CA certificates in
        PEM format. Default is an empty string. For a description of what
        this means, see Certificate Authority Resolution below.
    SCOPE_EVENT_FORMAT
        ndjson
        Default is ndjson.
    SCOPE_EVENT_LOGFILE
        Create events from writes to log files.
        true,false  Default is false.
    SCOPE_EVENT_LOGFILE_NAME
        An extended regex to filter log file events by file name.
        Used only if SCOPE_EVENT_LOGFILE is true. Default is .*log.*
    SCOPE_EVENT_LOGFILE_VALUE
        An extended regex to filter log file events by field value.
        Used only if SCOPE_EVENT_LOGFILE is true. Default is .*
    SCOPE_EVENT_CONSOLE
        Create events from writes to stdout, stderr.
        true,false  Default is false.
    SCOPE_EVENT_CONSOLE_NAME
        An extended regex to filter console events by event name.
        Used only if SCOPE_EVENT_CONSOLE is true. Default is .*
    SCOPE_EVENT_CONSOLE_VALUE
        An extended regex to filter console events by field value.
        Used only if SCOPE_EVENT_CONSOLE is true. Default is .*
    SCOPE_EVENT_METRIC
        Create events from metrics.
        true,false  Default is false.
    SCOPE_EVENT_METRIC_NAME
        An extended regex to filter metric events by event name.
        Used only if SCOPE_EVENT_METRIC is true. Default is .*
    SCOPE_EVENT_METRIC_FIELD
        An extended regex to filter metric events by field name.
        Used only if SCOPE_EVENT_METRIC is true. Default is .*
    SCOPE_EVENT_METRIC_VALUE
        An extended regex to filter metric events by field value.
        Used only if SCOPE_EVENT_METRIC is true. Default is .*
    SCOPE_EVENT_HTTP
        Create events from HTTP headers.
        true,false  Default is false.
    SCOPE_EVENT_HTTP_NAME
        An extended regex to filter http events by event name.
        Used only if SCOPE_EVENT_HTTP is true. Default is .*
    SCOPE_EVENT_HTTP_FIELD
        An extended regex to filter http events by field name.
        Used only if SCOPE_EVENT_HTTP is true. Default is .*
    SCOPE_EVENT_HTTP_VALUE
        An extended regex to filter http events by field value.
        Used only if SCOPE_EVENT_HTTP is true. Default is .*
    SCOPE_EVENT_HTTP_HEADER
        An extended regex that defines user defined headers
        that will be extracted. Default is NULL
    SCOPE_EVENT_NET
        Create events describing network connectivity.
        true,false  Default is false.
    SCOPE_EVENT_NET_NAME
        An extended regex to filter network events by event name.
        Used only if SCOPE_EVENT_NET is true. Default is .*
    SCOPE_EVENT_NET_FIELD
        An extended regex to filter network events by field name.
        Used only if SCOPE_EVENT_NET is true. Default is .*
    SCOPE_EVENT_NET_VALUE
        An extended regex to filter network events by field value.
        Used only if SCOPE_EVENT_NET is true. Default is .*
    SCOPE_EVENT_FS
        Create events describing file connectivity.
        true,false  Default is false.
    SCOPE_EVENT_FS_NAME
        An extended regex to filter file events by event name.
        Used only if SCOPE_EVENT_FS is true. Default is .*
    SCOPE_EVENT_FS_FIELD
        An extended regex to filter file events by field name.
        Used only if SCOPE_EVENT_FS is true. Default is .*
    SCOPE_EVENT_FS_VALUE
        An extended regex to filter file events by field value.
        Used only if SCOPE_EVENT_FS is true. Default is .*
    SCOPE_EVENT_DNS
        Create events describing DNS activity.
        true,false  Default is false.
    SCOPE_EVENT_DNS_NAME
        An extended regex to filter dns events by event name.
        Used only if SCOPE_EVENT_DNS is true. Default is .*
    SCOPE_EVENT_DNS_FIELD
        An extended regex to filter DNS events by field name.
        Used only if SCOPE_EVENT_DNS is true. Default is .*
    SCOPE_EVENT_DNS_VALUE
        An extended regex to filter dns events by field value.
        Used only if SCOPE_EVENT_DNS is true. Default is .*
    SCOPE_EVENT_MAXEPS
        Limits number of events that can be sent in a single second.
        0 is 'no limit'; 10000 is the default.
    SCOPE_ENHANCE_FS
        Controls whether uid, gid, and mode are captured for each open.
        Used only if SCOPE_EVENT_FS is true. true,false Default is true.
    SCOPE_LOG_LEVEL
        debug, info, warning, error, none. Default is error.
    SCOPE_LOG_DEST
        same format as SCOPE_METRIC_DEST above.
        Default is file:///tmp/scope.log
    SCOPE_LOG_TLS_ENABLE
        Flag to enable Transport Layer Security (TLS). Only affects
        tcp:// destinations. true,false  Default is false.
    SCOPE_LOG_TLS_VALIDATE_SERVER
        false allows insecure (untrusted) TLS connections, true uses
        certificate validation to ensure the server is trusted.
        Default is true.
    SCOPE_LOG_TLS_CA_CERT_PATH
        Path on the local filesystem which contains CA certificates in
        PEM format. Default is an empty string. For a description of what
        this means, see Certificate Authority Resolution below.
    SCOPE_TAG_
        Specify a tag to be applied to every metric and event.
        Environment variable expansion is available, 
        e.g.: SCOPE_TAG_user=$USER
    SCOPE_CMD_DIR
        Specifies a directory to look for dynamic configuration files.
        See Dynamic Configuration below.
        Default is /tmp
    SCOPE_PAYLOAD_ENABLE
        Flag that enables payload capture.  true,false  Default is false.
    SCOPE_PAYLOAD_DIR
        Specifies a directory where payload capture files can be written.
        Default is /tmp
    SCOPE_CRIBL_ENABLE
        Single flag to make it possible to disable cribl backend.
        true,false  Default is true.
    SCOPE_CRIBL_CLOUD
        Intended as an alternative to SCOPE_CRIBL below. Identical
        behavior, except that where SCOPE_CRIBL can have TLS settings
        modified via related SCOPE_CRIBL_TLS_* environment variables,
        SCOPE_CRIBL_CLOUD hardcodes TLS settings as though these were
        individually specified:
            SCOPE_CRIBL_TLS_ENABLE=true
            SCOPE_CRIBL_TLS_VALIDATE_SERVER=true
            SCOPE_CRIBL_TLS_CA_CERT_PATH=\"\"
        As a note, library behavior will be undefined if this variable is
        set simultaneously with SCOPE_CRIBL, or any of SCOPE_CRIBL_TLS_*.
    SCOPE_CRIBL
        Defines a connection with Cribl LogStream
        Default is NULL
        Format is one of:
            tcp://host:port
                If no port is provided, defaults to 10090
            unix://socketpath
                Output to a unix domain server using TCP.
                Use unix://@abstractname, unix:///var/run/mysock for
                abstract address or filesystem address.
    SCOPE_CRIBL_AUTHTOKEN
        Authentication token provided by Cribl.
        Default is an empty string.
    SCOPE_CRIBL_TLS_ENABLE
        Flag to enable Transport Layer Security (TLS). Only affects
        tcp:// destinations. true,false  Default is false.
    SCOPE_CRIBL_TLS_VALIDATE_SERVER
        false allows insecure (untrusted) TLS connections, true uses
        certificate validation to ensure the server is trusted.
        Default is true.
    SCOPE_CRIBL_TLS_CA_CERT_PATH
        Path on the local filesystem which contains CA certificates in
        PEM format. Default is an empty string. For a description of what
        this means, see Certificate Authority Resolution below.
    SCOPE_CONFIG_EVENT
        Sends a single process-identifying event, when a transport
        connection is established.  true,false  Default is true.

Dynamic Configuration:
    Dynamic Configuration allows configuration settings to be
    changed on the fly after process start time. At every
    SCOPE_SUMMARY_PERIOD, the library looks in SCOPE_CMD_DIR to
    see if a file scope.<pid> exists. If it exists, the library processes
    every line, looking for environment variable–style commands
    (e.g., SCOPE_CMD_DBG_PATH=/tmp/outfile.txt). The library changes the
    configuration to match the new settings, and deletes the
    scope.<pid> file when it's complete.

    While every environment variable defined here can be changed via
    Dynamic Configuration, there are a few environment variable-style
    commands that are only accepted during Dynamic Configuration.
    These will be ignored if specified as actual environment variables.
    They are listed here:
        SCOPE_CMD_ATTACH
            Flag that controls whether interposed functions are
            processed by AppScope (true), or not (false).
        SCOPE_CMD_DBG_PATH
            Causes AppScope to write debug information to the
            specified file path. Absolute paths are recommended.
        SCOPE_CONF_RELOAD
            Causes AppScope to reload its configuration file. If
            the value is \"true\", it is reloaded according to the
            Config File Resolution above. If any other value is
            specified, it is handled like SCOPE_CONF_PATH, but without
            the \"Used only at start time\" limitation.

Certificate Authority Resolution
    If SCOPE_*_TLS_ENABLE and SCOPE_*_TLS_VALIDATE_SERVER are true then
    AppScope performs TLS server validation. For this to be successful
    a CA certificate must be found that can authenticate the certificate
    the server provides during the TLS handshake process.
    If SCOPE_*_TLS_CA_CERT_PATH is set, AppScope will use this file path
    which is expected to contain CA certificates in PEM format. If this
    env variable is an empty string or not set, AppScope searches for a
    usable root CA file on the local filesystem, using the first one
    found from this list:

        /etc/ssl/certs/ca-certificates.crt
        /etc/pki/tls/certs/ca-bundle.crt
        /etc/ssl/ca-bundle.pem
        /etc/pki/tls/cacert.pem
        /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
        /etc/ssl/cert.pem

METRICS:
Metrics can be enabled or disabled with a single config element
(metric: enable: true|false). Specific types of metrics, and specific 
field content, are managed with a Metric Verbosity setting.

Metric Verbosity
    Controls two different aspects of metric output – 
    Tag Cardinality and Summarization.

    Tag Cardinality
        0   No expanded StatsD tags
        1   adds 'data', 'unit'
        2   adds 'class', 'proto'
        3   adds 'op'
        4   adds 'pid', 'host', 'proc', 'http_status'
        5   adds 'domain', 'file'
        6   adds 'localip', 'remoteip', 'localp', 'port', 'remotep'
        7   adds 'fd', 'args'
        8   adds 'duration','numops','req_per_sec','req','resp','protocol'

    Summarization
        0-4 has full event summarization
        5   turns off 'error'
        6   turns off 'filesystem open/close' and 'dns'
        7   turns off 'filesystem stat' and 'network connect'
        8   turns off 'filesystem seek'
        9   turns off 'filesystem read/write' and 'network send/receive'

The http.status metric is emitted when the http watch type has been
enabled as an event. The http.status metric is not controlled with
summarization settings.

EVENTS:
All events can be enabled or disabled with a single config element
(event: enable: true|false). Unlike metrics, event content is not 
managed with verbosity settings. Instead, you use regex filters that 
manage which field types and field values to include.

 Events are organized as 7 watch types: 
 1) File Content. Provide a pathname, and all data written to the file
    will be organized in JSON format and emitted over the event channel.
 2) Console Output. Select stdin and/or stdout, and all data written to
    these endpoints will be formatted in JSON and emitted over the event
    channel.
 3) Metrics. Event metrics provide the greatest level of detail from
    libscope. Events are created for every read, write, send, receive,
    open, close, and connect. These raw events are configured with regex
    filters to manage which event, which specific fields within an event,
    and which value patterns within a field to include.
 4) HTTP Headers. HTTP headers are extracted, formatted in JSON, and
    emitted over the event channel. Three types of events are created
    for HTTP headers: 1) HTTP request events, 2) HTTP response events,
    and 3) a metric event corresponding to the request and response
    sequence. A response event includes the corresponding request,
    status and duration fields. An HTTP metric event provides fields
    describing bytes received, requests per second, duration, and status.
    Any header defined as X-appscope (case insensitive) will be emitted.
    User defined headers are extracted by using the headers field.
    The headers field is a regular expression.
 5) File System. Events are formatted in JSON for each file system open,
    including file name, permissions, and cgroup. Events for file system
    close add a summary of the number of bytes read and written, the
    total number of read and write operations, and the total duration
    of read and write operations. The specific function performing open
    and close is reported as well.
 6) Network. Events are formatted in JSON for network connections and 
    corresponding disconnects, including type of protocol used, and 
    local and peer IP:port. Events for network disconnect add a summary
    of the number of bytes sent and received, and the duration of the
    sends and receives while the connection was active. The reason
    (source) for disconnect is provided as local or remote. 
 7) DNS. Events are formatted in JSON for DNS requests and responses.
    The event provides the domain name being resolved. On DNS response,
    the event provides the duration of the DNS operation.

PROTOCOL DETECTION:
Scope can detect any defined network protocol. You provide protocol
definitions in the \"protocol\" section of the config file. You describe 
protocol specifics in one or more regex definitions. PCRE2 regular 
expressions are supported. The stock scope.yml file for examples.

Scope detects binary and string protocols. Detection events, 
formatted in JSON, are emitted over the event channel unless the \"detect\"
property is set to \"false\".

PAYLOAD EXTRACTION:
When enabled, libscope extracts payload data from network operations.
Payloads are emitted in binary. No formatting is applied to the data.
Payloads are emitted to either a local file or the LogStream channel.
Configuration elements for libscope support defining a path for payload
data.
