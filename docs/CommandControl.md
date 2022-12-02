# Command and Control in Appscope

AppScope provides two mechanisms to affect the behavior of a running process. The first is a file-based mechanism we've called Dynamic Configuration.  The other is a socket-based request/response interface I'll call Socket Interface. The initial use case for both of these mechanisms is to be able to change the configuration of the AppScope library after the process is up and running. Another more recent use case includes the ability to detach and reattach to the process [link](https://github.com/criblio/appscope/blob/master/docs/Attach.md).  This page is intended to document the basics of these mechanisms including the format of requests, and expected responses.

## Dynamic Configuration

This feature provides an "environment variable-like" interface that can be used to alter the current configuration of a running process.

[This page](https://appscope.dev/docs/cli-using#dynamic-configuration) has a nice description of this file-based mechanism with examples for using this feature and what the "environment variable" format looks like.  The library provides no success or failure response when using the Dynamic Configuration mechanism, though the library will delete the scope.<pid> file after it's completed processing the file.

All configuration environment variables can be specified in this file. Values and names of the configuration environment variables can be found [here](https://appscope.dev/docs/config-file/#scopeyml-config-file) in comments "Values:" and "Override:" respectively.  By our own convention, all environment variable names used by AppScope start with "SCOPE_".

There are also a few environment variables that are unique to the Dynamic Configuration interface; they do not directly change configuration.  These act more like commands, and are listed here with brief descriptions:
* SCOPE_CMD_DBG_PATH - Can be used to dump limited debug information by specifying an absolute path.  The file will be created if necessary, if path describes an existing file, new data will be appended to the end of the file. In the first of these cases, the directory in which the file is to be created must already exist.
* SCOPE_CMD_ATTACH - "false" unscopes the process, stopping function interposition](https://appscope.dev/docs/how-works#how-appscope-works).  "true" reestablishes function interposition.
* SCOPE_CONF_RELOAD - Can be used to change the configuration by supplying an absolute file path to a [scope.yml](https://appscope.dev/docs/config-file/#scopeyml-config-file) file. There is no requirement for all configuration fields to be supplied, though library defaults will be used for each field which is omitted.

## Socket Interface

This feature accepts incoming requests, and for each request sends a response indicating if the request was seen as valid or not. There are only a few valid requests that can be issued to the library.  They can be issued over an event transport or cribl transport (via a unix socket, or tcp socket over the network, but not currently over a TLS connection).

### Requests

Socket Interface requests are made by sending newline-delimited json to the library.  All requests must have three fields - "type", "req", and "reqId".  Most will also have a field named "body".  Assuming the whole message is valid (parseable) json, fields of any other name are allowed are allowed in a request, but will be ignored by the library.

For every request, the value of the "type" field must be the string "req".
This lists supported string values for "req" field. For each value of "req", the expectations for the "body" are described:
* GetCfg - This request needs no body field.  The response will contain a "body" field (json representation) with the current AppScope configuration. 
* SetCfg - The "body" field must be a json representation of the configuration described [here](https://appscope.dev/docs/config-file/#scopeyml-config-file).  There is no requirement for all configuration fields to be supplied, though library defaults will be used for each field which is omitted. 
* Switch - The "body" field may have string values "detach" and "attach".  "detach" unscopes the process, stopping [function interposition](https://appscope.dev/docs/how-works#how-appscope-works).  "Attach" reestablishes function interposition.

The value of "reqId" can be any integer that can be represented in 64 bits.  There are no other requirements for this value.

### Responses

The library will generate a newline-delimited response message for each request the library receives over the Socket Interface.  The response will always contain three fields - "type", "reqId", and "status".  Successful responses will contain always contain a "req" field, unsuccessful responses will contain a "message" field.  Most successful responses will also contain a "body" field as well.

For every response, the value of "type" will be "resp", and the value of "reqId" will be equal to whatever value of "reqId" was provided in the request.
The value of "status" will be an integer - 200 indicates success, 400 indicates failure.  When the "status" has a value of 400, a "message" field will exist and have a string that's intended to help a developer debug what's seen as wrong with the request, such as: 
* Request could not be parsed as a json object
* "type" was not "req", or required fields were missing or of wrong type
* "req" field was not an expected value
* Based on the "req" field, expected fields were missing

The whenever the request can be parsed, the value of the field "req" will be equal to whatever value of "req" was provided in the request.

### Example Requests

GetCfg request:

    {
        "type": "req",
        "req": "GetCfg",
        "reqId": 2
    }

SetCfg request:

    {
        "type": "req",
        "req": "SetCfg",
        "reqId": 3,
        "body": {
            "metric": {
                "format": {
                    "type": "metricjson",
                    "statsdprefix": "cribl.scope",
                    "statsdmaxlen": "42",
                    "verbosity": "0",
                    "tags": [
                        {
                            "tagA": "val1"
                        },
                        {
                            "tagB": "val2"
                        }
                    ]
                },
                "transport": {
                    "type": "file",
                    "path": "/var/log/scope.log"
                }
            },
            "event": {
                "format": {
                    "type": "ndjson"
                },
                "watch": [
                    {
                        "type": "file",
                        "name": ".*[.]log$"
                    },
                    {
                        "type": "console"
                    },
                    {
                        "type": "syslog"
                    },
                    {
                        "type": "metric"
                    }
                ]
            },
            "libscope": {
                "transport": {
                    "type": "file",
                    "path": "/var/log/event.log"
                },
                "summaryperiod": "13",
                "log": {
                    "level": "debug",
                    "transport": {
                        "type": "shm"
                    }
                }
            }
        }
    }

Switch request:

    {
        "type": "req",
        "req": "Switch",
        "reqId": 4,
        "body": "attach"
    }
