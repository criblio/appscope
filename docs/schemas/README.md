# JSON Schema

AppScope generates events and metrics whose structure is defined in JSON Schema. Each event and metric is defined in its own JSON Schema file in this directory. 

To generate a set of Markdown files from the JSON Schema files, call:

```
make docs-generate
```

The Markdown files will appear in `docs/md_files`.

This README also explains how to add a new event or metric.

## Contents
- [JSON Schema](#json-schema)
  - [Contents](#contents)
  - [Defining New Events or Metrics in JSON Schema](#defining-new-events-or-metrics-in-json-schema)
    - [Event](#event)
    - [Metric](#metric)

## Defining New Events or Metrics in JSON Schema

[JSON Schema](https://json-schema.org/) is a vocabulary with which to annotate and validate JSON documents.

### Event

To add a new event named `foo.bar`, create a file named `event_foo_bar.schema.json`.

- `root` section required update of `"$id"` and `"description"` fields
- `body` section required update value of following `$ref` in properties in `"sourcetype` and `"source"`
- `data` section required update of expected `properties` which will be in `data` section of event, in
  template below - example `properties` are presented as `property_name_1` and `property_name_2`.

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://appscope.dev/docs/schemas/event_foo_bar.schema.json",
  "type": "object",
  "title": "AppScope Event",
  "description": "Structure of the `foo.bar` event",
  "required": [
    "type",
    "id",
    "_channel",
    "body"
  ],
  "properties": {
    "type": {
      "$ref": "definitions/metadata.schema.json#/$defs/event_type"
    },
    "id": {
      "$ref": "definitions/metadata.schema.json#/$defs/id"
    },
    "_channel": {
      "$ref": "definitions/metadata.schema.json#/$defs/_channel"
    },
    "body": {
      "title": "body",
      "description": "body",
      "type": "object",
      "required": [
        "sourcetype",
        "_time",
        "source",
        "host",
        "proc",
        "cmd",
        "pid",
        "data"
      ],
      "properties": {
        "sourcetype": {
          "$ref": "definitions/body.schema.json#/$defs/sourcetypefootype"
        },
        "_time": {
          "$ref": "definitions/body.schema.json#/$defs/_time"
        },
        "source": {
          "$ref": "definitions/body.schema.json#/$defs/sourcefoobar"
        },
        "host": {
          "$ref": "definitions/data.schema.json#/$defs/host"
        },
        "proc": {
          "$ref": "definitions/data.schema.json#/$defs/proc"
        },
        "cmd": {
          "$ref": "definitions/body.schema.json#/$defs/cmd"
        },
        "pid": {
          "$ref": "definitions/data.schema.json#/$defs/pid"
        },
        "data": {
          "title": "data",
          "description": "data",
          "type": "object",
          "properties": {
            "property_name_1": {
              "$ref": "definitions/data.schema.json#/$defs/property_name_1"
            },
            "property_name_2": {
              "$ref": "definitions/data.schema.json#/$defs/property_name_2"
            }
          }
        }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}

```

### Metric

To add a new metric named `foo.bar`, create a file named `metric_foo_bar.schema.json`.

- `root` section required update of `"$id"` and `"description"` fields
- `body` section required update value of following `$ref` in properties in `"_metric` and `"_metric_type"`
- `body` section required update of expected `properties` which will be in `body` section of metrics, in
  template below - example `properties` is presented as `property_name_1`.

```
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://appscope.dev/docs/schemas/metric_foo_bar.schema.json",
  "type": "object",
  "title": "AppScope Metric",
  "description": "Structure of the `foo.bar` metric",
  "required": [
    "type",
    "body"
  ],
  "properties": {
    "type": {
      "$ref": "definitions/metadata.schema.json#/$defs/metric_type"
    },
    "body": {
      "title": "body",
      "description": "body",
      "type": "object",
      "required": [
        "_metric",
        "_metric_type",
        "_value",
        "property_name_1",
        "_time"
      ],
      "properties": {
        "_metric": {
          "$ref": "definitions/body.schema.json#/$defs/sourcefoobar"
        },
        "_metric_type": {
          "$ref": "definitions/body.schema.json#/$defs/metric_type_foo"
        },
        "_value": {
          "$ref": "definitions/body.schema.json#/$defs/_value"
        },
        "property_name_1": {
          "$ref": "definitions/body.schema.json#/$defs/property_name_1"
        },
        "_time": {
          "$ref": "definitions/body.schema.json#/$defs/_time"
        }
      }
    }
  },
  "additionalProperties": false
}

```
