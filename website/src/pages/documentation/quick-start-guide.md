# Quick-Start Guide

First, see <link>[Requirements](#bookmark=id.2gpv5bl9l16m)</link> to ensure that you’re completing these steps on a supported system. Getting started is easy.

## Get AppScope

Directly download the CLI binary from [https://cdn.cribl.io/dl/scope/cli/linux/scope](https://s3-us-west-2.amazonaws.com/io.cribl.cdn/dl/scope/cli/linux/scope). Use this curl command:

```
curl -Lo scope https://cdn.cribl.io/dl/scope/cli/linux/scope && chmod 755 ./scope
```

## Explore the CLI

Run `scope --help` or `scope -h` to view CLI options. Also see the complete <link>[CLI Reference](#bookmark=id.q6rt37xg7u0g)</link>.

## Scope Some Commands

Test-run one or more well-known Linux commands, and view the results. E.g.:

```
scope curl https://google.com
scope ps -ef
```

## Explore the Data

To see the monitoring and visualization features AppScope offers, exercise some of its options. E.g.:

Show captured metrics:

`scope metrics`

Plot a chart of the `proc.cpu` metric:

`scope metrics -m proc.cpu -g`

Display captured events:

`scope events`

Filter out events, for just http:

`scope events -t http`

List this AppScope session's history:

`scope history`

## Next Steps

For guidance on taking AppScope to the next level, [j[oin](https://cribl.io/community/)](https://cribl.io/community/) our [[community on Slack](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100)](https://app.slack.com/client/TD0HGJPT5/CPYBPK65V/thread/C01BM8PU30V-1611001888.001100).
