# Cribl AppScope - Alpine Integration Test

This is an integration test to wring out AppScope with musl libc on an Alpine
container. Supporting musl required adjustments in the way we handle console
IO, DNS lookups, and the Go runtime so we will be focusing there for now.

I'm starting with the Go tests in `../go/` and adjusting...

  - Changed the FROM image to `alpine:latest` and `apk add`'ed packages.
  - Removed the InfluxDB tests because they're slow and the client we use
    is built on glibc and won't work in Alpine. Copied `test_go.sh` from
    `../go/` and cut out the InfluxDB bits.

Run the test with `docker-compose run alpine` and get a shell with
`docker-compose run alpine /bin/bash`.
