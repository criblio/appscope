# Cribl AppScope - Alpine Integration Test

This is an integration test to wring out AppScope with musl libc on an Alpine
container. Supporting musl required adjustments in the way we handle console
IO, DNS lookups, and the Go runtime so we will be focusing there for now.

I'm starting with the Go tests in `../go/` and adjusting...

  - Changed the FROM image to `alpine:latest` and `apk add`'ed packages.
  - Removed the InfluxDB tests because they're slow and the client we use
    is built on glibc and won't work in Alpine. Copied `test_go.sh` from
    `../go/` and cut out the InfluxDB bits.
  - `pkill` isn't working and there is a leftover server listening on 80 which
    prevents Apache from starting later so I've just changed the ports used
    in the Go tests from 80/81 to 8080/8081.
  - Had to rebuild the Rust test program in the build stage.
  - Added checks when running the versions of curl to ensure we're getting DNS
    events.

Run the test with `docker-compose run alpine` and get a shell with
`docker-compose run alpine /bin/bash`.
