# Cribl AppScope - Alpine Integration Test

This is an integration test to wring out AppScope with musl libc on an Alpine
container. Supporting musl required adjustments in the way we handle console
IO, DNS lookups, and the Go runtime so we will be focusing there for now.

I'm starting with the Go tests in `../go/` and adjust it so we can get a 
shell in the container.

Run the test with `docker-compose run alpine` and get a shell with
`docker-compose run alpine /bin/bash`.
