
CWD=$(shell pwd)
PLATFORM=$(shell "$(CWD)"/scope_env.sh platform)
VERSION=$(shell git describe --abbrev=0 --tags | tr -d v)
$(shell echo -n $(VERSION) > $(CWD)/cli/VERSION)

ifeq ($(PLATFORM),Linux)
	include os/linux/Makefile
	include cli/Makefile
else
	ifeq ($(PLATFORM),macOS)
		include os/macOS/Makefile
	else
$(info ERROR not a valid platform: "$(PLATFORM)")
endif
endif

ifeq ($(PLATFORM),Linux)

.PHONY: clean test all

clean:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile $@

test:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile $@

all:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile $@
endif

.PHONY: docker-build
docker-build: TAG ?= "appscope-builder"
docker-build: DOCKER ?= $(shell which docker 2>/dev/null)
docker-build: BUILD_ARGS ?=
docker-build:
	@[ -x "$(DOCKER)" ] || \
		( echo >&2 "error: Please install Docker first."; exit 1)
	@$(DOCKER) build \
		--tag $(TAG) \
		-f docker/builder/Dockerfile \
		$(BUILD_ARGS) .
	$(DOCKER) run -it --rm \
		-v "$(shell pwd):/root/appscope" \
		--entrypoint /bin/bash \
		$(TAG) \
		-c "make all test"

# Annoyingly not DRY
.PHONY: docker-run
docker-run: TAG?="appscope-builder"
docker-run: DOCKER?=$(shell which docker 2>/dev/null)
docker-run: PWD:=$(shell pwd)
docker-run: BUILD_ARGS ?=
docker-run:
	@[ -x "$(DOCKER)" ] || \
		( echo >&2 "error: Please install Docker first."; exit 1)
	@$(DOCKER) build \
		--tag $(TAG) \
		-f docker/builder/Dockerfile \
		$(BUILD_ARGS) .
	@$(DOCKER) run -it --rm \
		-v "$(shell pwd):/root/appscope" \
		-e SCOPE_LOG_DEST=file:///root/appscope/scope.log \
		$(TAG)

.PHONY: docker-run-alpine
docker-run-alpine: TAG?="appscope-builder-alpine"
docker-run-alpine: DOCKER?=$(shell which docker 2>/dev/null)
docker-run-alpine: PWD:=$(shell pwd)
docker-run-alpine: BUILD_ARGS ?=
docker-run-alpine:
	@[ -x "$(DOCKER)" ] || \
		( echo >&2 "error: Please install Docker first."; exit 1)
	@$(DOCKER) build \
		--tag $(TAG) \
		-f docker/builder/Dockerfile.alpine \
		$(BUILD_ARGS) .
	@$(DOCKER) run -it --rm \
		-v "$(shell pwd):/root/appscope" \
		-e SCOPE_LOG_DEST=file:///root/appscope/scope.log \
		$(TAG)

# Using the docker-* targets above, many files here are created as root in the
# container and end up inaccessible here. Added this here after getting bit
# repeatedly.
.PHONY: chown
chown:
	sudo chown -R $(USER):$(USER) .

