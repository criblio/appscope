
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
	$(MAKE) -f cli/Makefile cli$@
	$(MAKE) -C contrib $@
	@$(RM) -rf newdir testtempdir1 testtempdir2 coverage go .bash_history .cache .viminfo scope.log *.gcda

test:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile cli$@

all:
	$(MAKE) -f os/linux/Makefile core$@
	$(MAKE) -f cli/Makefile cli$@
endif

.PHONY: docker-build
docker-build: TAG ?= "appscope-builder"
docker-build: DOCKER ?= $(shell which docker 2>/dev/null)
docker-build: BUILD_ARGS ?=
docker-build: CMD ?= make all test
docker-build:
	@[ -x "$(DOCKER)" ] || \
		( echo >&2 "error: Please install Docker first."; exit 1)
	@$(DOCKER) build \
		--tag $(TAG) \
		-f docker/builder/Dockerfile \
		$(BUILD_ARGS) .
	$(DOCKER) run -it --rm \
		-v "$(shell pwd):/home/builder/appscope" \
		-u $(shell id -u):$(shell id -g) \
		--entrypoint /bin/bash \
		$(TAG) \
		-c "$(CMD)"

# Annoyingly not DRY
.PHONY: docker-run
docker-run: TAG?="appscope-builder"
docker-run: DOCKER?=$(shell which docker 2>/dev/null)
docker-run: BUILD_ARGS ?=
docker-run:
	@[ -x "$(DOCKER)" ] || \
		( echo >&2 "error: Please install Docker first."; exit 1)
	@$(DOCKER) build \
		--tag $(TAG) \
		-f docker/builder/Dockerfile \
		$(BUILD_ARGS) .
	@$(DOCKER) run -it --rm \
		-v "$(shell pwd):/home/builder/appscope" \
		-u $(shell id -u):$(shell id -g) \
		$(TAG) bash --login

.PHONY: docker-run-alpine
docker-run-alpine: TAG?="appscope-builder-alpine"
docker-run-alpine: DOCKER?=$(shell which docker 2>/dev/null)
docker-run-alpine: BUILD_ARGS ?=
docker-run-alpine:
	@[ -x "$(DOCKER)" ] || \
		( echo >&2 "error: Please install Docker first."; exit 1)
	@$(DOCKER) build \
		--tag $(TAG) \
		-f docker/builder/Dockerfile.alpine \
		$(BUILD_ARGS) .
	@$(DOCKER) run -it --rm \
		-v "$(shell pwd):/home/builder/appscope" \
		-u $(shell id -u):$(shell id -g) \
		$(TAG) bash --login
