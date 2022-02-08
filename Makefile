#
# AppScope Build System
#
# See docs/BUILD.md for details.
#
# Use `make`, `make all`, `make test`, and `make clean` to use the local environment.
#
# Use `make build` to build (or `make run` to get a shell) in our builder container.
#
# Specify the processor architecture by setting `ARCH` to `x86_64` or `aarch64`.
# Set `DIST` to `ubuntu` or `alpine` for `make run`.
#

# detect local OS, warn/fail if unsupported
UNAME_S=$(shell uname -s | tr A-Z a-z)
ifeq ($(UNAME_S),linux)
	OS=linux
	OS_ID=$(shell egrep '^ID=' /etc/*-release | cut -d= -f2)
	OS_VER=$(shell egrep '^VERSION_ID=' /etc/*-release | cut -d= -f2 | sed 's/"//g')
else ifeq ($(UNAME_S),darwin)
	OS=macOS
else
$(error Building on $(UNAME_S) is unsupported!)
endif

# version number without the leading `v` from release tags
# this is set in CI so don't overwrite
VERSION ?= $(shell git describe --match "v*" --abbrev=0 --tags | tr -d v)

# cli expects us to write this file
$(shell echo -n $(VERSION) > cli/VERSION)

# github repository name; i.e. criblio/appscope
# set automatically in CI so don't overwrite that
GITHUB_REPOSITORY ?= $(shell git config --get remote.origin.url | cut -d: -f2 | sed -e 's/^\/\///' -e 's/\.git$$//')

# architectures we build for
# `uname -m` values; not Docker platform values like `amd64` or `arm64`
ARCH_LIST := x86_64 aarch64

# docker architectures for each $ARCH
PLATFORM_x86_64 := amd64
PLATFORM_aarch64 := arm64

FROM_PLATFORM_amd64 := x86_64
FROM_PLATFORM_arm64 := aarch64

# platforms we build for
_space := $(subst ,, )
_comma := ,
PLATFORMS := $(subst $(_space),$(_comma),$(foreach ARCH,$(ARCH_LIST),linux/$(PLATFORM_$(ARCH))))

# docker buildx builder name
# set in CI so don't overwrite it
BUILDER ?= appscope

# docker registries
REGISTRY_github := ghcr.io
REGISTRY_docker := docker.io

# docker image tag (without :version suffix)
# allow it to be overridden
DIST_IMAGE ?= $(GITHUB_REPOSITORY)
BUILD_IMAGE ?= $(REGISTRY_github)/$(GITHUB_REPOSITORY)-builder

# default target builds everything
all:
	@[ "ubuntu-18.04" = "$(OS_ID)-$(OS_VER)" ] || \
		echo >&2 "warning: building on $(OS_ID)-$(OS_VER) is unsupported; use \`make build\` instead"
	@$(MAKE) coreall
	@$(MAKE) -C cli all

# run unit tests
test:
	@$(MAKE) coretest
	@$(MAKE) -C cli test

# remove built content
clean:
	@$(MAKE) coreclean
	@$(MAKE) -C cli clean
	@$(MAKE) -C contrib clean
	@$(RM) bin/*/*/scope bin/*/*/ldscope bin/*/*/ldscopedyn lib/*/*/libscope.so
	@$(RM) -rf newdir testtempdir1 testtempdir2 coverage go .bash_history .cache .viminfo scope.log *.gcda

# target architecture and OS/ARCH-specific Makefile
ARCH ?= $(shell uname -m)

ifeq (,$(PLATFORM_$(ARCH))) 
ifeq (,$(FROM_PLATFORM_$(ARCH))) 
$(error error: invalid ARCH alias; "$(ARCH)")
else
$(info info: translating "$(ARCH)" ARCH to "$(FROM_PLATFORM_$(ARCH))")
override ARCH := $(FROM_PLATFORM_$(ARCH))
endif
endif

ifneq (,$(findstring $(ARCH),$(ARCH_LIST)))
include os/$(OS)/Makefile
else
$(error error: invalid ARCH; "$(ARCH)")
endif

# include the CLI targets with "cli" prefix plus "scope" shortcut
cli%:
	@$(MAKE) -C cli $(@:cli%=%)
scope:
	@$(MAKE) -C cli $@

# build in our builder container (legacy target)
docker-build:
	@echo >&2 'warning: the "docker-build" target is deprecated; use "build" instead'
	@$(MAKE) -s build

# get a shell in our builder container (legacy target)
docker-run: CMD ?= /bin/bash --login
docker-run:
	@echo >&2 'warning: the "docker-run" target is deprecated; use "run" instead'
	@$(MAKE) -s build CMD="$(CMD)"

# get a shell in our Alpine builder container (legacy target)
docker-run-alpine: CMD ?= /bin/bash --login
docker-run-alpine:
	@echo >&2 'warning: the "docker-run-alpine" target is deprecated; use "run DIST=alpine" instead'
	@$(MAKE) -s build CMD="$(CMD)" DIST="alpine"

# build in our builder container for the given ARCH
build: DIST ?= ubuntu
build: CMD ?= make all test
build: require-docker require-qemu-binfmt
	@[ -n "$(NOBUILD)" ] || $(MAKE) -s builder DIST="$(DIST)" ARCH="$(ARCH)"
	@docker run --rm $(if $(CI),,-it) \
		-v $(shell pwd):/home/builder/appscope \
		-u $(shell id -u):$(shell id -g) \
		--privileged \
		--platform $(ARCH) \
		$(if $(CI),,--name appscope-builder-$(DIST)-$(ARCH)) \
		$(BUILD_IMAGE):$(DIST)-$(ARCH) \
	       	$(CMD)

# run a shell our builder container without starting a build
run: DIST ?= ubuntu
run: CMD ?= /bin/bash --login
run:
	@echo Running the AppScope $(DIST)/$(ARCH) Builder Container
	@$(MAKE) -s build DIST="$(DIST)" ARCH="$(ARCH)" CMD="$(CMD)"

# get another shell in an existing builder container
exec: DIST ?= ubuntu
exec: CMD ?= /bin/bash
exec:
	@[ -n "$(shell docker ps -q -f "name=appscope-builder-$(DIST)-$(ARCH)")" ] || \
		{ echo >&2 "error: appscope-builder-$(DIST)-$(ARCH) not running"; exit 1; }
	@echo "Exec'ing into the AppScope $(DIST)/$(ARCH) Builder Container"
	@docker exec -it $(shell docker ps -q -f "name=appscope-builder-$(DIST)-$(ARCH)") $(CMD)

# build the builder image for the given ARCH
builder: DIST ?= ubuntu
builder: TAG := $(BUILD_IMAGE):$(DIST)-$(ARCH)
builder: require-docker-buildx-builder
	@[ -z "$(CI)" ] || echo "Update $(DIST)/$(ARCH) Builder Image"
	@docker buildx build \
		--builder $(BUILDER) \
		--tag $(TAG) \
		$(if $(CACHE_FROM),--cache-from $(CACHE_FROM)) \
		$(if $(CACHE_TO),--cache-to $(CACHE_TO)) \
		--platform linux/$(PLATFORM_$(ARCH)) \
		--label "org.opencontainers.image.description=AppScope $(ARCH) Builder ($(PLATFORM_$(ARCH)))" \
		$(if $(NOLOAD),,--load) \
		--file docker/builder/Dockerfile.$(DIST) \
		.

image: TAG := cribl/scope:dev-$(ARCH)
image: require-qemu-binfmt
	@docker buildx build \
		--tag $(TAG) \
		--platform linux/$(PLATFORM_$(ARCH)) \
		--file docker/base/Dockerfile \
		--load \
		.

docs-generate: TAG := cribl/scope:docs-$(ARCH)
docs-generate: require-docker-buildx-builder
	@echo Building the AppScope docs generator
	@docker buildx build \
		--tag $(TAG) \
		--platform linux/$(PLATFORM_$(ARCH)) \
		--file docker/docs/Dockerfile \
		.
	@echo Running the AppScope docs generator
	@docker run \
		-v $(shell pwd)/docs:/md \
		--user node:node \
		--rm cribl/scope:docs-$(ARCH) 
	@echo AppScope docs generator finished: md files are available in docs/md_files

k8s-test: require-kind require-kubectl image
	docker tag cribl/scope:dev-x86_64 cribl/scope:$(VERSION)
	kind delete cluster
	kind create cluster
	kind load docker-image cribl/scope:$(VERSION)
	kubectl create namespace test
	kubectl create namespace scope
	docker run -it cribl/scope:$(VERSION) scope k8s -m /tmp/metrics.log -e /tmp/events.log --namespace scope --debug | kubectl apply -f -
	kubectl label namespace test scope=enabled
	kubectl wait --for=condition=available deployment/scope -n scope
	kubectl run ubuntu --image=ubuntu:20.04 -n test --restart=Never --command -- sleep infinity

# setup the buildx builder if it's not running already
require-docker-buildx-builder: require-docker-buildx require-qemu-binfmt
	@if ! docker buildx inspect $(BUILDER) >/dev/null 2>&1; then \
		docker buildx create --name $(BUILDER) --driver docker-container; \
		docker buildx inspect --bootstrap --builder $(BUILDER); \
	fi

# fail of docker not in $PATH
require-docker:
	@[ -n "$(shell which docker)" ] || \
		{ echo >&2 "error: docker required"; exit 1; }

# fail of Docker Buildx isn't installed
# see https://docs.docker.com/buildx/working-with-buildx/
require-docker-buildx: require-docker
	@docker buildx version >/dev/null || \
		{ echo >&1 "error: docker buildx required"; exit 1; }

# enable execution of different architecture containers with QEMU
# see https://github.com/multiarch/qemu-user-static
require-qemu-binfmt: require-docker
	@[ -n "$(wildcard /proc/sys/fs/binfmt_misc/qemu-*)" ] || \
		docker run --rm --privileged tonistiigi/binfmt:latest --install all

# fail of kind not in $PATH
require-kind:
	@[ -n "$(shell which kind)" ] || \
		{ echo >&2 "error: kind required"; exit 1; }

# fail of kubectl not in $PATH
require-kubectl:
	@[ -n "$(shell which kubectl)" ] || \
		{ echo >&2 "error: kubectl required"; exit 1; }

.PHONY: all test clean
.PHONY: cli% scope
.PHONY: docker-build docker-run 
.PHONY: build-arch builder-arch
.PHONY: docs-generate
.PHONY: image
.PHONY: require-docker-buildx-builder require-docker require-docker-buildx require-qemu-binfmt require-kind require-kubectl
