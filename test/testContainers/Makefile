#
# Cribl AppScope™ - Integration Test Runner
#
# See ./README.md for details.
#

# ---
# Head's Up...
#
# There is a nasty issue with docker-compose 1.25.0 which is current at this
# time. `docker-compose pull` doesn't actually pull anything for a service that
# has a `build` config. The "feature" has since been reverted in their code but
# it's not out in general release yet. I'm not sure if/when it will appear in
# the images GitHub uses for workflows either. So, we're using `docker pull`
# and assuming we can guess the image names here instead of the normal
# `docker-compose pull`.
#
# Screw You Docker Compose!
#
# pd
# ---

COMPOSE ?= $(shell which docker-compose 2>/dev/null)
ifeq (, $(COMPOSE))
    $(error Missing docker-compose)
endif

DOCKER ?= $(shell which docker 2>/dev/null)
ifeq (, $(DOCKER))
    $(error Missing docker)
endif

TESTS=$(shell $(COMPOSE) config --services | sort -V)

MARKER=AppScopeTest

help:
	@echo "AppScope Integration Test Runner"
	@echo "  \`make help\` - show this info"
	@echo "  \`make all\` - run all tests"
	@echo "  \`make build\` - build all test images"
	@echo "  \`make push\` - push all test images"
	@echo "  \`make (test)\` - run a single test"
	@echo "  \`make (test)-build\` - build the test's image"
	@echo "  \`make (test)-shell\` - run a shell in the test's container"
	@echo "Tests: $(TESTS)"

all: CONTINUE := 1
all: $(TESTS)

$(TESTS):
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " $(MARKER): Building Image for \"$@\" Test"
	@echo "==================================================================="
	-@[ -n "$(NOPULL)" ] || $(DOCKER) pull criblci/scope-$(@)-it
	@$(COMPOSE) build $@ || if [ -z "$(CONTINUE)" ]; then \
		{ echo >&2 "error: $(@) build failed."; exit 1; } \
	fi
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " $(MARKER): Running \"$@\" Test"
	@echo "==================================================================="
	@$(COMPOSE) run $@ || if [ -z "$(CONTINUE)" ]; then \
		{ echo >&2 "error: $(@) test failed."; exit 1; } \
	fi

build: $(addsuffix -build, $(TESTS))

%-build:
	[ -n "$(filter $(@:%-build=%),$(TESTS))" ] || \
		{ echo >&2 "error: $(@:%-build=%) isn't a test."; exit 1; }
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " $(MARKER): Building Image for \"$@\" Test"
	@echo "==================================================================="
	-@[ -n "$(NOPULL)" ] || $(DOCKER) pull criblci/scope-$(@:%-build=%)-it
	@$(COMPOSE) build $(@:%-build=%)

push: $(TESTS_BUILD)
	$(COMPOSE) push

%-push:
	@[ -n "$(filter $(@:%-push=%),$(TESTS))" ] || \
		{ echo >&2 "error: $(@:%-push=%) isn't a test."; exit 1; }
	@[ -n "$(SCOPECI_TOKEN)" ] || \
		{ echo >&2 "error: SCOPECI_TOKEN not set."; exit 1; }
	@echo "$(SCOPECI_TOKEN)" | $(DOCKER) login -u scopeci --password-stdin
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " $(MARKER): Pushing Image for \"$@\" Test"
	@echo "==================================================================="
	@$(COMPOSE) push $(@:%-push=%)
	@$(DOCKER) logout
	-@[ -f "./README.$(@:%-push=%).md" ] && $(DOCKER) run --rm \
		-v "./README.$(@:%-push=%).md/:/data/README.md:ro" \
		-e "DOCKERHUB_USERNAME=scopeci" \
		-e "DOCKERHUB_PASSWORD=$(SCOPECI_TOKEN)" \
		-e "DOCKERHUB_REPO_PREFIX=criblci" \
		-e "DOCKERHUB_REPO_NAME=scope-$(@:%-push=%)-it" \
		-e "SHORT_DESCRIPTION=AppScope\ Integration\ Test" \
		sheogorath/readme-to-dockerhub;

%-shell: SHELL ?= /bin/bash
%-shell:
	@[ -n "$(filter $(@:%-shell=%),$(TESTS))" ] || \
		{ echo >&2 "error: $(@:%-shell=%) isn't a test."; exit 1; }
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " $(MARKER): Shell in Container for \"$@\" Test"
	@echo "==================================================================="
	@$(COMPOSE) -f docker-compose.yml -f docker-compose.privileged.yml \
		run $(@:%-shell=%) $(SHELL)

.PHONY: help all build push $(TESTS) %-shell %-build %-push

# vim: sw=4 ts=4 noet :
