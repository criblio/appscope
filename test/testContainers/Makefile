#
# AppScope - Integration Test Runner
#
# See ./README.md for details.
#

COMPOSE ?= $(shell which docker-compose 2>/dev/null)
ifeq (, $(COMPOSE))
    $(error Missing docker-compose)
endif

DOCKER ?= $(shell which docker 2>/dev/null)
ifeq (, $(DOCKER))
    $(error Missing docker)
endif

TESTS=$(shell $(COMPOSE) config --services | sort -V)

help:
	@echo "AppScope Integration Test Runner"
	@echo "  \`make help\` - show this info"
	@echo "  \`make all\` - run all tests, stop of first failure"
	@echo "  \`make build\` - build all test images"
	@echo "  \`make push\` - push all test images"
	@echo "  \`make (test)\` - run a single test"
	@echo "  \`make (test)-build\` - build the test's image"
	@echo "  \`make (test)-shell\` - run a shell in the test's container"
	@echo "Tests: $(TESTS)"

all:
	@STARTED=; for TEST in $(TESTS); do \
		if [ -n "$(START)" -a -z "$$STARTED" ]; then \
			if [ "$(START)" = "$$TEST" ]; then \
				STARTED=1; \
			else \
				continue; \
			fi; \
		fi; \
		if ! $(MAKE) -s $${TEST}; then  \
		if [ -z "$(CONTINUE)" ]; then \
				echo >&2 "error: aborting \`make all\` on test failure"; \
				exit 1; \
			fi; \
		fi; \
	done

$(TESTS):
	@$(MAKE) -s $(@)-build
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " Running \"$@\" Test"
	@echo "==================================================================="
	@time $(COMPOSE) run $@ || { echo >&2 "error: $(@) test failed."; exit 1; }

build: $(addsuffix -build, $(TESTS))

# --
# Check if an image already exists locally with the tag before pulling it again
# so we can build a newer version with newer configs and not have to rebuilt it
# every time. That's what ID is for. Should have no effect in CI as the local
# images list should be empty.
# --
%-build: IMAGE=scopeci/scope-$(@:%-build=%)-it
%-build: ID=$(shell $(DOCKER) images -q --filter=reference=$(IMAGE))
%-build:
	@[ -n "$(filter $(@:%-build=%),$(TESTS))" ] || \
		{ echo >&2 "error: $(@:%-build=%) isn't a test."; exit 1; }
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " Building Image for \"$(@:%-build=%)\" Test"
	@echo "==================================================================="
	-[ -n "$(ID)" ] || $(COMPOSE) pull $(@:%-build=%)
	@$(COMPOSE) build --progress tty $(@:%-build=%)

push: $(addsuffix -push, $(TESTS))

%-push:
	@[ -n "$(filter $(@:%-push=%),$(TESTS))" ] || \
		{ echo >&2 "error: $(@:%-push=%) isn't a test."; exit 1; }
	@$(MAKE) -s $(@:%-push=%)-build
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " Pushing Image for \"$(@:%-push=%)\" Test"
	@echo "==================================================================="
	@time $(COMPOSE) push $(@:%-push=%)
	@[ -z "$(SCOPECI_TOKEN)"] || $(DOCKER) run --rm \
		-v "$(shell pwd)/OVERVIEW.md:/data/README.md:ro" \
		-e "DOCKERHUB_USERNAME=scopeci" \
		-e "DOCKERHUB_PASSWORD=$(SCOPECI_TOKEN)" \
		-e "DOCKERHUB_REPO_PREFIX=scopeci" \
		-e "DOCKERHUB_REPO_NAME=scope-$(@:%-push=%)-it" \
		-e "SHORT_DESCRIPTION=AppScope\ Integration\ Test" \
		sheogorath/readme-to-dockerhub;

%-shell: CMD?=/bin/bash
%-shell:
	@[ -n "$(filter $(@:%-shell=%),$(TESTS))" ] || \
		{ echo >&2 "error: $(@:%-shell=%) isn't a test."; exit 1; }
	@$(MAKE) -s $(@:%-shell=%)-build
	@echo
	@echo "==================================================================="
	@echo " $(shell date)"
	@echo " Shell in Container for \"$(@:%-shell=%)\" Test"
	@echo "==================================================================="
	@$(COMPOSE) -f docker-compose.yml -f docker-compose.privileged.yml run $(@:%-shell=%) $(CMD)

.PHONY: help all build push $(TESTS) %-build %-push %-shell

# vim: sw=4 ts=4 noet :
