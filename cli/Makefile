ARCH := $(shell uname -m)

GO ?= $(shell which go 2>&1)
ifeq (,$(GO))
$(error "error: \`go\` not in PATH; install or set GO to it's path")
endif
GO_VER := $(shell $(GO) version | cut -d' ' -f3)
GOPATH ?= $(shell $(GO) env | grep GOPATH | cut -d\" -f2)
GO_BINDIR := $(GOPATH)/bin

BINDIR := ../bin/linux/$(ARCH)
LIBDIR := ../lib/linux/$(ARCH)
CONFDIR := ../conf

SCOPE := $(BINDIR)/scope
LDSCOPE := $(BINDIR)/ldscope
LIBSCOPE := $(LIBDIR)/libscope.so
CONF_YML := $(wildcard $(CONFDIR)/*.yml)

GO_BINDATA := $(GO_BINDIR)/go-bindata
GOVVV := $(GO_BINDIR)/govvv
GOLANGCI_LINT := $(GO_BINDIR)/golangci-lint

## build: build all (default)
build: all

## debug: build with gdb support
debug:
	$(MAKE) scope BUILD_OPTS="-gcflags=all=\"-N -l\""

## all: build and compress \`scope\`
all: scope compressscope

## clean: remove the binary from appscope/build and your GOPATH/bin directory
clean:
	$(RM) -r build/* 
	$(RM) run/bundle.go $(SCOPE)
	$(RM) coverage.out coverage.html

## deps: install dependencies
deps: $(GO_BINDATA) $(GOVVV) $(GOLANGCI_LINT)
$(GO_BINDATA):
	$(GO) env
	@echo GO_VER=$(GO_VER)
	@echo GOPATH=$(GOPATH)
	$(GO) get github.com/go-bindata/go-bindata/...
$(GOVVV):
	$(GO) get github.com/ahmetb/govvv
$(GOLANGCI_LINT):
	curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | \
		sh -s -- -b $(GO_BINDIR) v1.40.1

rebuild: clean build 

run/bundle.go: $(GO_BINDATA) $(LDSCOPE) $(LIBSCOPE) $(CONF_YML)
	$(RM) -r build/*
	cp $(LDSCOPE) $(LIBSCOPE) $(CONF_YML) build/
	$(GO_BINDATA) -o $@ -pkg run build

# in case someone is building here before the C code is built
$(LDSCOPE):
	@$(MAKE) -C .. bin/linux/$(ARCH)/ldscope
$(LIBSCOPE):
	@$(MAKE) -C .. lib/linux/$(ARCH)/libscope.so

# shortcut "easy button"
scope: $(SCOPE)

$(SCOPE): export GOOS=linux
$(SCOPE): export CGO_ENABLED=0
$(SCOPE): run/bundle.go $(GOVVV)
	$(GO) build \
		-ldflags="$(shell $(GOVVV) -flags)" \
		$(BUILD_OPTS) \
	       	-o $(SCOPE)

compressscope: $(SCOPE)
	upx -l $(SCOPE) >/dev/null 2>&1 || upx -9 $(SCOPE)

## test: run unit tests
test:
	$(GO) test -v -p 1 -failfast -timeout 600s -coverprofile coverage.out ./...

## fmt: format all Go files
fmt:
	$(GO) fmt ./...

## lint: lint all Go files
lint: $(GOLANGCI_LINT)
	$(GOLANGCI_LINT) --new-from-rev=HEAD~1 run ./...

## vet: vet all Go files
vet: 
	$(GO) vet ./...

## cover: produce a test coverage report
cover: 
	$(GO) tool cover -html=coverage.out -o coverage.html

## check: format, vet, lint and test all Go files
check: fmt vet lint test

help: Makefile
	@echo
	@echo "Choose a command to run:"
	@echo
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'
	@echo

.PHONY: build debug all clean deps rebuild scope compressscope test fmt lint vet cover check help
