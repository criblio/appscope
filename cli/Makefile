
DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

GO_VER:=$(shell go version | cut -d' ' -f3)

# Allow GOPATH to be set, but don't require it.
ifndef GOPATH
GOPATH:=$(shell go env | grep GOPATH | cut -d\" -f2)
endif

GO:=go

GO_VER_FLAGS=$(shell cd $(DIR) && $(GOPATH)/bin/govvv -flags)

.PHONY: build all install uninstall clean deps rebuild scope compressscope test fmt lint vet cover check help

## build (default): install dependencies and install the scope binary in your GOPATH/bin directory
build: deps
	cd $(DIR) && GOOS=linux CGO_ENABLED=0 $(GO) build -ldflags="$(GO_VER_FLAGS)" -o $(GOPATH)/bin/scope

## debug: build, but with gdb support
debug: deps
	cd $(DIR) && GOOS=linux CGO_ENABLED=0 $(GO) build -ldflags="$(GO_VER_FLAGS)" -gcflags=all="-N -l" -o $(GOPATH)/bin/scope

## cliall (popular): install dependencies and install the scope binary in appscope/build with compression
cliall: deps scope compressscope

## cliclean: remove the binary from appscope/build and your GOPATH/bin directory
cliclean:
	rm -rf $(DIR)build/*
	rm $(GOPATH)/bin/scope

## deps: install all project dependencies
deps:
	@echo DIR=$(DIR)
	@echo GO_VER=$(GO_VER)
	@echo GOPATH=$(GOPATH)
	$(GO) get github.com/go-bindata/go-bindata/...
	$(GO) get github.com/ahmetb/govvv
	curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(GOPATH)/bin v1.40.1

rebuild: cliclean build 

run/bundle.go:
	rm -rf $(DIR)/build/*
	cp $(DIR)../bin/linux/ldscope $(DIR)build/
	cp $(DIR)../lib/linux/libscope.so $(DIR)build/
	cp $(DIR)../conf/*.yml $(DIR)build/
	cd $(DIR) && $(GOPATH)/bin/go-bindata -o run/bundle.go -pkg run build

scope: run/bundle.go
	cd $(DIR) && GOOS=linux CGO_ENABLED=0 $(GO) build -ldflags="$(GO_VER_FLAGS)" -o $(DIR)../bin/linux/scope

compressscope:
	upx -9 $(DIR)../bin/linux/scope

## clitest: test all Go files
clitest:
	cd $(DIR) && $(GO) test -v -p 1 -failfast -timeout 600s -coverprofile coverage.out ./...

## fmt: format all Go files
fmt:
	$(GO) fmt ./...

## lint: lint all Go files
lint: 
	golangci-lint run ./...

## vet: vet all Go files
vet: 
	$(GO) vet ./...

## cover: produce a test coverage report
cover: 
	cd $(DIR) && $(GO) tool cover -html=coverage.out -o coverage.html

## check: format, vet, lint and test all Go files
check: fmt vet lint test

help: Makefile
	@echo
	@echo "Choose a command to run:"
	@echo
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'
	@echo
