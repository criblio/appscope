
DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

GO_VER:=$(shell go version | cut -d' ' -f3)

# Allow GOPATH to be set, but don't require it.
ifndef GOPATH
GOPATH:=$(shell go env | grep GOPATH | cut -d\" -f2)
endif

GO:=go

GO_VER_FLAGS=$(shell cd $(DIR) && $(GOPATH)/bin/govvv -flags)

.PHONY: cliall compressscope

cliall: deps scope compressscope

cliclean:
	rm -rf $(DIR)build/*

deps:
	@echo DIR=$(DIR)
	@echo GO_VER=$(GO_VER)
	@echo GOPATH=$(GOPATH)
	$(GO) get github.com/go-bindata/go-bindata/...
	$(GO) get github.com/ahmetb/govvv

rebuild: cliclean all

local: scope
	sudo cp $(DIR)build/scope /usr/local/bin/scope

run/bundle.go:
	rm -rf $(DIR)/build/*
	cp $(DIR)../bin/linux/ldscope $(DIR)build/
	cp $(DIR)../lib/linux/libscope.so $(DIR)build/
	cp $(DIR)../conf/*.yml $(DIR)build/
	cd $(DIR) && $(GOPATH)/bin/go-bindata -o run/bundle.go -pkg run build

scope: run/bundle.go
	# GOOS=linux CGO_ENABLED=0 $(GO) build -a -tags netgo -ldflags="-s -w" -o build/scope
	cd $(DIR) && GOOS=linux $(GO) build -ldflags="$(GO_VER_FLAGS)" -o $(DIR)build/scope

compressscope:
	upx -9 $(DIR)build/scope

install: 
	cp $(DIR)build/scope /usr/local/bin/scope

clitest:
	cd $(DIR) && $(GO) test -v -coverprofile coverage.out ./...

cover:
	cd $(DIR) && $(GO) tool cover -html=coverage.out -o coverage.html
