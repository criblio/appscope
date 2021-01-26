GOVVV = $(shell command -v govvv 2>/dev/null)
ifndef GOVVV
    GO=go
else 
    GO=govvv
endif

DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

GO_VER:=$(shell go version | cut -d' ' -f3)
GOPATH:=$(shell go env | grep GOPATH | cut -d\" -f2)
GO:=go

GO_VER_FLAGS=$(shell $(GOPATH)/bin/govvv -flags)

.PHONY: cliall

cliall: deps scope

cliclean:
	rm -rf $(DIR)build/*

deps:
	@echo DIR=$(DIR)
	@echo GO_VER=$(GO_VER)
	@echo GOPATH=$(GOPATH)
	$(GO) get github.com/go-bindata/go-bindata/...
	$(GO) get github.com/ahmetb/govvv

rebuild: cliclean all

local:
	cd $(DIR) && $(GO) build -o /usr/local/bin/scope

run/scopec.go:
	cp $(DIR)../bin/linux/scopec $(DIR)build/
	cd $(DIR) && go-bindata -o run/scopec.go -pkg run build/scopec

scope: run/scopec.go
	# GOOS=linux CGO_ENABLED=0 $(GO) build -a -tags netgo -ldflags="-s -w" -o build/scope
	cd $(DIR) && GOOS=linux $(GO) build -o $(DIR)build/scope 
	upx -9 $(DIR)build/scope

install: 
	cp $(DIR)build/scope /usr/local/bin/scope

test:
	cd $(DIR) && go test -v -coverprofile coverage.out ./...

cover:
	cd $(DIR) && go tool cover -html=coverage.out -o coverage.html