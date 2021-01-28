
DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

GO_VER:=$(shell go version | cut -d' ' -f3)

# Allow GOPATH to be set, but don't require it.
ifndef GOPATH
GOPATH:=$(shell go env | grep GOPATH | cut -d\" -f2)
endif

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

demo:
	go build -o $(DIR)demo/demo-scripts/goHttps demo/demo-scripts/https.go
	docker build -t criblio/appscope-demo -f $(DIR)demo/Dockerfile .

rebuild: cliclean all

local:
	cd $(DIR) && $(GO) build -o /usr/local/bin/scope

run/scopec.go:
	cp $(DIR)../bin/linux/scopec $(DIR)build/
	cd $(DIR) && $(GOPATH)/bin/go-bindata -o run/scopec.go -pkg run build/scopec

scope: run/scopec.go
	# GOOS=linux CGO_ENABLED=0 $(GO) build -a -tags netgo -ldflags="-s -w" -o build/scope
	cd $(DIR) && GOOS=linux $(GO) build -ldflags="$(GO_VER_FLAGS)" -o $(DIR)build/scope
	upx -9 $(DIR)build/scope

install: 
	cp $(DIR)build/scope /usr/local/bin/scope

clitest:
	cd $(DIR) && $(GO) test -v -coverprofile coverage.out ./...

cover:
	cd $(DIR) && $(GO) tool cover -html=coverage.out -o coverage.html
