GOVVV = $(shell command -v govvv 2>/dev/null)
ifndef GOVVV
    GO=go
else 
    GO=govvv
endif

DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.PHONY: all

all: scope

clean:
	rm -rf $(DIR)build/*

deps:
	go get github.com/go-bindata/go-bindata/...
	go get github.com/ahmetb/govvv

rebuild: clean all

local:
	cd $(DIR) && $(GO) build -o /usr/local/bin/scope

run/cscope.go:
	cp $(DIR)../bin/linux/cscope $(DIR)build/
	cd $(DIR) && go-bindata -o run/cscope.go -pkg run build/cscope

scope: run/cscope.go
	# GOOS=linux CGO_ENABLED=0 $(GO) build -a -tags netgo -ldflags="-s -w" -o build/scope
	cd $(DIR) && GOOS=linux $(GO) build -o $(DIR)build/scope 
	upx -9 build/scope

install: 
	cp $(DIR)build/scope /usr/local/bin/scope

test:
	cd $(DIR) && go test -v -coverprofile coverage.out ./...

cover:
	cd $(DIR) && go tool cover -html=coverage.out -o coverage.html