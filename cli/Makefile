GOVVV = $(shell command -v govvv 2>/dev/null)
ifndef GOVVV
    GO=go
else 
    GO=govvv
endif

.PHONY: all

all: scope

clean: clean-scope
	rm -rf build/*

clean-scope:
	rm -f ./scope

rebuild: clean all

rebuild-scope: clean-scope all

local:
	$(GO) build -o /usr/local/bin/scope

run/cscope.go:
	cp ../bin/linux/scope build/cscope
	go-bindata -o run/cscope.go -pkg run build/cscope

scope: run/cscope.go
	# GOOS=linux CGO_ENABLED=0 $(GO) build -a -tags netgo -ldflags="-s -w" -o build/scope
	GOOS=linux $(GO) build -o build/scope
	# upx -9 build/scope

install: 
	cp build/scope /usr/local/bin/scope

test:
	go test -v -coverprofile coverage.out ./...

cover:
	go tool cover -html=coverage.out