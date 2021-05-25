#
# Cribl AppScope™ - Automake Input
#
# See docs/BUILD.md for details.
#

bin_PROGRAMS = scope

scope$(EXEEXT): main.go cmd/*.go run/bundle.go
	$(GO) build

run/bundle.go: $(GOPATH)/bin/go-bindata
	$(RM) -r ./build/*
	cp ../src/ldscope ./build/
	cp ../src/.libs/libscope.so ./build/
	cp ../conf/*.yml ./build/
	$(GOPATH)/bin/go-bindata -o run/bundle.go -pkg run build

$(GOPATH)/bin/go-bindata:
	$(GO) get github.com/go-bindata/go-bindata/...

check-local:
	$(GO) test -v -coverprofile coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html

.PHONY: clean-local
clean-local:
	$(RM) scope

.PHONY: maintainer-clean-local
maintainer-clean-local:
	$(RM) -r Makefile.in .deps

