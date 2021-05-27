#
# Cribl AppScope - Automake Input
#
# See docs/BUILD.md for details.
#

bin_PROGRAMS = scope$(EXEEXT)

scope$(EXEEXT): main.go cmd/*.go run/bundle.go
	$(GO) build -o $@

run/bundle.go: $(GOPATH)/bin/go-bindata
	$(RM) -r ./build/*
	cp ../src/ldscope ./build/
	cp ../src/.libs/libscope.so ./build/
	cp ../conf/*.yml ./build/
	$(GOPATH)/bin/go-bindata -o run/bundle.go -pkg run build

$(GOPATH)/bin/go-bindata:
	$(GO) get github.com/go-bindata/go-bindata/...

check-local:
	$(GO) fmt
	$(GO) vet
	$(GO) test -v -coverprofile coverage.out ./...
	$(GO) tool cover -html=coverage.out -o coverage.html

install-exec-local:
	$(INSTALL) scope$(EXEEXT) $(gobindir)/scope$(EXEEXT)

uninstall-local:
	$(RM) $(gobindir)/scope$(EXEEXT)

.PHONY: clean-local
clean-local:
	$(RM) scope$(EXEEXT)

.PHONY: maintainer-clean-local
maintainer-clean-local:
	$(RM) -r Makefile.in .deps

