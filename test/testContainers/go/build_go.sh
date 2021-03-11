#!/bin/bash
set -e
source ~/.gvm/scripts/gvm
gvm install go$1 -B
gvm use $1 --default

cd /go/thread && CGO_ENABLED=0 go build fileThread.go
cd /go/net
go build -o plainServerDynamic plainServer.go
CGO_ENABLED=0 go build -o plainServerStatic plainServer.go
go build -o tlsServerDynamic tlsServer.go
CGO_ENABLED=0 go build -o tlsServerStatic tlsServer.go
go build -o plainClientDynamic plainClient.go
CGO_ENABLED=0 go build -o plainClientStatic plainClient.go
go build -o tlsClientDynamic tlsClient.go
CGO_ENABLED=0 go build -o tlsClientStatic tlsClient.go
cd /go/cgo
make cleanDynamic cleanStatic && make all
