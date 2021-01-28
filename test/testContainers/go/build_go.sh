#!/bin/sh
cd /go/thread && CGO_ENABLED=0 go build fileThread.go
cd /go/net && go build -o plainServerDynamic plainServer.go
cd /go/net && CGO_ENABLED=0 go build -o plainServerStatic plainServer.go
cd /go/net && go build -o tlsServerDynamic tlsServer.go
cd /go/net && CGO_ENABLED=0 go build -o tlsServerStatic tlsServer.go
cd /go/net && go build -o plainClientDynamic plainClient.go
cd /go/net && CGO_ENABLED=0 go build -o plainClientStatic plainClient.go
cd /go/net && go build -o tlsClientDynamic tlsClient.go
cd /go/net && CGO_ENABLED=0 go build -o tlsClientStatic tlsClient.go
cd /go/cgo && make cleanDynamic cleanStatic && make all