// https://github.com/denji/golang-tls
//
// openssl genrsa -out server.key 2048
// openssl ecparam -genkey -name secp384r1 -out server.key
// openssl req -new -x509 -sha256 -key server.key -out server.crt -days 3650 -subj "/C=US/ST=California/L=San Francisco/O=Cribl/OU=Cribl/CN=localhost"
//
// CGO_ENABLED=0 go/bin/go build tlsServer.go
// ./tlsServer
//
// curl -k --key server.key --cert server.crt https://localhost:4430/hello

package main

import (
    "net/http"
    "log"
)

func HelloServer(w http.ResponseWriter, req *http.Request) {
    w.Header().Set("Content-Type", "text/plain")
    w.Write([]byte("This is an example server.\n"))
}

func main() {
    http.HandleFunc("/hello", HelloServer)
    err := http.ListenAndServeTLS(":4430", "server.crt", "server.key", nil)
    if err != nil {
        log.Fatal("ListenAndServe: ", err)
    }
}

