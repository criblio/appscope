// CGO_ENABLED=0 go/bin/go build serverPlain.go
// ./serverPlain
//
// curl http://localhost/hello

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
    err := http.ListenAndServe(":80", nil)
    if err != nil {
        log.Fatal("ListenAndServe: ", err)
    }
}

