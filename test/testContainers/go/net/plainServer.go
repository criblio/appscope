// CGO_ENABLED=0 go/bin/go build serverPlain.go
// ./serverPlain
//
// curl http://localhost/hello

package main

import (
    "fmt"
    "log"
    "net/http"
    "os"
)

func HelloServer(w http.ResponseWriter, req *http.Request) {
    w.Header().Set("Content-Type", "text/plain")
    w.Write([]byte("This is an example server.\n"))
}

func main() {
    if len(os.Args) < 2 {
        log.Fatal("Missing required arg of port")
    }
    port := fmt.Sprint(":", os.Args[1])

    http.HandleFunc("/hello", HelloServer)
    err := http.ListenAndServe(port, nil)
    if err != nil {
        log.Fatal("ListenAndServe: ", err)
    }
}
