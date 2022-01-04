package main

import (
    // #include "./myc.h"
    // #cgo LDFLAGS: -L. -lmyc
    "C"
    "fmt"
)

func main() {
    fmt.Println("Starting mygo.go")
    C.myCFunc()
    fmt.Println("Ending mygo.go")
}
