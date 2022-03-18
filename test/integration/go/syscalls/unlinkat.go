package main

import (
	"log"
	"os"
)

func main() {
	_, e := os.Create("/tmp/test_file")
	if e != nil {
		log.Fatal(e)
	}

	// Call syscall.unlinkat
	e = os.Remove("/tmp/test_file")
	if e != nil {
		log.Fatal(e)
	}
}
