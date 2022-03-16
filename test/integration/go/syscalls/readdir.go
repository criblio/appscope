package main

import (
	"log"
	"os"
)

func main() {
	// Call syscall.Readdir
	_, err := os.ReadDir(".")
	if err != nil {
		log.Fatal(err)
	}
}
