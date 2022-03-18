package main

import (
	"io/ioutil"
	"log"
)

func main() {
	// Call syscall.Readdir
	// note: ioutil.ReadDir was superceded by os.ReadDir
	// but the latter will fail in some versions < go1.16
	_, err := ioutil.ReadDir(".")
	if err != nil {
		log.Fatal(err)
	}
}
