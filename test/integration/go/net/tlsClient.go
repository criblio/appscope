package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
)

const maxScanTokenSize = 1024 * 1024

func main() {

	resp, err := http.Get("https://cribl.io")
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	fmt.Println("Response status:", resp.Status)

	reader := bufio.NewReaderSize(resp.Body, maxScanTokenSize)

	for {
		token, _, err := reader.ReadLine()
		if len(token) > 0 {
			fmt.Println(string(token))
		}
		if err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err)
			}
		}
	}

	// force close the connection to produce a net.close event
	// without this, we would have to wait for a timeout;
	// below is not available in go 1.8
	//http.DefaultClient.CloseIdleConnections()
	// alternative is to sleep but that will slow down integration testing
	// instead, do not expect a net.close event
}
