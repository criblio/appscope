package main

import (
	"bufio"
	"fmt"
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

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, bufio.MaxScanTokenSize)
	scanner.Buffer(buf, maxScanTokenSize)

	for i := 0; scanner.Scan() && i < 5; i++ {
		fmt.Println(scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}

	// force close the connection to produce a net.close event
	// without this, we would have to wait for a timeout;
	// below is not available in go 1.8
	//http.DefaultClient.CloseIdleConnections()
	// alternative is to sleep but that will slow down integration testing
	// instead, do not expect a net.close event
}
