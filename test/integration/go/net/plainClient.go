package main

import (
	"bufio"
	"fmt"
	"net/http"
)

func main() {

	// ensure the website does not redirect to https;
	// also use the www. to avoid an unnecessary redirect
	resp, err := http.Get("http://www.gnu.org")
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	fmt.Println("Response status:", resp.Status)

	scanner := bufio.NewScanner(resp.Body)
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
