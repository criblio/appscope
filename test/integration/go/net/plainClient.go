package main

import (
	"bufio"
	"fmt"
	"net/http"
)

func main() {

	// ensure the website does not redirect to https
	resp, err := http.Get("http://gnu.org")
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
}
