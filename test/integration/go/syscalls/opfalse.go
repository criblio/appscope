package main

import (
	"os"
)

func main() {
	os.Remove("")
	os.Remove("/tmp/test_file_not_exist")
	os.Open("/tmp/test_file_not_exist")
}
