package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"time"
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func main() {
	/*    conn, errc := net.Dial("tcp", "golang.org:80")
	    //conn, errc := net.Dial("tcp", "216.58.193.81:80")
	    if errc != nil {
		   // handle error
	    }

	    fmt.Printf("using %s\n", conn.LocalAddr());
	    fmt.Fprintf(conn, "GET / HTTP/1.0\r\n\r\n")
	    status, errc := bufio.NewReader(conn).ReadString('\n')
	    fmt.Printf("http status: %s", status)
	*/
	fmt.Printf("Hello, World\n")

	d1 := []byte("hello\ngo\n")
	err := ioutil.WriteFile("/tmp/dat1", d1, 0644)
	check(err)
	fmt.Printf("wrote %s\n", d1)

	//time.Sleep(20 * time.Second)
	//return

	dat, err := ioutil.ReadFile("/tmp/dat1")
	check(err)
	if dat == nil {
		fmt.Print(string(dat))
	}

	//time.Sleep(20 * time.Second)
	//return

	f, err := os.Create("/tmp/dat2")
	check(err)
	fmt.Printf("file %d\n", f)

	d2 := []byte{115, 111, 109, 101, 10}
	n2, err := f.Write(d2)
	check(err)
	fmt.Printf("wrote %d bytes\n", n2)

	f.Sync()
	o2, err := f.Seek(0, 0)
	check(err)
	fmt.Printf("Seek %d\n", o2)

	b1 := make([]byte, 10)
	n1, err := f.Read(b1)
	check(err)
	fmt.Printf("read %d bytes: %s\n", n1, string(b1[:n1]))

	f.Close()

	//conn, err := net.Dial("tcp", "golang.org:80")
	conn, err := net.Dial("tcp", "216.58.193.81:80")
	if err != nil {
		// handle error
	}

	fmt.Printf("using %s\n", conn.LocalAddr())
	fmt.Fprintf(conn, "GET / HTTP/1.0\r\n\r\n")
	status, err := bufio.NewReader(conn).ReadString('\n')
	fmt.Printf("http status: %s", status)

	time.Sleep(20 * time.Second)
}
