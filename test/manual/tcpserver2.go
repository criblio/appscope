// 
// Debug TCP Server that Mimics LogStream's AppScope Source
//
// `go run tcpserver2.go 10091`
//

package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
)

func getBusyInABurgerKingBathroom(c net.Conn) {
	defer c.Close()
	fmt.Fprintf(os.Stderr, "%s: Connected\n", c.RemoteAddr().String())
	for {
		data, err := bufio.NewReader(c).ReadString('\n')
		if err != nil {
			if err == io.EOF {
				fmt.Fprintf(os.Stderr, "%s: Disconnected\n", c.RemoteAddr().String())
				return
			}
			fmt.Fprintf(os.Stderr, "error: closing on readln failure; %v\n", err)
			return
		}

		if (json.Valid([]byte(data))) {
			fmt.Fprintf(os.Stdout, "%s: %s", c.RemoteAddr().String(), data);
		} else {
			fmt.Fprintf(os.Stderr, "%s: closing on non-JSON data\n%s", c.RemoteAddr().String(), hex.Dump([]byte(data)))
			return
		}

		var header map[string]interface{}

		if err := json.Unmarshal([]byte(data), &header); err != nil {
			fmt.Fprintf(os.Stderr, "%s: invalid JSON; %v\n", c.RemoteAddr().String(), err)
			return
		}

		if "payload" == header["type"] {
			all := int(header["len"].(float64)) + 1
			bin := make([]byte,0)
			got := 0
			for got < all {
				tmp := make([]byte,all-got)
				n, err := io.ReadFull(c, tmp)
				if err != nil {
					fmt.Fprintf(os.Stderr, "error: closing on read(%d) failure; %v\n", all-got, err)
					return
				}
				fmt.Fprintf(os.Stderr, "info: read %d of %d\n", n, all-got)
				got += n
				bin = append(bin, tmp...)
			}
			fmt.Fprintf(os.Stderr, "%s: payload detected; %d bytes\n%s", c.RemoteAddr().String(), all, hex.Dump(bin))
		}
	}
}

func main() {
	arguments := os.Args
	if len(arguments) == 1 {
		fmt.Println("error: missing port argument")
		os.Exit(1)
	}

	l, err := net.Listen("tcp4", ":" + arguments[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: listen failed; %v\n", err)
		os.Exit(1)
	}
	defer l.Close()

	for {
		c, err := l.Accept()
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: accept failed; %v\n", err)
			os.Exit(1)
		}
		go getBusyInABurgerKingBathroom(c)
	}
}
