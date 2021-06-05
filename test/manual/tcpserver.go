//
// Debug TCP Server that Mimics LogStream's AppScope Source
//
// `go run tcpserver.go 10091`
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
	"time"
)

var start time.Time

func log(c net.Conn, format string, args ...interface{}) {
	fmt.Printf(fmt.Sprintf("%8s %s ", time.Since(start).Truncate(time.Millisecond).String(), c.RemoteAddr().String())+format, args...)
}

func getBusyInABurgerKingBathroom(c net.Conn) {
	defer c.Close()
	log(c, "Connected\n")
	r := bufio.NewReader(c)
	for {
		data, err := r.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				log(c, "Disconnected\n")
				return
			}
			fmt.Fprintf(os.Stderr, "error: closing on readln failure; %v\n", err)
			return
		}

		if json.Valid([]byte(data)) {
			log(c, data)
		} else {
			log(c, "closing on non-JSON data\n%s", hex.Dump([]byte(data)))
			return
		}

		var header map[string]interface{}

		if err := json.Unmarshal([]byte(data), &header); err != nil {
			log(c, "closing on invalid JSON; %v\n%s\n", err, hex.Dump([]byte(data)))
			return
		}

		if "payload" == header["type"] {
			need := int(header["len"].(float64))
			bin := make([]byte, 0)
			got := 0
			for got < need {
				chunk := make([]byte, need-got)
				n, err := r.Read(chunk)
				if err != nil {
					log(c, "closing on read(%d) failure; %v\n", need-got, err)
					return
				}
				got += n
				bin = append(bin, chunk...)
			}
			log(c, "%d-byte payload\n%s", need-1, hex.Dump(bin[:need-1]))
		}
	}
}

func main() {
	start = time.Now()

	port := ":10091"
	arguments := os.Args
	if len(arguments) > 1 {
		port = ":" + arguments[1]
	}

	l, err := net.Listen("tcp4", port)
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
