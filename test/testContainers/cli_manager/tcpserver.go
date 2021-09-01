//
// Debug TCP Server that Mimics LogStream's AppScope Source/Input
//
// `go run tcpserver.go | egrep -e ^ -e '"(type|src|protocol)":"[^"]*"'`
//

package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

var port int
var payloadsEnabled bool
var prettyEnabled bool
var timestampsEnabled bool

var count uint64

var start time.Time

func log(c net.Conn, format string, args ...interface{}) {
	if timestampsEnabled {
		fmt.Printf(fmt.Sprintf("%8s %s ", time.Since(start).Truncate(time.Millisecond).String(), c.RemoteAddr().String())+format, args...)
	} else {
		fmt.Printf(c.RemoteAddr().String()+" "+format, args...)
	}
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

		var header map[string]interface{}
		if err := json.Unmarshal([]byte(data), &header); err != nil {
			log(c, "closing on JSON error; %v\n%s", err, hex.Dump([]byte(data)))
			return
		}

		if prettyEnabled {
			headerJSON, _ := json.MarshalIndent(header, "", "  ")
			log(c, "%s\n", string(headerJSON))
		} else {
			if strings.Contains(data, "http-req") {
				fmt.Println("Received : ", atomic.AddUint64(&count, 1))
			}
			log(c, data)
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
			if payloadsEnabled {
				log(c, "%d-byte payload\n%s", need-1, hex.Dump(bin[:need]))
			}
		}
	}
}

func main() {

	// command-line options
	flag.IntVar(&port, "p", 9109, "TCP port to listen on")
	flag.BoolVar(&payloadsEnabled, "x", false, "Enable hexdumpsed payloads")
	flag.BoolVar(&prettyEnabled, "j", false, "Enable pretty-printed JSON headers")
	flag.BoolVar(&timestampsEnabled, "t", false, "Enabled relative-timestamped output")
	flag.Parse()

	start = time.Now()

	l, err := net.Listen("tcp4", ":"+strconv.Itoa(port))
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
