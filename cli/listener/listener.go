package listener

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/notify"
	"github.com/criblio/scope/util"
)

// Listen for incoming AppScope data
// Save incoming data in process channel
func listenScopeEventServer(addr string, processChan chan<- libscope.EventBody) {
	listen, err := net.Listen("tcp", addr)
	if err != nil {
		util.ErrAndExit("Listen: %s", err)
	}
	defer listen.Close()
	for {
		conn, err := listen.Accept()
		if err != nil {
			util.ErrAndExit("Accept %s", err)
		}
		go listenScopeData(conn, processChan)
	}
}

// Main method for listener and server
func ListenAndServer(addr string, slacktoken string) {
	notifyChan := make(chan string, 1)
	processChan := make(chan libscope.EventBody, 512)
	notifSender := notify.NewSlackSender(slacktoken, "testchannel")

	go listenScopeEventServer(addr, processChan)
	go processScopeData(processChan, notifyChan)
	for {
		select {
		case notifyEvent := <-notifyChan:
			notifSender.Notify(notifyEvent)
		}
	}
}

// Handles incoming scope data.
func listenScopeData(conn net.Conn, processChan chan<- libscope.EventBody) {
	buf := make([]byte, 0, 4096)
	tmp := make([]byte, 4096)
	// Read whole data
	for {
		n, err := conn.Read(tmp)
		if err != nil {
			if err != io.EOF {
				fmt.Println("read error:", err)
			}
			break
		}
		buf = append(buf, tmp[:n]...)
	}
	buffer := bytes.NewBuffer(buf)
	decoder := json.NewDecoder(buffer)

	for decoder.More() {
		var obj libscope.Event

		// Decode the next JSON object
		err := decoder.Decode(&obj)
		if err != nil {
			fmt.Println("Error decoding JSON:", err)
			return
		}
		// Check only for dns objects
		if obj.Body.Source == "dns.req" || obj.Body.Source == "dns.resp" {
			processChan <- obj.Body
		}
	}
	conn.Close()
}
