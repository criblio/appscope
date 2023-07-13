package listener

import (
	"bufio"
	"encoding/json"
	"io"
	"net"
	"os"

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
func ListenAndServer(addr string, slacktoken string, channeldId string) {
	notifyChan := make(chan string, 1)
	processChan := make(chan libscope.EventBody, 512)
	var n notify.Notifier
	if slacktoken != "" && channeldId != "" {
		n = notify.NewSlackNotifier(slacktoken, channeldId)
	} else {
		n = notify.NewFileNotifier(os.Stdout)
	}

	go listenScopeEventServer(addr, processChan)
	go processScopeData(processChan, notifyChan)
	for {
		select {
		case notifyEvent := <-notifyChan:
			n.Notify(notifyEvent)
		}
	}
}

// Handles incoming scope data.
func listenScopeData(conn net.Conn, processChan chan<- libscope.EventBody) {
	decoder := json.NewDecoder(conn)
	bufio.NewReader(conn)

	for {
		var obj libscope.Event
		if err := decoder.Decode(&obj); err == io.EOF {
			break
		}
		// Check only for dns objects
		if obj.Body.Source == "dns.req" || obj.Body.Source == "dns.resp" {
			processChan <- obj.Body
		}
	}

	conn.Close()
}
