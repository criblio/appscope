package listener

import (
	"fmt"
	"net"

	"github.com/criblio/scope/notify"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

func listenServer(listenSource string, notifyChan chan<- string) {
	listen, err := net.Listen("tcp", listenSource)
	if err != nil {
		util.ErrAndExit("Listen: %s", err)
	}
	defer listen.Close()
	for {
		conn, err := listen.Accept()
		if err != nil {
			util.ErrAndExit("Accept %s", err)
		}
		go handleRequest(conn, notifyChan)
	}
}

func Listener(listenSource string, slacktoken string) {
	slackChan := make(chan string, 1)
	slackSender := notify.NewSlackSender(slacktoken, "testchannel")
	go listenServer(listenSource, slackChan)
	for {
		select {
		case slackEvent := <-slackChan:
			log.Debug().Msgf("Notification event: %s", slackEvent)
			slackSender.Notify(slackEvent)
		}
	}
}

// Handles incoming requests.
func handleRequest(conn net.Conn, notifyChan chan<- string) {
	// Make a buffer to hold incoming data.
	buf := make([]byte, 1024)
	// Read the incoming connection into the buffer.
	_, err := conn.Read(buf)
	if err != nil {
		fmt.Println("Error reading:", err.Error())
	}
	// if notifyLogic {
	notifyChan <- "test"
	// }
	// Send a response back to person contacting us.
	conn.Write([]byte("Message received."))
	// Close the connection when you're done with it.
	conn.Close()
}
