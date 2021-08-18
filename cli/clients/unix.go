package clients

import (
	"net"
	"os"
)

const socketFile = "@abstest"

// UnixConnection is a unix domain socket connection object model
type UnixConnection struct {
	Conn net.Conn
}

// Listen removes a socketFile and tells the os to listen
// for connections.
// Be sure to close the listener when finished
func Listen() (net.Listener, error) {
	os.Remove(socketFile)
	listener, err := net.Listen("unix", socketFile)
	if err != nil {
		return nil, err
	}
	return listener, nil
}
