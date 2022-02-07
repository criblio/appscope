package live

import (
	"net"
	"os"
)

// Abstract socket
const scopeSock = "@scopelive"

// UnixConnection is a unix domain socket connection object model
type UnixConnection struct {
	Conn net.Conn
}

// Listen removes a socket file and tells the os to listen
// for connections.
// Be sure to close the listener when finished
func Listen() (net.Listener, error) {
	os.Remove(scopeSock)
	listener, err := net.Listen("unix", scopeSock)
	if err != nil {
		return nil, err
	}
	return listener, nil
}
