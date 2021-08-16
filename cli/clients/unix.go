package clients

import "net"

// UnixConnection is a unix domain socket connection object model
type UnixConnection struct {
	Conn net.Conn
}
