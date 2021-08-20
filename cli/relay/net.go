package relay

import (
	"fmt"
	"net"

	"github.com/criblio/scope/run"
)

// Connection is a connection object model
type Connection struct {
	Conn net.Conn
}

// NewConnection makes a connection to the Cribl Destination
func NewConnection() (*Connection, error) {

	transport, err := run.ParseDest(Config.CriblDest)
	if err != nil {
		return nil, err
	}

	proto := "tcp"
	if transport.Tls.Enable == "true" {
		proto = "tls"
	}

	addr := transport.Host + ":" + fmt.Sprint(transport.Port)

	conn, err := net.Dial(proto, addr)
	if err != nil {
		return nil, err
	}

	c := &Connection{
		Conn: conn,
	}

	return c, nil
}

// Send sends a message to the Cribl Destination
func (c *Connection) Send(msg Message) error {

	c.Conn.Write([]byte(msg))

	return nil
}
