package clients

import (
	"sync"

	"github.com/criblio/scope/libscope"
)

type Clients struct {
	Clients ClientsMap
	*sync.RWMutex
}

type ClientsMap map[uint]Client

// Client is an object model for our unix clients
type Client struct {
	Conn         UnixConnection
	OutBuffer    string
	ProcessStart libscope.Header
}

// NewClient constructs and returns a new Client object
func NewClient(conn UnixConnection, psm libscope.Header) *Client {
	return &Client{
		Conn:         conn,
		ProcessStart: psm,
	}
}

// NewClients constructs and returns a new Clients object
func NewClients() Clients {
	return Clients{
		Clients: make(ClientsMap),
	}
}
