package clients

import (
	"errors"
	"sync"

	"github.com/criblio/scope/libscope"
)

// Client is an object model for our unix clients
type Client struct {
	Conn         UnixConnection
	OutBuffer    string
	ProcessStart libscope.Header
}

// Clients is an object model for client storage
type Clients struct {
	ClientsMap map[uint]*Client
	*sync.RWMutex
	Groups
}

// NewClients constructs and returns a new Clients object
func NewClients() Clients {
	return Clients{
		ClientsMap: make(map[uint]*Client),
		Groups:     NewGroups(),
	}
}

// Create a client in Clients
func (c *Clients) Create(id uint, conn UnixConnection, psm libscope.Header) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.ClientsMap[id]; exists {
		return errors.New("Client already exists")
	}

	client := Client{
		Conn:         conn,
		ProcessStart: psm,
	}

	c.ClientsMap[id] = &client

	return nil
}

// Read a client from Clients
func (c *Clients) Read(id uint) (*Client, error) {
	c.RLock()
	defer c.RUnlock()

	if _, exists := c.ClientsMap[id]; !exists {
		return nil, errors.New("Client not found")
	}

	return c.ClientsMap[id], nil
}

// Update a client in Clients
func (c *Clients) Update(id uint, conn UnixConnection, psm libscope.Header) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.ClientsMap[id]; !exists {
		return errors.New("Client not found")
	}

	if conn.Conn != nil {
		c.ClientsMap[id].Conn = conn
	}
	if psm.Format != "" {
		c.ClientsMap[id].ProcessStart = psm
	}

	return nil
}

// Delete a client from Clients
func (c *Clients) Delete(id uint) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.ClientsMap[id]; !exists {
		return errors.New("Client not found")
	}

	delete(c.ClientsMap, id)

	return nil
}
