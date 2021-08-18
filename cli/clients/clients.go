package clients

import (
	"errors"
	"sync"

	"github.com/criblio/scope/libscope"
)

// Client is an object model for our unix Clients
type Client struct {
	Id           uint
	Conn         UnixConnection
	OutBuffer    string
	ProcessStart libscope.Header
}

// Clients is an object model for Client storage
type Clients struct {
	Id         uint
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

// Search for a Client in Clients
func (c *Clients) Search() []Client {
	c.RLock()
	defer c.RUnlock()

	clients := make([]Client, c.Id)

	for _, c := range c.ClientsMap {
		clients = append(clients, *c)
	}

	return clients
}

// Create a Client in Clients
func (c *Clients) Create(conn UnixConnection, psm libscope.Header) *Client {
	c.Lock()
	defer c.Unlock()

	client := &Client{
		Id:           c.Id,
		Conn:         conn,
		ProcessStart: psm,
	}

	c.ClientsMap[c.Id] = client

	c.Id++

	return client
}

// Read a Client from Clients
func (c *Clients) Read(id uint) (Client, error) {
	c.RLock()
	defer c.RUnlock()

	if _, exists := c.ClientsMap[id]; !exists {
		return Client{}, errors.New("Client not found")
	}

	return *c.ClientsMap[id], nil
}

// Update a Client in Clients
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

// Delete a Client from Clients
func (c *Clients) Delete(id uint) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.ClientsMap[id]; !exists {
		return errors.New("Client not found")
	}

	delete(c.ClientsMap, id)

	return nil
}

// Push Config to one Client
func (c *Clients) PushConfig(id uint, config libscope.HeaderConf) {

}
