package clients

import (
	"errors"
	"sync"

	"github.com/criblio/scope/libscope"
)

// Client is an object model for our unix Clients
type Client struct {
	Id           uint            `json:"id"`
	Conn         UnixConnection  `json:"-"`
	OutBuffer    string          `json:"-"`
	ProcessStart libscope.Header `json:"processStart"`
}

// Clients is an object model for Client storage
type Clients struct {
	Id         uint
	ClientsMap map[uint]*Client
	*sync.RWMutex
	Groups
}

// NewClients constructs and returns a new Clients object
func NewClients() *Clients {
	return &Clients{
		ClientsMap: make(map[uint]*Client),
		RWMutex:    new(sync.RWMutex),
		Groups:     NewGroups(),
	}
}

// Search for a Client in Clients
func (c *Clients) Search() []Client {
	c.RLock()
	defer c.RUnlock()

	clients := make([]Client, 0, c.Id)

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
func (c *Clients) Update(id uint, obj interface{}) error {
	c.Lock()
	defer c.Unlock()

	if _, exists := c.ClientsMap[id]; !exists {
		return errors.New("Client not found")
	}

	switch o := obj.(type) {
	// Update UnixConnection
	case UnixConnection:
		c.ClientsMap[id].Conn = o
	// Update Header
	case libscope.Header:
		c.ClientsMap[id].ProcessStart = o
	default:
		return errors.New("Invalid type passed into Update")
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
func (c *Clients) PushConfig(id uint, config libscope.HeaderConfCurrent) {

}
