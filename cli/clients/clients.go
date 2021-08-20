package clients

import (
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/run"
	log "github.com/sirupsen/logrus"
	"gopkg.in/yaml.v2"
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
func (c *Clients) PushConfig(id uint, config run.ScopeConfig) error {

	client, exists := c.ClientsMap[id]
	if !exists {
		return errors.New("Client not found")
	}

	if client.ProcessStart.Info.Process.Pid == 0 {
		return errors.New("No Client PID saved")
	}

	confPath := "/tmp/conf." + fmt.Sprint(client.ProcessStart.Info.Process.Pid)
	reloadPath := "/tmp/scope." + fmt.Sprint(client.ProcessStart.Info.Process.Pid)
	reloadEnv := "SCOPE_CONF_RELOAD=" + confPath

	// Marshal the config changes into a byte array
	b, err := yaml.Marshal(config)
	if err != nil {
		return err
	}

	// Create and open the new config file
	cf, err := os.OpenFile(confPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}

	// Write the new config
	_, err = cf.Write([]byte(string(b)))
	if err != nil {
		cf.Close()
		return err
	}

	cf.Close()

	log.Info("Config change pushed to client " + fmt.Sprint(client.Id))

	// Open the reload file. If the file doesn't exist, create it.
	rf, err := os.OpenFile(reloadPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer rf.Close()

	// Write the reload commands
	_, err = rf.Write([]byte(reloadEnv))
	if err != nil {
		return err
	}

	log.Info("Config reload requested for client " + fmt.Sprint(client.Id))

	return nil
}

// Push Config to a group of clients
func (c *Clients) PushGroupConfig(id uint) error {

	group, err := c.Groups.Read(id)
	if err != nil {
		return err
	}
	cfg := group.Config

	for _, client := range c.ClientsMap {

		clientTags := client.ProcessStart.Info.Configuration.Current.Metric.Format.Tags
		clientHost := client.ProcessStart.Info.Process.Hostname
		clientProc := client.ProcessStart.Info.Process.ProcName

		if group.Filter.HostName != "" && group.Filter.HostName != clientHost {
			continue
		}
		if group.Filter.ProcName != "" && group.Filter.ProcName != clientProc {
			continue
		}

		tagsSubset := true

		// Walk filterTags, looking for matching key value pairs in clientTags
		// If a filterTags key does not exist (or it's value doesn't match) in
		// clientTags, skip that client
		for k, v := range group.Filter.Tags {
			if val, exists := clientTags[k]; exists {
				if v != val {
					tagsSubset = false
					break
				}
			} else {
				tagsSubset = false
				break
			}
		}
		if !tagsSubset {
			continue
		}

		if err := c.PushConfig(client.Id, cfg); err != nil {
			return err
		}
	}

	return nil
}
