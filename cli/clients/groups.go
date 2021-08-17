package clients

import (
	"errors"
	"sync"

	"github.com/criblio/scope/libscope"
)

// Group is an object model for our client groups
type Group struct {
	Name    string
	Config  libscope.HeaderConf
	Filters map[string]interface{}
}

// Groups is an object model for group storage
type Groups struct {
	GroupsMap map[string]*Group
	*sync.RWMutex
}

// NewGroups constructs and returns a new Groups object
func NewGroups() Groups {
	return Groups{
		GroupsMap: make(map[string]*Group),
	}
}

// Create a group in Groups
func (g *Groups) Create(name string, config libscope.HeaderConf, filters map[string]interface{}) error {
	g.Lock()
	defer g.Unlock()

	if _, exists := g.GroupsMap[name]; exists {
		return errors.New("Group already exists")
	}

	group := Group{
		Name:    name,
		Config:  config,
		Filters: filters,
	}

	g.GroupsMap[name] = &group

	return nil
}

// Read a group from Groups
func (g *Groups) Read(name string) (*Group, error) {
	g.RLock()
	defer g.RUnlock()

	if _, exists := g.GroupsMap[name]; !exists {
		return nil, errors.New("Group not found")
	}

	return g.GroupsMap[name], nil
}

// Update a group in Groups
func (g *Groups) Update(name string, config libscope.HeaderConf, filters map[string]interface{}) error {
	g.Lock()
	defer g.Unlock()

	if _, exists := g.GroupsMap[name]; !exists {
		return errors.New("Group not found")
	}

	if config != (libscope.HeaderConf{}) {
		g.GroupsMap[name].Config = config
	}
	if filters != nil {
		g.GroupsMap[name].Filters = filters
	}

	return nil
}

// Delete a group from Groups
func (g *Groups) Delete(name string) error {
	g.Lock()
	defer g.Unlock()

	if _, exists := g.GroupsMap[name]; !exists {
		return errors.New("Group not found")
	}

	delete(g.GroupsMap, name)

	return nil
}
