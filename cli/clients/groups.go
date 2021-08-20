package clients

import (
	"errors"
	"sync"

	"github.com/criblio/scope/run"
)

// Group is an object model for our client groups
type Group struct {
	Id     uint            `json:"id"`
	Name   string          `json:"name"`
	Config run.ScopeConfig `json:"config"`
	Filter Filter          `json:"filter"`
}

// Groups is an object model for group storage
type Groups struct {
	Id        uint
	GroupsMap map[uint]*Group
	*sync.RWMutex
}

// Filter is an object model for filter
type Filter struct {
	HostName string                 `json:"hostName,omitempty"`
	ProcName string                 `json:"procName,omitempty"`
	Tags     map[string]interface{} `json:"tags,omitempty"`
}

// NewGroups constructs and returns a new Groups object
func NewGroups() Groups {
	return Groups{
		GroupsMap: make(map[uint]*Group),
		RWMutex:   new(sync.RWMutex),
	}
}

// Search for a Group in Groups
func (g *Groups) Search() []Group {
	g.RLock()
	defer g.RUnlock()

	groups := make([]Group, 0, g.Id)

	for _, g := range g.GroupsMap {
		groups = append(groups, *g)
	}

	return groups
}

// Create a Group in Groups
func (g *Groups) Create(name string, config run.ScopeConfig, filter Filter) *Group {
	g.Lock()
	defer g.Unlock()

	group := &Group{
		Id:     g.Id,
		Name:   name,
		Config: config,
		Filter: filter,
	}

	g.GroupsMap[g.Id] = group

	g.Id++

	return group
}

// Read a Group from Groups
func (g *Groups) Read(id uint) (*Group, error) {
	g.RLock()
	defer g.RUnlock()

	if _, exists := g.GroupsMap[id]; !exists {
		return nil, errors.New("Group not found")
	}

	return g.GroupsMap[id], nil
}

// Update a Group in Groups
func (g *Groups) Update(id uint, obj interface{}) error {
	g.Lock()
	defer g.Unlock()

	if _, exists := g.GroupsMap[id]; !exists {
		return errors.New("Group not found")
	}

	switch o := obj.(type) {
	// Update name
	case string:
		g.GroupsMap[id].Name = o
	// Update Config
	case run.ScopeConfig:
		g.GroupsMap[id].Config = o
	// Update Filter
	case Filter:
		g.GroupsMap[id].Filter = o
	default:
		return errors.New("Invalid type passed to Update")
	}

	return nil
}

// Delete a Group from Groups
func (g *Groups) Delete(id uint) error {
	g.Lock()
	defer g.Unlock()

	if _, exists := g.GroupsMap[id]; !exists {
		return errors.New("Group not found")
	}

	delete(g.GroupsMap, id)

	return nil
}
