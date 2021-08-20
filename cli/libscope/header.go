package libscope

import "github.com/criblio/scope/run"

// Header represents the JSON object a client sends to
// LogStream when it connects.
type Header struct {
	Format    string     `json:"format,omitempty"`    // `ndjson` for event/metric connections or `scope` for payloads
	AuthToken string     `json:"authToken,omitempty"` // optional authentication token
	Info      HeaderInfo `json:"info,omitempty"`      // the rest of the header information
}

// The `info` node in the JSON header object
type HeaderInfo struct {
	Configuration HeaderConf        `json:"configuration,omitempty"` // AppScope configuration; i.e. `scope.yml`
	Environment   map[string]string `json:"environment,omitempty"`   // Environment variables
	Process       HeaderProcess     `json:"process,omitempty"`       // Process details
}

// HeaderConf is the `info.configuration` node in the JSON header object
type HeaderConf struct {
	Current run.ScopeConfig `json:"current,omitempty"`
}

// HeaderProcess is the `info.process` node in the JSON header object
type HeaderProcess struct {
	Cmd         string `json:"cmd,omitempty"`
	Hostname    string `json:"hostname,omitempty"`
	Id          string `json:"id,omitempty"`
	LibScopeVer string `json:"libscopever,omitempty"`
	Pid         int    `json:"pid,omitempty"`
	Ppid        int    `json:"ppid,omitempty"`
	ProcName    string `json:"procname"`
}

// NewHeader creates a Header object
func NewHeader() Header {
	header := Header{}

	return header
}
