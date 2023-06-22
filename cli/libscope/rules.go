package libscope

// Filter represents a libscope filter
type Filter map[string][]Entry

type Entry struct {
	ProcName      string      `json:"procname,omitempty" yaml:"procname,omitempty"`
	ProcArg       string      `json:"arg,omitempty" yaml:"arg,omitempty"`
	SourceId      string      `json:"sourceid,omitempty" yaml:"sourceid,omitempty"`
	Configuration ScopeConfig `json:"config,omitempty" yaml:"config,omitempty"`
}
