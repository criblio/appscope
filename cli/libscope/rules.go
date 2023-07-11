package libscope

// Rules represents a libscope rules
type Rules struct {
	Allow  []Entry `json:"allow,omitempty" yaml:"allow,omitempty"`
	Source Source  `json:"source,omitempty" yaml:"source,omitempty"`
}

type Entry struct {
	ProcName      string      `json:"procname,omitempty" yaml:"procname,omitempty"`
	ProcArg       string      `json:"arg,omitempty" yaml:"arg,omitempty"`
	SourceId      string      `json:"sourceid,omitempty" yaml:"sourceid,omitempty"`
	Configuration ScopeConfig `json:"config,omitempty" yaml:"config,omitempty"`
}

type Source struct {
	UnixSocketPath string `yaml:"unixSocketPath"`
	AuthToken      string `yaml:"authToken"`
}
