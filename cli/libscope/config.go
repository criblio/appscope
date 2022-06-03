package libscope

// ScopeConfig represents our current running configuration
type ScopeConfig struct {
	Cribl    ScopeCriblConfig    `mapstructure:"cribl,omitempty" json:"cribl,omitempty" yaml:"cribl"`
	Metric   ScopeMetricConfig   `mapstructure:"metric,omitempty" json:"metric,omitempty" yaml:"metric,omitempty"`
	Event    ScopeEventConfig    `mapstructure:"event,omitempty" json:"event,omitempty" yaml:"event,omitempty"`
	Payload  ScopePayloadConfig  `mapstructure:"payload,omitempty" json:"payload,omitempty" yaml:"payload,omitempty"`
	Libscope ScopeLibscopeConfig `mapstructure:"libscope" json:"libscope" yaml:"libscope"`
}

// ScopeCriblConfig represents how to output metrics
type ScopeCriblConfig struct {
	Enable    bool           `mapstructure:"enable" json:"enable" yaml:"enable"`
	Transport ScopeTransport `mapstructure:"transport" json:"transport" yaml:"transport"`
	AuthToken string         `mapstructure:"authtoken" json:"authtoken" yaml:"authtoken"`
}

// ScopeMetricConfig represents how to output metrics
type ScopeMetricConfig struct {
	Enable    bool              `mapstructure:"enable" json:"enable" yaml:"enable"`
	Format    ScopeOutputFormat `mapstructure:"format" json:"format" yaml:"format"`
	Transport ScopeTransport    `mapstructure:"transport,omitempty" json:"transport,omitempty" yaml:"transport,omitempty"`
}

// ScopeEventConfig represents how to output events
type ScopeEventConfig struct {
	Enable    bool               `mapstructure:"enable" json:"enable" yaml:"enable"`
	Format    ScopeOutputFormat  `mapstructure:"format" json:"format" yaml:"format"`
	Transport ScopeTransport     `mapstructure:"transport,omitempty" json:"transport,omitempty" yaml:"transport,omitempty"`
	Watch     []ScopeWatchConfig `mapstructure:"watch" json:"watch" yaml:"watch"`
}

// ScopePayloadConfig represents how to capture payloads
type ScopePayloadConfig struct {
	Enable bool   `mapstructure:"enable" json:"enable" yaml:"enable"`
	Dir    string `mapstructure:"dir" json:"dir" yaml:"dir"`
}

// ScopeWatchConfig represents a watch configuration
type ScopeWatchConfig struct {
	WatchType   string `mapstructure:"type" json:"type" yaml:"type"`
	Name        string `mapstructure:"name" json:"name" yaml:"name"`
	Field       string `mapstructure:"field,omitempty" json:"field,omitempty" yaml:"field,omitempty"`
	Value       string `mapstructure:"value" json:"value" yaml:"value"`
	AllowBinary bool   `mapstructure:"allowbinary,omitempty" json:"allowbinary,omitempty" yaml:"allowbinary,omitempty"`
}

// ScopeLibscopeConfig represents how to configure libscope
type ScopeLibscopeConfig struct {
	ConfigEvent   bool           `mapstructure:"configevent" json:"configevent" yaml:"configevent"`
	SummaryPeriod int            `mapstructure:"summaryperiod" jaon:"summaryperiod" yaml:"summaryperiod"`
	CommandDir    string         `mapstructure:"commanddir" json:"commanddir" yaml:"commanddir"`
	Log           ScopeLogConfig `mapstructure:"log" json:"log" yaml:"log"`
}

// ScopeLogConfig represents how to configure libscope logs
type ScopeLogConfig struct {
	Level     string         `mapstructure:"level" json:"level" yaml:"level"`
	Transport ScopeTransport `mapstructure:"transport" json:"transport" yaml:"transport"`
}

// ScopeOutputFormat represents how to output a data type
type ScopeOutputFormat struct {
	FormatType   string            `mapstructure:"type" json:"type" yaml:"type"`
	Statsdmaxlen int               `mapstructure:"statsdmaxlen,omitempty" json:"statsdmaxlen,omitempty" yaml:"statsdmaxlen,omitempty"`
	Verbosity    int               `mapstructure:"verbosity,omitempty" json:"verbosity,omitempty" yaml:"verbosity,omitempty"`
	Tags         map[string]string `mapstructure:"tags,omitempty" json:"tags,omitempty" yaml:"tags,omitempty"`
}

// ScopeTransport represents how we transport data
type ScopeTransport struct {
	TransportType string    `mapstructure:"type" json:"type" yaml:"type"`
	Host          string    `mapstructure:"host,omitempty" json:"host,omitempty" yaml:"host,omitempty"`
	Port          int       `mapstructure:"port,omitempty" json:"port,omitempty" yaml:"port,omitempty"`
	Path          string    `mapstructure:"path,omitempty" json:"path,omitempty" yaml:"path,omitempty"`
	Buffering     string    `mapstructure:"buffering,omitempty" json:"buffering,omitempty" yaml:"buffering,omitempty"`
	Tls           TlsConfig `mapstructure:"tls" json:"tls" yaml:"tls"`
}

// TlsConfig is used when
type TlsConfig struct {
	Enable         bool   `mapstructure:"enable" json:"enable" yaml:"enable"`
	ValidateServer bool   `mapstructure:"validateserver" json:"validateserver" yaml:"validateserver"`
	CaCertPath     string `mapstructure:"cacertpath" json:"cacertpath" yaml:"cacertpath"`
}
