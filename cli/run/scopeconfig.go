package run

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

// ScopeConfig represents our current running configuration
type ScopeConfig struct {
	Cribl    ScopeCriblConfig    `mapstructure:"cribl,omitempty" json:"cribl,omitempty" yaml:"cribl,omitempty"`
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
	WatchType string `mapstructure:"type" json:"type" yaml:"type"`
	Name      string `mapstructure:"name" json:"name" yaml:"name"`
	Field     string `mapstructure:"field,omitempty" json:"field,omitempty" yaml:"field,omitempty"`
	Value     string `mapstructure:"value" json:"value" yaml:"value"`
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

// SetDefault sets the default scope configuration as a struct
func (c *Config) SetDefault() error {
	if c.WorkDir == "" {
		return fmt.Errorf("workDir not set")
	}
	c.sc = &ScopeConfig{
		Cribl: ScopeCriblConfig{},
		Metric: ScopeMetricConfig{
			Enable: true,
			Format: ScopeOutputFormat{
				FormatType: "ndjson",
				Verbosity:  4,
			},
			Transport: ScopeTransport{
				TransportType: "file",
				Path:          filepath.Join(c.WorkDir, "metrics.json"),
				Buffering:     "line",
			},
		},
		Event: ScopeEventConfig{
			Enable: true,
			Format: ScopeOutputFormat{
				FormatType: "ndjson",
			},
			Transport: ScopeTransport{
				TransportType: "file",
				Path:          filepath.Join(c.WorkDir, "events.json"),
				Buffering:     "line",
			},
			Watch: []ScopeWatchConfig{
				{
					WatchType: "file",
					Name:      scopeLogRegex(),
					Value:     ".*",
				},
				{
					WatchType: "console",
					Name:      "(stdout|stderr)",
					Value:     ".*",
				},
				{
					WatchType: "net",
					Name:      ".*",
					Field:     ".*",
					Value:     ".*",
				},
				{
					WatchType: "fs",
					Name:      ".*",
					Field:     ".*",
					Value:     ".*",
				},
				{
					WatchType: "dns",
					Name:      ".*",
					Field:     ".*",
					Value:     ".*",
				},
				// {
				// 	WatchType: "metric",
				// 	Name:      ".*",
				// 	Value:     ".*",
				// },
			},
		},
		Libscope: ScopeLibscopeConfig{
			SummaryPeriod: 10,
			CommandDir:    filepath.Join(c.WorkDir, "cmd"),
			ConfigEvent:   false,
			Log: ScopeLogConfig{
				Level: "warning",
				Transport: ScopeTransport{
					TransportType: "file",
					Path:          filepath.Join(c.WorkDir, "ldscope.log"),
					Buffering:     "line",
				},
			},
		},
	}

	if c.CriblDest == "" {
		c.sc.Event.Watch = append(c.sc.Event.Watch, ScopeWatchConfig{
			WatchType: "http",
			Name:      ".*",
			Field:     ".*",
			Value:     ".*",
		})
	}

	return nil
}

// ConfigFromFile loads a configuration from a yml file
func (c *Config) ConfigFromFile() error {
	c.sc = &ScopeConfig{}

	yamlFile, err := ioutil.ReadFile(c.UserConfig)
	if err != nil {
		return err
	}
	if err = yaml.Unmarshal(yamlFile, c.sc); err != nil {
		return err
	}

	return nil
}

// configFromRunOpts creates a configuration from run options
func (c *Config) configFromRunOpts() error {
	err := c.SetDefault()
	if err != nil {
		return err
	}

	if c.Verbosity != 0 {
		c.sc.Metric.Format.Verbosity = c.Verbosity
	}

	if c.Payloads {
		c.sc.Payload = ScopePayloadConfig{
			Enable: true,
			Dir:    filepath.Join(c.WorkDir, "payloads"),
		}
	}

	if c.MetricsFormat != "" {
		if c.MetricsFormat != "ndjson" && c.MetricsFormat != "statsd" {
			return fmt.Errorf("invalid metrics format %s", c.MetricsFormat)
		}
		c.sc.Metric.Format.FormatType = c.MetricsFormat
	}

	parseDest := func(t *ScopeTransport, dest string) error {
		//
		// The regexp matches "proto://something:port" where the leading
		// "proto://" is optional. If given, "proto" must be "tcp", "udp",
		// "tls" or "file". The ":port" suffix is optional and ignored
		// for files but required for sockets.
		//
		//     "relative/path"
		//     "file://another/relative/path"
		//     "/absolute/path"
		//     "file:///another/absolute/path"
		//     "tcp://host:port"
		//     "udp://host:port"
		//     "tls://host:port"
		//     "host:port" (assume tls)
		//
		// m[0][0] is the full match
		// m[0][1] is the "proto" string or an empty string
		// m[0][2] is the "something" string
		// m[0][3] is the "port" string or an empty string
		//
		m := regexp.MustCompile("^(?:(?i)(file|tcp|udp|tls)://)?([^:]+)(?::(\\d+))?$").FindAllStringSubmatch(dest, -1)
		//fmt.Printf("debug: m=%+v\n", m)
		if len(m) <= 0 {
			// no match
			return fmt.Errorf("Invalid dest: %s", dest)
		} else if m[0][1] != "" {
			// got "proto://..."
			proto := strings.ToLower(m[0][1])
			if proto == "udp" || proto == "tcp" {
				// clear socket
				t.TransportType = proto
				t.Host = m[0][2]
				if m[0][3] == "" {
					return fmt.Errorf("Missing :port at the end of %s", dest)
				}
				port, err := strconv.Atoi(m[0][3])
				if err != nil {
					return fmt.Errorf("Cannot parse port from %s: %s", dest, m[0][3])
				}
				t.Port = port
				t.Path = ""
				t.Tls.Enable = false
				t.Tls.ValidateServer = true
			} else if proto == "tls" {
				// encrypted socket
				t.TransportType = "tcp"
				t.Host = m[0][2]
				if m[0][3] == "" {
					return fmt.Errorf("Missing :port at the end of %s", dest)
				}
				port, err := strconv.Atoi(m[0][3])
				if err != nil {
					return fmt.Errorf("Cannot parse port from %s: %s", dest, m[0][3])
				}
				t.Port = port
				t.Path = ""
				t.Tls.Enable = true
				t.Tls.ValidateServer = true
			} else if proto == "file" {
				t.TransportType = proto
				t.Path = m[0][2]
			} else {
				return fmt.Errorf("Invalid dest protocol: %s", proto)
			}
		} else if m[0][3] != "" {
			// got "something:port", assume tls://
			t.TransportType = "tcp"
			t.Host = m[0][2]
			port, err := strconv.Atoi(m[0][3])
			if err != nil {
				return fmt.Errorf("Cannot parse port from %s: %s", dest, m[0][3])
			}
			t.Port = port
			t.Path = ""
			t.Tls.Enable = true
			t.Tls.ValidateServer = true
		} else {
			// got "something", assume file://
			t.TransportType = "file"
			t.Path = m[0][2]
		}
		//fmt.Printf("debug: t=%+v\n", t)
		return nil // success
	}

	if c.MetricsDest != "" {
		err := parseDest(&c.sc.Metric.Transport, c.MetricsDest)
		if err != nil {
			return err
		}
	}

	if c.EventsDest != "" {
		err := parseDest(&c.sc.Event.Transport, c.EventsDest)
		if err != nil {
			return err
		}
	}

	// Add AuthToken to config regardless of cribldest being set
	// To support mixing of config and environment variables
	c.sc.Cribl.AuthToken = c.AuthToken

	if c.CriblDest != "" {
		err := parseDest(&c.sc.Cribl.Transport, c.CriblDest)
		if err != nil {
			return err
		}
		c.sc.Cribl.Enable = true
		// If we're outputting to Cribl, disable metrics and event outputs
		c.sc.Metric.Transport = ScopeTransport{}
		c.sc.Event.Transport = ScopeTransport{}
		c.sc.Libscope.ConfigEvent = true
	}

	if c.Loglevel != "" {
		levels := map[string]bool{
			"error":   true,
			"warning": true,
			"info":    true,
			"debug":   true,
			"none":    true,
		}
		if _, ok := levels[c.Loglevel]; !ok {
			return fmt.Errorf("%s is an invalid log level. Log level must be one of debug, info, warning, error, or none", c.Loglevel)
		}
		c.sc.Libscope.Log.Level = c.Loglevel
	}
	return nil
}

// ScopeConfigYaml serializes Config to YAML
func (c *Config) ScopeConfigYaml() ([]byte, error) {
	if c.sc == nil {
		err := c.configFromRunOpts()
		if err != nil {
			return []byte{}, err
		}
	}
	scb, err := yaml.Marshal(c.sc)
	if err != nil {
		return scb, fmt.Errorf("error marshaling's ScopeConfig to YAML: %v", err)
	}
	return scb, nil
}

// WriteScopeConfig writes a scope config to a file
func (c *Config) WriteScopeConfig(path string, filePerms os.FileMode) error {
	scb, err := c.ScopeConfigYaml()
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(path, scb, filePerms)
	if err != nil {
		return fmt.Errorf("error writing ScopeConfig to file %s: %v", path, err)
	}
	return nil
}

func scopeLogRegex() string {
	// see scopeconfig_test.go for example paths that match
	return `(\/logs?\/)|(\.log$)|(\.log[.\d])`
}
