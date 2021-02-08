package run

import (
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v2"
)

// ScopeConfig represents our current running configuration
type ScopeConfig struct {
	Metric   ScopeMetricConfig   `mapstructure:"metric" json:"metric" yaml:"metric"`
	Event    ScopeEventConfig    `mapstructure:"event" json:"event" yaml:"event"`
	Payload  ScopePayloadConfig  `mapstructure:"payload,omitempty" json:"payload,omitempty" yaml:"payload,omitempty"`
	Libscope ScopeLibscopeConfig `mapstructure:"libscope" json:"libscope" yaml:"libscope"`
}

// ScopeMetricConfig represents how to output metrics
type ScopeMetricConfig struct {
	Enable    bool              `mapstructure:"enable" json:"enable" yaml:"enable"`
	Format    ScopeOutputFormat `mapstructure:"format" json:"format" yaml:"format"`
	Transport ScopeTransport    `mapstructure:"transport" json:"transport" yaml:"transport"`
}

// ScopeEventConfig represents how to output events
type ScopeEventConfig struct {
	Enable    bool               `mapstructure:"enable" json:"enable" yaml:"enable"`
	Format    ScopeOutputFormat  `mapstructure:"format" json:"format" yaml:"format"`
	Transport ScopeTransport     `mapstructure:"transport" json:"transport" yaml:"transport"`
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
	Level         string         `mapstructure:"level" json:"level" yaml:"level"`
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
	TransportType string `mapstructure:"type" json:"type" yaml:"type"`
	Host          string `mapstructure:"host,omitempty" json:"host,omitempty" yaml:"host,omitempty"`
	Port          int    `mapstructure:"port,omitempty" json:"port,omitempty" yaml:"port,omitempty"`
	Path          string `mapstructure:"path,omitempty" json:"path,omitempty" yaml:"path,omitempty"`
	Buffering     string `mapstructure:"buffering,omitempty" json:"buffering,omitempty" yaml:"buffering,omitempty"`
}

// setDefault sets the default scope configuration as a struct
func (c *Config) setDefault() error {
	if c.WorkDir == "" {
		return fmt.Errorf("workDir not set")
	}
	c.sc = &ScopeConfig{
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
					WatchType: "http",
					Name:      ".*",
					Field:     ".*",
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
				Level: "error",
				Transport: ScopeTransport{
					TransportType: "file",
					Path:          filepath.Join(c.WorkDir, "ldscope.log"),
					Buffering:     "line",
				},
			},
		},
	}
	return nil
}

// configFromRunOpts writes the scope configuration to the workdir
func (c *Config) configFromRunOpts() error {
	err := c.setDefault()
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
		if strings.HasPrefix(dest, "tcp://") || strings.HasPrefix(dest, "udp://") {
			if strings.HasPrefix(dest, "tcp") {
				t.TransportType = "tcp"
			} else {
				t.TransportType = "udp"
			}
			parts := strings.Split(dest[6:], ":")
			if len(parts) != 2 {
				return fmt.Errorf("Cannot parse dest: %s", dest)
			}
			t.Host = parts[0]
			port, err := strconv.Atoi(parts[1])
			if err != nil {
				return fmt.Errorf("Cannot parse port from %s: %s", dest, parts[1])
			}
			t.Port = port
			t.Path = ""
			return nil
		} else if strings.HasPrefix(dest, "file://") {
			t.TransportType = "file"
			t.Path = dest[7:]
			return nil
		}
		return fmt.Errorf("Invalid dest: %s", dest)
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
	return nil
}

// WriteScopeConfig writes a scope config to a file
func (c *Config) WriteScopeConfig(path string) error {
	if c.sc == nil {
		err := c.configFromRunOpts()
		if err != nil {
			return err
		}
	}
	scb, err := yaml.Marshal(c.sc)
	if err != nil {
		return fmt.Errorf("error marshaling's ScopeConfig to YAML: %v", err)
	}
	err = ioutil.WriteFile(path, scb, 0644)
	if err != nil {
		return fmt.Errorf("error writing ScopeConfig to file %s: %v", path, err)
	}
	return nil
}

func scopeLogRegex() string {
	return `[\s\/\\\.]log[s]?[\/\\\.]?`
}
