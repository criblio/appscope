package run

import (
	"fmt"
	"io/ioutil"
	"path/filepath"

	"gopkg.in/yaml.v2"
)

// ScopeConfig represents our current running configuration
type ScopeConfig struct {
	Metric   ScopeMetricConfig   `mapstructure:"metric" json:"metric" yaml:"metric"`
	Event    ScopeEventConfig    `mapstructure:"event" json:"event" yaml:"event"`
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
	Enable bool               `mapstructure:"enable" json:"enable" yaml:"enable"`
	Format ScopeOutputFormat  `mapstructure:"format" json:"format" yaml:"format"`
	Watch  []ScopeWatchConfig `mapstructure:"watch" json:"watch" yaml:"watch"`
}

// ScopeWatchConfig represents a watch configuration
type ScopeWatchConfig struct {
	WatchType string `mapstructure:"type" json:"type" yaml:"type"`
	Name      string `mapstructure:"name" json:"name" yaml:"name"`
	Value     string `mapstructure:"value" json:"value" yaml:"value"`
}

// ScopeLibscopeConfig represents how to configure libscope
type ScopeLibscopeConfig struct {
	Level         string         `mapstructure:"level" json:"level" yaml:"level"`
	Transport     ScopeTransport `mapstructure:"transport" json:"transport" yaml:"transport"`
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

// GetDefaultScopeConfig returns the default scope configuration as a struct
func GetDefaultScopeConfig(workDir string) *ScopeConfig {
	return &ScopeConfig{
		Metric: ScopeMetricConfig{
			Enable: true,
			Format: ScopeOutputFormat{
				FormatType: "metricjson",
				Verbosity:  4,
			},
			Transport: ScopeTransport{
				TransportType: "file",
				Path:          filepath.Join(workDir, "metrics.dogstatsd"),
			},
		},
		Event: ScopeEventConfig{
			Enable: true,
			Format: ScopeOutputFormat{
				FormatType: "ndjson",
			},
			Watch: []ScopeWatchConfig{
				{
					WatchType: "file",
					Name:      ".*log.*",
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
					Value:     ".*",
				},
			},
		},
		Libscope: ScopeLibscopeConfig{
			Transport: ScopeTransport{
				TransportType: "file",
				Path:          filepath.Join(workDir, "events.json"),
			},
			SummaryPeriod: 10,
			CommandDir:    filepath.Join(workDir, "cmd"),
			Log: ScopeLogConfig{
				Level: "error",
				Transport: ScopeTransport{
					TransportType: "file",
					Path:          filepath.Join(workDir, "cscope.log"),
					Buffering:     "line",
				},
			},
		},
	}
}

// WriteScopeConfig writes a scope config to a file
func WriteScopeConfig(sc *ScopeConfig, path string) error {
	scb, err := yaml.Marshal(sc)
	if err != nil {
		return fmt.Errorf("error marshaling's ScopeConfig to YAML: %v", err)
	}
	err = ioutil.WriteFile(path, scb, 0644)
	if err != nil {
		return fmt.Errorf("error writing ScopeConfig to file %s: %v", path, err)
	}
	return nil
}
