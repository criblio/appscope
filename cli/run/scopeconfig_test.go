package run

import (
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v2"
)

// var defaultScopeConfigYaml string

// func init() {
// 	defaultScopeConfigYaml = `metric:
//   enable: true
//   format:
//     type: ndjson
//     verbosity: 4
//   transport:
//     type: file
// event:
//   enable: true
//   format:
//     type: ndjson
//   watch: []
// libscope:
//   level: ""
//   transport:
//     type: file
//   summaryperiod: 10
//   commanddir: /tmp
//   log:
//     level: error
//     transport:
//       type: file
//       buffering: line
// `
// }

// This may not be deterministic. Working for now.
func TestSetDefault(t *testing.T) {
	opt := Config{}
	// No workdir set, should error
	err := opt.setDefault()
	assert.Error(t, err)

	opt.WorkDir = "/foo"
	err = opt.setDefault()
	sc := opt.sc

	testYaml := testDefaultScopeConfigYaml("/foo", 4)

	configYaml, err := yaml.Marshal(sc)
	assert.NoError(t, err)
	assert.Equal(t, testYaml, string(configYaml))
}

func TestConfigFromRunOpts(t *testing.T) {
	c := Config{WorkDir: ".foo"}
	err := c.configFromRunOpts()

	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, ".foo/metrics.json", c.sc.Metric.Transport.Path)
	assert.Equal(t, 4, c.sc.Metric.Format.Verbosity)
	assert.Equal(t, "file", c.sc.Event.Transport.TransportType)
	assert.Equal(t, ".foo/events.json", c.sc.Event.Transport.Path)

	c.Verbosity = 5
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, 5, c.sc.Metric.Format.Verbosity)

	c.Payloads = true
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.True(t, c.sc.Payload.Enable)
	assert.Equal(t, ".foo/payloads", c.sc.Payload.Dir)

	c.MetricsFormat = "foo"
	err = c.configFromRunOpts()
	assert.Error(t, err)

	c.MetricsFormat = "statsd"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "statsd", c.sc.Metric.Format.FormatType)

	c.MetricsDest = "foo://bar"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Invalid dest: foo://bar"), err)

	c.MetricsDest = "file://foo"
	err = c.configFromRunOpts()
	assert.NoError(t, err)

	c.MetricsDest = "tcp://foo"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Cannot parse dest: tcp://foo"), err)

	c.MetricsDest = "tcp://foo:bar"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Cannot parse port from tcp://foo:bar: bar"), err)

	c.MetricsDest = "tcp://foo.com:8000"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "foo.com", c.sc.Metric.Transport.Host)
	assert.Equal(t, 8000, c.sc.Metric.Transport.Port)
	assert.Equal(t, "tcp", c.sc.Metric.Transport.TransportType)

	c.MetricsDest = "udp://foo.com:8000"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "foo.com", c.sc.Metric.Transport.Host)
	assert.Equal(t, 8000, c.sc.Metric.Transport.Port)
	assert.Equal(t, "udp", c.sc.Metric.Transport.TransportType)

	c.MetricsDest = "file:///path/to/file.json"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "/path/to/file.json", c.sc.Metric.Transport.Path)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
}

func TestWriteScopeConfig(t *testing.T) {
	opt := Config{WorkDir: "/foo"}
	opt.configFromRunOpts()
	var err error
	// f, err := os.OpenFile(".cantwrite", os.O_RDONLY|os.O_CREATE, 0400)
	// assert.NoError(t, err)
	// f.Close()
	// err = WriteScopeConfig(sc, ".cantwrite")
	// assert.Error(t, err, "operational not permitted")
	// os.Remove(".cantwrite")

	testYaml := testDefaultScopeConfigYaml("/foo", 4)
	err = opt.WriteScopeConfig(".scope.yml")
	assert.NoError(t, err)
	yamlBytes, err := ioutil.ReadFile(".scope.yml")
	assert.Equal(t, testYaml, string(yamlBytes))
	os.Remove(".scope.yml")
}

func TestScopeLogRegex(t *testing.T) {
	tests := []struct {
		Test    string
		Matches bool
	}{
		{"/var/log/messages", true},
		{"/opt/cribl/log/cribl.log", true},
		{"/some/container/path.log", true},
		{"/opt/cribl/blog/foo.txt", false},
		{"/opt/cribl/log/foo.txt", true},
	}

	re := regexp.MustCompile(scopeLogRegex())
	for _, test := range tests {
		assert.Equal(t, test.Matches, re.MatchString(test.Test))
	}
}
