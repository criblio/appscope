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
//     level: warning
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
	assert.Equal(t, "true", c.sc.Payload.Enable)
	assert.Equal(t, ".foo/payloads", c.sc.Payload.Dir)

	c.MetricsFormat = "foo"
	err = c.configFromRunOpts()
	assert.Error(t, err)

	c.MetricsFormat = "statsd"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "statsd", c.sc.Metric.Format.FormatType)

	// empty
	c.MetricsDest = ""
	err = c.configFromRunOpts()
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, ".foo/metrics.json", c.sc.Metric.Transport.Path)

	// invalid proto
	c.MetricsDest = "foo://bar"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Invalid dest: foo://bar"), err)

	// missing port
	c.MetricsDest = "tcp://bar"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Missing :port at the end of tcp://bar"), err)
	c.MetricsDest = "udp://bar"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Missing :port at the end of udp://bar"), err)
	c.MetricsDest = "tls://bar"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Missing :port at the end of tls://bar"), err)

	// invalid port
	c.MetricsDest = "tcp://bar:foo"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Invalid dest: tcp://bar:foo"), err)
	c.MetricsDest = "udp://bar:foo"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Invalid dest: udp://bar:foo"), err)
	c.MetricsDest = "tls://bar:foo"
	err = c.configFromRunOpts()
	assert.Equal(t, fmt.Errorf("Invalid dest: tls://bar:foo"), err)

	// file
	c.MetricsDest = "file://foo"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Path)
	c.MetricsDest = "foo"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Path)
	c.MetricsDest = "file:///foo/bar"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "/foo/bar", c.sc.Metric.Transport.Path)
	c.MetricsDest = "/foo/bar"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "/foo/bar", c.sc.Metric.Transport.Path)
	c.MetricsDest = "file://foo:1234" // port is ignored
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Path)
	c.MetricsDest = "file://stdout" // stdout is valid
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "stdout", c.sc.Metric.Transport.Path)
	c.MetricsDest = "file://stderr" // stderr is valid
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "file", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "stderr", c.sc.Metric.Transport.Path)

	// tcp
	c.MetricsDest = "tcp://foo:1234"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "tcp", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Host)
	assert.Equal(t, "1234", c.sc.Metric.Transport.Port)
	assert.Equal(t, "", c.sc.Metric.Transport.Path)
	assert.Equal(t, "false", c.sc.Metric.Transport.Tls.Enable)

	// udp
	c.MetricsDest = "udp://foo:1234"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "udp", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Host)
	assert.Equal(t, "1234", c.sc.Metric.Transport.Port)
	assert.Equal(t, "", c.sc.Metric.Transport.Path)
	assert.Equal(t, "false", c.sc.Metric.Transport.Tls.Enable)

	// tls
	c.MetricsDest = "tls://foo:1234"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "tcp", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Host)
	assert.Equal(t, "1234", c.sc.Metric.Transport.Port)
	assert.Equal(t, "", c.sc.Metric.Transport.Path)
	assert.Equal(t, "true", c.sc.Metric.Transport.Tls.Enable)
	assert.Equal(t, "true", c.sc.Metric.Transport.Tls.ValidateServer)
	assert.Equal(t, "", c.sc.Metric.Transport.Tls.CaCertPath)

	// host:port
	c.MetricsDest = "foo:1234"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, "tcp", c.sc.Metric.Transport.TransportType)
	assert.Equal(t, "foo", c.sc.Metric.Transport.Host)
	assert.Equal(t, "1234", c.sc.Metric.Transport.Port)
	assert.Equal(t, "", c.sc.Metric.Transport.Path)
	assert.Equal(t, true, c.sc.Metric.Transport.Tls.Enable)
	assert.Equal(t, true, c.sc.Metric.Transport.Tls.ValidateServer)
	assert.Equal(t, "", c.sc.Metric.Transport.Tls.CaCertPath)

	c.Loglevel = "foo"
	err = c.configFromRunOpts()
	assert.Error(t, err)

	c.Loglevel = "debug"
	err = c.configFromRunOpts()
	assert.NoError(t, err)
	assert.Equal(t, c.sc.Libscope.Log.Level, "debug")
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
	err = opt.WriteScopeConfig(".scope.yml", 0644)
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
		{"/app/logs/stdout", true},
		{"/app/logs/stderr", true},
		{"/opt/cribl/log/foo.txt", true},
		{"/opt/cribl/log/cribl.log", true},
		{"/some/container/path.log", true},
		{"/some/container/path.log1", true},
		{"/some/container/path.log42", true},
		{"/some/container/path.log.1", true},
		{"/some/container/path.log.2", true},
		{"/some/container/path.log.fancy", true},
		// negative tests
		{"/opt/cribl/blog/foo.txt", false},
		{"/opt/cribl/log420/foo.txt", false},
		{"/opt/cribl/local/logger.yml", false},
		{"/opt/cribl/local/file.logger", false},
		{"/opt/cribl/local/file.420", false},
	}

	re := regexp.MustCompile(scopeLogRegex())
	for _, test := range tests {
		assert.Equal(t, test.Matches, re.MatchString(test.Test))
	}
}
