package run

import (
	"io/ioutil"
	"os"
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
func TestGetDefaultScopeConfig(t *testing.T) {
	sc := GetDefaultScopeConfig("/foo")

	testYaml := testDefaultScopeConfigYaml("/foo")

	configYaml, err := yaml.Marshal(sc)
	assert.NoError(t, err)
	assert.Equal(t, testYaml, string(configYaml))
}

func TestWriteScopeConfig(t *testing.T) {
	sc := GetDefaultScopeConfig("/foo")
	f, err := os.OpenFile(".cantwrite", os.O_RDONLY|os.O_CREATE, 0400)
	assert.NoError(t, err)
	f.Close()
	err = WriteScopeConfig(sc, ".cantwrite")
	assert.Error(t, err, "operational not permitted")
	os.Remove(".cantwrite")

	testYaml := testDefaultScopeConfigYaml("/foo")
	err = WriteScopeConfig(sc, ".scope.yml")
	assert.NoError(t, err)
	yamlBytes, err := ioutil.ReadFile(".scope.yml")
	assert.Equal(t, testYaml, string(yamlBytes))
	os.Remove(".scope.yml")
}
