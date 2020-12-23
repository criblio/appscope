package run

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"

	"github.com/stretchr/testify/assert"
)

func TestCreateCScope(t *testing.T) {
	tmpCScope := filepath.Join(".foo", "cscope")
	os.Setenv("SCOPE_HOME", ".foo")
	err := CreateCScope()
	os.Unsetenv("SCOPE_HOME")
	assert.NoError(t, err)

	wb, _ := buildCscopeBytes()
	hash1 := md5.Sum(wb)
	fb, _ := ioutil.ReadFile(tmpCScope)
	hash2 := md5.Sum(fb)
	assert.Equal(t, hash1, hash2)

	fb, _ = ioutil.ReadFile("../build/cscope")
	hash3 := md5.Sum(fb)
	assert.Equal(t, hash2, hash3)

	// Call again, should use cached copy
	err = CreateCScope()
	assert.NoError(t, err)

	os.RemoveAll(".foo")

}

func TestEnvironNoScope(t *testing.T) {
	hasScope := func(testArr []string) bool {
		for _, s := range testArr {
			if strings.HasPrefix(s, "SCOPE") {
				return true
			}
		}
		return false
	}
	assert.False(t, hasScope(os.Environ()))
	os.Setenv("SCOPE_FOO", "true")
	assert.True(t, hasScope(os.Environ()))
	assert.False(t, hasScope(environNoScope()))
	os.Unsetenv("SCOPE_FOO")
	assert.False(t, hasScope(os.Environ()))
}

func TestMain(m *testing.M) {
	// Borrowed from http://cs-guy.com/blog/2015/01/test-main/
	switch os.Getenv("TEST_MAIN") {
	case "runpassthrough":
		internal.InitConfig("")
		Run([]string{"/bin/echo", "true"})
	case "createWorkDir":
		internal.InitConfig("")
		CreateWorkDir("foo")
	case "setupWorkDir":
		internal.InitConfig("")
		SetupWorkDir([]string{"/bin/foo"})
	case "run":
		internal.InitConfig("")
		Run([]string{"/bin/echo", "true"})
	default:
		os.Exit(m.Run())
	}
}

// Simplest test, runs passthrough and outputs to stderr
// Passthrough sends environment directly to scope, rather
// than clearing it and creating configuration files
func TestRunPassthrough(t *testing.T) {
	// Test Passthrough, read from Stderr
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=runpassthrough", "SCOPE_PASSTHROUGH=true", "SCOPE_METRIC_DEST=file://stderr")
	var outb, errb bytes.Buffer
	cmd.Stdout = &outb
	cmd.Stderr = &errb
	err := cmd.Run()
	assert.NoError(t, err)
	assert.Equal(t, "true\n", outb.String())
	assert.Contains(t, errb.String(), "proc.start")
}

func TestCreateWorkDir(t *testing.T) {
	// Test CreateWorkDir, fail first
	f, err := os.OpenFile(".test", os.O_RDONLY|os.O_CREATE, 0400)
	assert.NoError(t, err)
	f.Close()
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=createWorkDir", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err = cmd.Run()
	assert.Error(t, err, "exit status 1")
	os.Remove(".test")

	// Test CreateWorkDir, successfully
	cmd = exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=createWorkDir", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err = cmd.Run()
	wd := fmt.Sprintf("%s_%d_%d", ".test/history/foo", cmd.Process.Pid, 0)
	exists := util.CheckFileExists(wd)
	assert.True(t, exists)
	os.RemoveAll(".test")
}

func testDefaultScopeConfigYaml(wd string) string {
	expectedYaml := `metric:
  enable: true
  format:
    type: ndjson
    verbosity: 4
  transport:
    type: file
    path: METRICSPATH
event:
  enable: true
  format:
    type: ndjson
  watch: []
libscope:
  level: ""
  transport:
    type: file
    path: EVENTSPATH
  summaryperiod: 10
  commanddir: CMDDIR
  log:
    level: error
    transport:
      type: file
      path: CSCOPELOGPATH
      buffering: line
`
	expectedYaml = strings.Replace(expectedYaml, "METRICSPATH", filepath.Join(wd, "metrics.json"), 1)
	expectedYaml = strings.Replace(expectedYaml, "EVENTSPATH", filepath.Join(wd, "events.json"), 1)
	expectedYaml = strings.Replace(expectedYaml, "CMDDIR", filepath.Join(wd, "cmd"), 1)
	expectedYaml = strings.Replace(expectedYaml, "CSCOPELOGPATH", filepath.Join(wd, "cscope.log"), 1)
	return expectedYaml
}

func TestSetupWorkDir(t *testing.T) {
	// Test SetupWorkDir, successfully
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=setupWorkDir", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err := cmd.Run()
	assert.NoError(t, err)
	wd := fmt.Sprintf("%s_%d_%d", ".test/history/foo", cmd.Process.Pid, 0)
	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	argsJSONBytes, err := ioutil.ReadFile(filepath.Join(wd, "args.json"))
	assert.NoError(t, err)
	assert.Equal(t, `["/bin/foo"]`, string(argsJSONBytes))

	expectedYaml := testDefaultScopeConfigYaml(wd)

	scopeYAMLBytes, err := ioutil.ReadFile(filepath.Join(wd, "scope.yml"))
	assert.NoError(t, err)
	assert.Equal(t, expectedYaml, string(scopeYAMLBytes))

	cmdDirExists := util.CheckFileExists(filepath.Join(wd, "cmd"))
	assert.True(t, cmdDirExists)
	os.RemoveAll(".test")
}

func TestRun(t *testing.T) {
	// Test SetupWorkDir, successfully
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err := cmd.Run()
	assert.NoError(t, err)
	wd := fmt.Sprintf("%s_%d_%d", ".test/history/echo", cmd.Process.Pid, 0)
	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	argsJSONBytes, err := ioutil.ReadFile(filepath.Join(wd, "args.json"))
	assert.NoError(t, err)
	assert.Equal(t, `["/bin/echo","true"]`, string(argsJSONBytes))

	expectedYaml := testDefaultScopeConfigYaml(wd)

	scopeYAMLBytes, err := ioutil.ReadFile(filepath.Join(wd, "scope.yml"))
	assert.NoError(t, err)
	assert.Equal(t, expectedYaml, string(scopeYAMLBytes))

	cmdDirExists := util.CheckFileExists(filepath.Join(wd, "cmd"))
	assert.True(t, cmdDirExists)
	// os.RemoveAll(".test")
}
