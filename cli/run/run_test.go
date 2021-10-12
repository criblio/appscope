package run

import (
	"bytes"
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

func TestMain(m *testing.M) {
	// Borrowed from http://cs-guy.com/blog/2015/01/test-main/
	switch os.Getenv("TEST_MAIN") {
	case "runpassthrough":
		internal.InitConfig()
		rc := Config{Passthrough: true}
		rc.Run([]string{"/bin/echo", "true"})
	case "createWorkDir":
		internal.InitConfig()
		rc := Config{}
		rc.createWorkDir([]string{"foo"}, false)
	case "run":
		internal.InitConfig()
		rc := Config{}
		rc.Run([]string{"/bin/echo", "true"})
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
	cmd.Env = append(os.Environ(), "TEST_MAIN=runpassthrough", "SCOPE_HOME=.test", "SCOPE_METRIC_DEST=file://stderr")
	var outb, errb bytes.Buffer
	cmd.Stdout = &outb
	cmd.Stderr = &errb
	err := cmd.Run()
	assert.NoError(t, err)
	assert.Equal(t, "true\n", outb.String())
	assert.Contains(t, errb.String(), "proc.start")
	os.RemoveAll(".test")
}

func TestRun(t *testing.T) {
	// Test SetupWorkDir, successfully
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err := cmd.Run()
	assert.NoError(t, err)
	files, err := ioutil.ReadDir(".test/history")
	matched := false
	wdbase := fmt.Sprintf("%s_%d_%d", "echo", 1, cmd.Process.Pid)
	var wd string
	for _, f := range files {
		if strings.HasPrefix(f.Name(), wdbase) {
			matched = true
			wd = fmt.Sprintf("%s%s", ".test/history/", f.Name())
			break
		}
	}
	assert.True(t, matched)

	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	argsJSONBytes, err := ioutil.ReadFile(filepath.Join(wd, "args.json"))
	assert.NoError(t, err)
	assert.Equal(t, `["/bin/echo","true"]`, string(argsJSONBytes))

	expectedYaml := testDefaultScopeConfigYaml(wd, 4)

	scopeYAMLBytes, err := ioutil.ReadFile(filepath.Join(wd, "scope.yml"))
	assert.NoError(t, err)
	assert.Equal(t, expectedYaml, string(scopeYAMLBytes))

	cmdDirExists := util.CheckFileExists(filepath.Join(wd, "cmd"))
	assert.True(t, cmdDirExists)
	os.RemoveAll(".test")
}

func TestSubprocess(t *testing.T) {
	os.Setenv("SCOPE_HOME", ".test")
	// Test SetupWorkDir, successfully
	internal.InitConfig()
	rc := Config{Subprocess: true}
	rc.Run([]string{"/bin/echo", "true"})
	files, _ := ioutil.ReadDir(".test/history")
	matched := false
	wdbase := fmt.Sprintf("%s_%d", "echo", 1)
	var wd string
	for _, f := range files {
		if strings.HasPrefix(f.Name(), wdbase) {
			matched = true
			wd = fmt.Sprintf("%s%s", ".test/history/", f.Name())
			break
		}
	}
	assert.True(t, matched)

	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	argsJSONBytes, err := ioutil.ReadFile(filepath.Join(wd, "args.json"))
	assert.NoError(t, err)
	assert.Equal(t, `["/bin/echo","true"]`, string(argsJSONBytes))

	expectedYaml := testDefaultScopeConfigYaml(wd, 4)

	scopeYAMLBytes, err := ioutil.ReadFile(filepath.Join(wd, "scope.yml"))
	assert.NoError(t, err)
	assert.Equal(t, expectedYaml, string(scopeYAMLBytes))

	cmdDirExists := util.CheckFileExists(filepath.Join(wd, "cmd"))
	assert.True(t, cmdDirExists)
	os.RemoveAll(".test")
}
