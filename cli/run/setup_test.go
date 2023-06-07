package run

import (
	"crypto/md5"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/criblio/scope/util"
	"github.com/stretchr/testify/assert"
)

func TestCreateAll(t *testing.T) {
	os.MkdirAll(".foo", 0755)
	CreateAll(".foo")
	files := []string{"libscope.so", "scope.yml"}
	perms := []os.FileMode{0755, 0644}
	for i, f := range files {
		path := fmt.Sprintf(".foo/%s", f)
		stat, _ := os.Stat(path)
		assert.Equal(t, stat.Mode(), perms[i])
		wb, _ := Asset(fmt.Sprintf("build/%s", f))
		hash1 := md5.Sum(wb)
		fb, _ := os.ReadFile(path)
		hash2 := md5.Sum(fb)
		assert.Equal(t, hash1, hash2)

		fb, _ = os.ReadFile(fmt.Sprintf("../build/%s", f))
		hash3 := md5.Sum(fb)
		assert.Equal(t, hash2, hash3)
	}
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

/* Todo this should be an integration test
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
	var outb, errb bytes.Buffer
	cmd.Stdout = &outb
	cmd.Stderr = &errb
	err = cmd.Run()
	assert.NoError(t, err, errb.String())
	files, err := ioutil.ReadDir(".test/history")
	matched := false
	wdbase := fmt.Sprintf("%s_%d_%d", "foo", 1, cmd.Process.Pid)
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
	os.RemoveAll(".test")
}
*/

func testDefaultScopeConfigYaml(wd string, verbosity int) string {
	wd, _ = filepath.Abs(wd)
	expectedYaml := `cribl:
  enable: false
  transport:
    type: ""
    tls:
      enable: false
      validateserver: false
      cacertpath: ""
  authtoken: ""
metric:
  enable: true
  format:
    type: ndjson
    verbosity: VERBOSITY
  transport:
    type: file
    path: METRICSPATH
    buffering: line
    tls:
      enable: false
      validateserver: false
      cacertpath: ""
  watch:
  - type: fs
  - type: net
  - type: http
  - type: dns
  - type: process
  - type: statsd
event:
  enable: true
  format:
    type: ndjson
  transport:
    type: file
    path: EVENTSPATH
    buffering: line
    tls:
      enable: false
      validateserver: false
      cacertpath: ""
  watch:
  - type: file
    name: (\/logs?\/)|(\.log$)|(\.log[.\d])
    value: .*
  - type: console
    name: (stdout|stderr)
    value: .*
    allowbinary: false
  - type: net
    name: .*
    field: .*
    value: .*
  - type: fs
    name: .*
    field: .*
    value: .*
  - type: dns
    name: .*
    field: .*
    value: .*
  - type: http
    name: .*
    field: .*
    value: .*
libscope:
  configevent: true
  summaryperiod: 10
  commanddir: CMDDIR
  log:
    level: warning
    transport:
      type: file
      path: LIBSCOPELOGPATH
      buffering: line
      tls:
        enable: false
        validateserver: false
        cacertpath: ""
  snapshot:
    coredump: false
    backtrace: false
`

	expectedYaml = strings.Replace(expectedYaml, "VERBOSITY", strconv.Itoa(verbosity), 1)
	expectedYaml = strings.Replace(expectedYaml, "METRICSPATH", filepath.Join(wd, "metrics.json"), 1)
	expectedYaml = strings.Replace(expectedYaml, "EVENTSPATH", filepath.Join(wd, "events.json"), 1)
	expectedYaml = strings.Replace(expectedYaml, "CMDDIR", filepath.Join(wd, "cmd"), 1)
	expectedYaml = strings.Replace(expectedYaml, "LIBSCOPELOGPATH", filepath.Join(wd, "libscope.log"), 1)
	return expectedYaml
}

func TestSetupWorkDir(t *testing.T) {
	os.Setenv("SCOPE_HOME", ".foo")
	os.Setenv("SCOPE_TEST", "true")
	rc := Config{}
	rc.now = func() time.Time { return time.Unix(0, 0) }
	rc.setupWorkDir([]string{"/bin/foo"}, false)
	wd := fmt.Sprintf("%s_%d_%d_%d", ".foo/history/foo", 1, os.Getpid(), 0)
	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	argsJSONBytes, err := os.ReadFile(filepath.Join(wd, "args.json"))
	assert.NoError(t, err)
	assert.Equal(t, `["/bin/foo"]`, string(argsJSONBytes))

	expectedYaml := testDefaultScopeConfigYaml(wd, 4)

	scopeYAMLBytes, err := os.ReadFile(filepath.Join(wd, "scope.yml"))
	assert.NoError(t, err)
	assert.Equal(t, expectedYaml, string(scopeYAMLBytes))

	cmdDirExists := util.CheckFileExists(filepath.Join(wd, "cmd"))
	assert.True(t, cmdDirExists)

	payloadsDirExists := util.CheckFileExists(filepath.Join(wd, "payloads"))
	assert.True(t, payloadsDirExists)
	os.RemoveAll(".foo")
	os.Unsetenv("SCOPE_TEST")
	os.Unsetenv("SCOPE_HOME")
}

func TestSetupWorkDirAttach(t *testing.T) {
	os.Setenv("SCOPE_HOME", ".foo")
	os.Setenv("SCOPE_TEST", "true")
	rc := Config{}
	rc.now = func() time.Time { return time.Unix(0, 0) }
	cmd := exec.Command("sleep", "600")
	err := cmd.Start()
	assert.Nil(t, err)
	defer cmd.Process.Kill()
	pid := cmd.Process.Pid
	rc.setupWorkDir([]string{strconv.Itoa(pid)}, true)
	wd := fmt.Sprintf("%s_%d_%d_%d", ".foo/history/sleep", 1, pid, 0)
	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	argsJSONBytes, err := os.ReadFile(filepath.Join(wd, "args.json"))
	assert.NoError(t, err)
	assert.Equal(t, `["sleep","600"]`, string(argsJSONBytes))

	expectedYaml := testDefaultScopeConfigYaml(wd, 4)

	scopeYAMLBytes, err := os.ReadFile(filepath.Join(wd, "scope.yml"))
	assert.NoError(t, err)
	assert.Equal(t, expectedYaml, string(scopeYAMLBytes))

	cmdDirExists := util.CheckFileExists(filepath.Join(wd, "cmd"))
	assert.True(t, cmdDirExists)

	payloadsDirExists := util.CheckFileExists(filepath.Join(wd, "payloads"))
	assert.True(t, payloadsDirExists)
	os.RemoveAll(wd)
	os.RemoveAll(".foo")
	os.Unsetenv("SCOPE_TEST")
	os.Unsetenv("SCOPE_HOME")
}
