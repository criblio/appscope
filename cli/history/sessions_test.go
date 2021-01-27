package history

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/stretchr/testify/assert"
)

func TestMain(m *testing.M) {
	// Borrowed from http://cs-guy.com/blog/2015/01/test-main/
	switch os.Getenv("TEST_MAIN") {
	case "run":
		internal.InitConfig()
		rc := run.Config{}
		rc.Run([]string{"/bin/echo", "true"})
	case "slow":
		internal.InitConfig()
		rc := run.Config{}
		rc.Run([]string{"/bin/sleep", "10"})
	default:
		os.Exit(m.Run())
	}
}

func TestGetSessions(t *testing.T) {
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err := cmd.Run()
	assert.NoError(t, err)
	files, err := ioutil.ReadDir(".test/history")
	matched := false
	wdbase := fmt.Sprintf("%s_%d_%d", "echo", 1, cmd.Process.Pid)
	var wd string
	var ts int
	for _, f := range files {
		if strings.HasPrefix(f.Name(), wdbase) {
			matched = true
			pattern := fmt.Sprintf("%s_(\\d+)", wdbase)
			r := regexp.MustCompile(pattern)
			matchTS := r.FindStringSubmatch(f.Name())
			ts, _ = strconv.Atoi(matchTS[1])
			wd = fmt.Sprintf("%s%s", ".test/history/", f.Name())
			break
		}
	}
	assert.True(t, matched)
	exists := util.CheckFileExists(wd)
	assert.True(t, exists)

	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 1)
	assert.Equal(t, "echo", sessions[0].Cmd)
	assert.Equal(t, 1, sessions[0].ID)
	assert.Equal(t, cmd.Process.Pid, sessions[0].Pid)
	assert.Equal(t, int64(ts), sessions[0].Timestamp)

	// assert.True(t, util.CheckFileExists(".test"))
	os.RemoveAll(".test")
}

func runThree(t *testing.T) []int {
	pids := []int{}
	for i := 0; i < 3; i++ {
		cmd := exec.Command(os.Args[0])
		cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test")
		err := cmd.Run()
		assert.NoError(t, err)
		pids = append(pids, cmd.Process.Pid)
	}
	return pids
}

func TestGetSessionLast(t *testing.T) {
	pids := runThree(t)

	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 3)

	l1 := sessions.Last(1)
	assert.Equal(t, "echo", l1[0].Cmd)
	assert.Equal(t, 3, l1[0].ID)
	assert.Equal(t, pids[2], l1[0].Pid)

	l2 := sessions.Last(2)
	assert.Greater(t, l2[1].Timestamp, l2[0].Timestamp)

	// assert.True(t, util.CheckFileExists(".test"))
	os.RemoveAll(".test")
}

func TestGetSessionFirst(t *testing.T) {
	pids := runThree(t)

	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 3)

	l1 := sessions.First(1)
	assert.Equal(t, "echo", l1[0].Cmd)
	assert.Equal(t, 1, l1[0].ID)
	assert.Equal(t, pids[0], l1[0].Pid)

	l2 := sessions.First(2)
	assert.Greater(t, l2[1].Timestamp, l2[0].Timestamp)

	// assert.True(t, util.CheckFileExists(".test"))
	os.RemoveAll(".test")
}

func TestSessionRemove(t *testing.T) {
	_ = runThree(t)
	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 3)

	sessionsBak := append(SessionList{}, sessions...)
	for _, s := range sessions {
		assert.True(t, util.CheckFileExists(s.WorkDir))
	}
	sessions.Remove()
	for _, s := range sessionsBak {
		assert.False(t, util.CheckFileExists(s.WorkDir))
	}

	// assert.True(t, util.CheckFileExists(".test"))
	os.RemoveAll(".test")
}

func TestGetSessionRun(t *testing.T) {
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test")
	err := cmd.Run()
	assert.NoError(t, err)

	cmd = exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=slow", "SCOPE_HOME=.test")
	err = cmd.Start()
	assert.NoError(t, err)

	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 2)

	r := sessions.Running()
	assert.Len(t, r, 1)

	os.RemoveAll(".test")
}

func TestSessionArgs(t *testing.T) {
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test")
	err := cmd.Run()
	assert.NoError(t, err)

	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 1)

	r := sessions.Args()
	assert.Len(t, r, 1)
	assert.Equal(t, []string{"/bin/echo", "true"}, r[0].Args)

	os.RemoveAll(".test")
}
