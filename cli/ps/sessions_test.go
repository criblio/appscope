package ps

import (
	"fmt"
	"os"
	"os/exec"
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
		internal.InitConfig("")
		run.Run([]string{"/bin/echo", "true"})
	case "slow":
		internal.InitConfig("")
		run.Run([]string{"/bin/sleep", "10"})
	default:
		os.Exit(m.Run())
	}
}

func TestGetSessions(t *testing.T) {
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test", "SCOPE_TEST=true")
	err := cmd.Run()
	assert.NoError(t, err)
	wd := fmt.Sprintf("%s_%d_%d_%d", ".test/history/echo", 1, cmd.Process.Pid, 0)
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
	assert.Equal(t, int64(0), sessions[0].Timestamp)

	// assert.True(t, util.CheckFileExists(".test"))
	os.RemoveAll(".test")
}

func TestGetSessionLast(t *testing.T) {
	pid := 0
	for i := 0; i < 3; i++ {
		cmd := exec.Command(os.Args[0])
		cmd.Env = append(os.Environ(), "TEST_MAIN=run", "SCOPE_HOME=.test")
		err := cmd.Run()
		assert.NoError(t, err)
		pid = cmd.Process.Pid
	}

	time.Sleep(100 * time.Millisecond)

	lastHome := os.Getenv("SCOPE_HOME")
	os.Setenv("SCOPE_HOME", ".test")
	sessions := GetSessions()
	os.Setenv("SCOPE_HOME", lastHome)
	assert.Len(t, sessions, 3)

	l1 := sessions.Last(1)
	assert.Equal(t, "echo", l1[0].Cmd)
	assert.Equal(t, 3, l1[0].ID)
	assert.Equal(t, pid, l1[0].Pid)

	l2 := sessions.Last(2)
	assert.Greater(t, l2[0].Timestamp, l2[1].Timestamp)

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
