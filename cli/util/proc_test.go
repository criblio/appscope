package util

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
 * This test will not pass on the github runner due to permissions on /proc
 * This is now covered in an integration test instead
 *
// TestProcessesByName
// Assertions:
// - The expected process array is returned
func TestProcessesByName(t *testing.T) {
	// Current process
	name := "util.test"
	result := ProcessesByName(name)
	user, _ := user.Current()
	exp := Processes{
		Process{
			ID:      1,
			Pid:     os.Getpid(),
			User:    user.Username,
			Command: "util.test",
			Scoped:  false,
		},
	}
	assert.Equal(t, exp, result)
}
*/

/*
 * This test will not pass on the github runner due to permissions on /proc
 * This is now covered in an integration test instead
 *
// TestPidUser
// Assertions:
// - The expected user value is returned
func TestPidUser(t *testing.T) {
	// Process 1 owner
	pid := 1
	result := PidUser(pid)
	assert.Equal(t, "root", result)

	// Current process owner
	currentUser, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}
	username := currentUser.Username
	pid = os.Getpid()
	result = PidUser(pid)
	assert.Equal(t, username, result)
}
*/

// TestPidCommand
// Assertions:
// - The expected command value is returned
func TestPidCommand(t *testing.T) {
	// Current process command
	pid := os.Getpid()
	result := PidCommand(pid)
	assert.Equal(t, "util.test", result)
}

// TestPidExists
// Assertions:
// - The expected boolean value is returned
func TestPidExists(t *testing.T) {
	// Current process PID
	pid := os.Getpid()
	result := PidExists(pid)
	assert.Equal(t, true, result)

	// PID 0
	pid = 0
	result = PidExists(pid)
	assert.Equal(t, false, result)
}
