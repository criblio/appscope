package util

import (
	"bytes"
)

// ScopeStatus represents the process status in context of libscope.so
type ScopeStatus int

// Scope Status description
//
// When process started it will be in one of the following states:
// Disable
// Active
// Setup
//
// Possible transfer between states:
// Disable -> Active (attach required sudo - ptrace)
// Setup -> Active   (attach required sudo - ptrace)
// Latent -> Active
// Active -> Latent
// Setup -> Latent
//
// If the process is in Active status it can be changed only to Latent status

const (
	Disable ScopeStatus = iota // libscope.so is not loaded
	Setup                      // libscope.so is loaded, reporting thread is not present
	Active                     // libscope.so is loaded, reporting thread is present we are sending data
	Latent                     // libscope.so is loaded, reporting thread is present we are not emiting any data
)

func (state ScopeStatus) String() string {
	return []string{"Disable", "Setup", "Active", "Latent"}[state]
}

// getScopeStatus retreive sinformation about Scope status from IPC
func getScopeStatus(pid int) ScopeStatus {
	bresp, err := ipcGetScopeStatus(pid)
	if err != nil {
		return Setup
	}
	if bytes.Equal(bresp, []byte("true")) {
		return Active
	} else if bytes.Equal(bresp, []byte("false")) {
		return Latent
	}
	return Disable
}
