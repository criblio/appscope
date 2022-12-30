package util

import (
	"github.com/criblio/scope/ipc"
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

// getScopeStatus retreives information about Scope status using IPC
func getScopeStatus(pid int) ScopeStatus {
	cmd := ipc.CmdGetScopeStatus{}
	resp, err := cmd.Request(pid)
	if err != nil {
		return Disable
	}
	err = cmd.UnmarshalResp(resp.ResponseScopeMsgData)
	if err != nil {
		return Disable
	}

	if cmd.Response.Status == ipc.ResponseOK {
		if cmd.Response.Scoped {
			return Active
		}
		return Latent
	}
	return Disable
}
