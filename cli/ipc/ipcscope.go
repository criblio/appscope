package ipc

import (
	"encoding/json"
	"fmt"

	"github.com/criblio/scope/libscope"
	"gopkg.in/yaml.v2"
)

type IpcResponseCtx struct {
	ResponseScopeMsgData []byte     // Contains scope message response (status + message response data)
	MetaMsgStatus        respStatus // Response status in msg metadata
}

// scopeReqCmd describes the command passed in the IPC request to library scope request
// Must be inline with server, see: ipc_scope_req_t

type scopeReqCmd int

const (
	reqCmdGetScopeStatus scopeReqCmd = iota
	reqCmdGetScopeCfg
	reqCmdSetScopeCfg
)

// Request which contains only cmd without data
type scopeRequestOnly struct {
	// Unique request command
	Req scopeReqCmd `mapstructure:"req" json:"req" yaml:"req"`
}

// Set configuration request
// Must be inline with server, see: ipcProcessSetCfg
type scopeRequestSetCfg struct {
	// Unique request command
	Req scopeReqCmd `mapstructure:"req" json:"req" yaml:"req"`
	// Scoped Cfg
	Cfg libscope.ScopeConfig `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
}

// Must be inline with server, see: ipcRespStatus
type scopeOnlyStatusResponse struct {
	// Response status
	Status *respStatus `mapstructure:"status" json:"status" yaml:"status"`
}

// Must be inline with server, see: ipcRespGetScopeStatus
type scopeGetStatusResponse struct {
	// Response status
	Status *respStatus `mapstructure:"status" json:"status" yaml:"status"`
	// Scoped status
	Scoped bool `mapstructure:"scoped" json:"scoped" yaml:"scoped"`
}

// Must be inline with server, see: ipcRespGetScopeCfg
type scopeGetCfgResponse struct {
	// Response status
	Status *respStatus `mapstructure:"status" json:"status" yaml:"status"`
	Cfg    struct {
		// Scoped Cfg
		Current libscope.ScopeConfig `mapstructure:"current" json:"current" yaml:"current"`
	} `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
}

// CmdGetScopeStatus describes Get Scope Status command request and response
type CmdGetScopeStatus struct {
	Response scopeGetStatusResponse
}

func (cmd *CmdGetScopeStatus) Request(pidCtx IpcPidCtx) (*IpcResponseCtx, error) {
	req, _ := json.Marshal(scopeRequestOnly{Req: reqCmdGetScopeStatus})

	return ipcDispatcher(req, pidCtx)
}

func (cmd *CmdGetScopeStatus) UnmarshalResp(respData []byte) error {
	err := yaml.Unmarshal(respData, &cmd.Response)
	if err != nil {
		return err
	}

	if cmd.Response.Status == nil {
		return fmt.Errorf("%w %v", errMissingMandatoryField, "status")
	}

	return nil
}

// CmdGetScopeCfg describes Get Scope Configuration command request and response
type CmdGetScopeCfg struct {
	Response scopeGetCfgResponse
}

func (cmd *CmdGetScopeCfg) Request(pidCtx IpcPidCtx) (*IpcResponseCtx, error) {
	req, _ := json.Marshal(scopeRequestOnly{Req: reqCmdGetScopeCfg})

	return ipcDispatcher(req, pidCtx)
}

func (cmd *CmdGetScopeCfg) UnmarshalResp(respData []byte) error {
	err := yaml.Unmarshal(respData, &cmd.Response)
	if err != nil {
		return err
	}

	if cmd.Response.Status == nil {
		return fmt.Errorf("%w %v", errMissingMandatoryField, "status")
	}

	return nil
}

// CmdSetScopeCfg describes Set Scope Configuration command request and response
type CmdSetScopeCfg struct {
	Response scopeOnlyStatusResponse
	CfgData  []byte
}

func (cmd *CmdSetScopeCfg) Request(pidCtx IpcPidCtx) (*IpcResponseCtx, error) {
	scopeReq := scopeRequestSetCfg{Req: reqCmdSetScopeCfg}

	err := yaml.Unmarshal(cmd.CfgData, &scopeReq.Cfg)
	if err != nil {
		return nil, err
	}
	req, _ := json.Marshal(scopeReq)

	return ipcDispatcher(req, pidCtx)
}

func (cmd *CmdSetScopeCfg) UnmarshalResp(respData []byte) error {
	err := yaml.Unmarshal(respData, &cmd.Response)
	if err != nil {
		return err
	}

	if cmd.Response.Status == nil {
		return fmt.Errorf("%w %v", errMissingMandatoryField, "status")
	}

	return nil
}
