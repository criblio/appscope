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
	reqCmdGetTransportStatus
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

type ScopeGetCfgResponseCfg struct {
	Current libscope.ScopeConfig `mapstructure:"current" json:"current" yaml:"current"`
}

// Must be inline with server, see: ipcRespGetScopeCfg
type scopeGetCfgResponse struct {
	// Response status
	Status *respStatus            `mapstructure:"status" json:"status" yaml:"status"`
	Cfg    ScopeGetCfgResponseCfg `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
}

// CmdGetScopeStatus describes Get Scope Status command request and response
type CmdGetScopeStatus struct {
	Response scopeGetStatusResponse
}

// ChannelDesc describes
type ChannelDesc []struct {
	// Name of channel
	Name string `mapstructure:"name" json:"name" yaml:"name"`
}

// Must be inline with server, see: ipcRespGetTransportStatus
type scopeGetTransportStatusResponse struct {
	// Response status
	Status *respStatus `mapstructure:"status" json:"status" yaml:"status"`
	// Channel description
	Channels ChannelDesc `mapstructure:"channels" json:"channels" yaml:"channels"`
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

// CmdGetTransportStatus describes Get Scope Status command request and response
type CmdGetTransportStatus struct {
	Response scopeGetTransportStatusResponse
}

func (cmd *CmdGetTransportStatus) Request(pidCtx IpcPidCtx) (*IpcResponseCtx, error) {
	req, _ := json.Marshal(scopeRequestOnly{Req: reqCmdGetTransportStatus})

	return ipcDispatcher(req, pidCtx)
}

func (cmd *CmdGetTransportStatus) UnmarshalResp(respData []byte) error {
	err := yaml.Unmarshal(respData, &cmd.Response)
	if err != nil {
		return err
	}

	if cmd.Response.Status == nil {
		return fmt.Errorf("%w %v", errMissingMandatoryField, "status")
	}

	return nil
}
