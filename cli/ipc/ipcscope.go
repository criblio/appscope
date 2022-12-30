package ipc

import (
	"encoding/json"

	"gopkg.in/yaml.v2"
)

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
	// Configuration
	Cfg struct {
		Metric struct {
			Enable bool `json:"enable"`
			Format struct {
				Type         string      `json:"type"`
				Statsdprefix interface{} `json:"statsdprefix"`
				Statsdmaxlen int         `json:"statsdmaxlen"`
				Verbosity    int         `json:"verbosity"`
			} `json:"format"`
			Watch []struct {
				Type string `json:"type"`
			} `json:"watch"`
			Transport struct {
				Type string `json:"type"`
				Host string `json:"host"`
				Port int    `json:"port"`
				TLS  struct {
					Enable         bool   `json:"enable"`
					Validateserver bool   `json:"validateserver"`
					Cacertpath     string `json:"cacertpath"`
				} `json:"tls"`
			} `json:"transport"`
		} `json:"metric"`
		Event struct {
			Enable bool `json:"enable"`
			Format struct {
				Type           string `json:"type"`
				Maxeventpersec int    `json:"maxeventpersec"`
				Enhancefs      bool   `json:"enhancefs"`
			} `json:"format"`
			Watch []struct {
				Type        string      `json:"type"`
				Name        string      `json:"name"`
				Value       string      `json:"value"`
				Allowbinary bool        `json:"allowbinary,omitempty"`
				Field       string      `json:"field,omitempty"`
				Headers     interface{} `json:"headers,omitempty"`
			} `json:"watch"`
			Transport struct {
				Type string `json:"type"`
				Host string `json:"host"`
				Port int    `json:"port"`
				TLS  struct {
					Enable         bool   `json:"enable"`
					Validateserver bool   `json:"validateserver"`
					Cacertpath     string `json:"cacertpath"`
				} `json:"tls"`
			} `json:"transport"`
		} `json:"event"`
		Payload struct {
			Enable bool   `json:"enable"`
			Dir    string `json:"dir"`
		} `json:"payload"`
		Libscope struct {
			Configevent   bool   `json:"configevent"`
			Summaryperiod int    `json:"summaryperiod"`
			Commanddir    string `json:"commanddir"`
			Log           struct {
				Level     string `json:"level"`
				Transport struct {
					Type   string `json:"type"`
					Path   string `json:"path"`
					Buffer string `json:"buffer"`
				} `json:"transport"`
			} `json:"log"`
		} `json:"libscope"`
		Cribl struct {
			Enable    bool `json:"enable"`
			Transport struct {
				Type string `json:"type"`
				Host string `json:"host"`
				Port int    `json:"port"`
				TLS  struct {
					Enable         bool   `json:"enable"`
					Validateserver bool   `json:"validateserver"`
					Cacertpath     string `json:"cacertpath"`
				} `json:"tls"`
			} `json:"transport"`
		} `json:"cribl"`
		Tags     interface{} `json:"tags"`
		Protocol interface{} `json:"protocol"`
		Custom   interface{} `json:"custom"`
	} `json:"cfg"`
}

// Must be inline with server, see: ipcRespStatus
type scopeOnlyStatusResponse struct {
	// Response status
	Status respStatus `mapstructure:"status" json:"status" yaml:"status"`
}

// Must be inline with server, see: ipcRespGetScopeStatus
type scopeGetStatusResponse struct {
	// Response status
	Status respStatus `mapstructure:"status" json:"status" yaml:"status"`
	// Scoped status
	Scoped bool `mapstructure:"scoped" json:"scoped" yaml:"scoped"`
}

// Must be inline with server, see: ipcRespGetScopeCfg
type scopeGetCfgResponse struct {
	// Response status
	Status respStatus `mapstructure:"status" json:"status" yaml:"status"`
	// TODO: Scoped Cfg
	// Cfg libscope.ScopeConfig `mapstructure:"cfg" json:"cfg" yaml:"cfg"`
	// Enable should be boolString to use it
	Cfg struct {
		Current struct {
			Metric struct {
				Enable    string `json:"enable"`
				Transport struct {
					Type      string `json:"type"`
					Path      string `json:"path"`
					Buffering string `json:"buffering"`
				} `json:"transport"`
				Format struct {
					Type         string `json:"type"`
					Statsdprefix string `json:"statsdprefix"`
					Statsdmaxlen int    `json:"statsdmaxlen"`
					Verbosity    int    `json:"verbosity"`
				} `json:"format"`
				Watch []struct {
					Type string `json:"type"`
				} `json:"watch"`
			} `json:"metric"`
			Libscope struct {
				Log struct {
					Level     string `json:"level"`
					Transport struct {
						Type      string `json:"type"`
						Path      string `json:"path"`
						Buffering string `json:"buffering"`
					} `json:"transport"`
				} `json:"log"`
				Configevent   string `json:"configevent"`
				Summaryperiod int    `json:"summaryperiod"`
				Commanddir    string `json:"commanddir"`
			} `json:"libscope"`
			Event struct {
				Enable    string `json:"enable"`
				Transport struct {
					Type      string `json:"type"`
					Path      string `json:"path"`
					Buffering string `json:"buffering"`
				} `json:"transport"`
				Format struct {
					Type           string `json:"type"`
					Maxeventpersec int    `json:"maxeventpersec"`
					Enhancefs      string `json:"enhancefs"`
				} `json:"format"`
				Watch []struct {
					Type        string        `json:"type"`
					Name        string        `json:"name"`
					Field       string        `json:"field"`
					Value       string        `json:"value"`
					Allowbinary string        `json:"allowbinary,omitempty"`
					Headers     []interface{} `json:"headers,omitempty"`
				} `json:"watch"`
			} `json:"event"`
			Payload struct {
				Enable string `json:"enable"`
				Dir    string `json:"dir"`
			} `json:"payload"`
			Tags struct {
			} `json:"tags"`
			Protocol []interface{} `json:"protocol"`
			Cribl    struct {
				Enable    string `json:"enable"`
				Transport struct {
					Type string `json:"type"`
				} `json:"transport"`
				Authtoken string `json:"authtoken"`
			} `json:"cribl"`
		} `json:"current"`
	} `json:"cfg"`
}

type IpcResponseCtx struct {
	ResponseScopeMsgData []byte
	ResponseStatus       respStatus
}

type CmdGetScopeStatus struct {
	Response scopeGetStatusResponse
}

func (cmd *CmdGetScopeStatus) Request(pid int) (*IpcResponseCtx, error) {
	req, _ := json.Marshal(scopeRequestOnly{Req: reqCmdGetScopeStatus})

	return ipcDispatcher(req, pid)
}

func (cmd *CmdGetScopeStatus) UnmarshalResp(respData []byte) error {
	return json.Unmarshal(respData, &cmd.Response)
}

type CmdGetScopeCfg struct {
	Response scopeGetCfgResponse
}

func (cmd *CmdGetScopeCfg) Request(pid int) (*IpcResponseCtx, error) {
	req, _ := json.Marshal(scopeRequestOnly{Req: reqCmdGetScopeCfg})

	return ipcDispatcher(req, pid)
}

func (cmd *CmdGetScopeCfg) UnmarshalResp(respData []byte) error {
	return json.Unmarshal(respData, &cmd.Response)
}

type CmdSetScopeCfg struct {
	Response scopeOnlyStatusResponse
	CfgData  []byte
}

func (cmd *CmdSetScopeCfg) Request(pid int) (*IpcResponseCtx, error) {
	scopeReq := scopeRequestSetCfg{Req: reqCmdSetScopeCfg}

	err := yaml.Unmarshal(cmd.CfgData, &scopeReq.Cfg)
	if err != nil {
		return nil, err
	}
	req, _ := json.Marshal(scopeReq)

	return ipcDispatcher(req, pid)
}

func (cmd *CmdSetScopeCfg) UnmarshalResp(respData []byte) error {
	return json.Unmarshal(respData, &cmd.Response)
}
