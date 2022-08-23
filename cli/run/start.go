package run

import (
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/util"
	"gopkg.in/yaml.v2"
)

// startConfig represents processed configuration
type startConfig struct {
	AllowProc []allowProcConfig `mapstructure:"allow,omitempty" json:"allow,omitempty" yaml:"allow"`
	DenyProc  []procConfig      `mapstructure:"deny,omitempty" json:"deny,omitempty" yaml:"deny"`
}

// procConfig represents a process configuration
type procConfig struct {
	Procname string `mapstructure:"procname,omitempty" json:"procname,omitempty" yaml:"procname,omitempty"`
	Arg      string `mapstructure:"arg,omitempty" json:"arg,omitempty" yaml:"arg,omitempty"`
	Hostname string `mapstructure:"hostname,omitempty" json:"hostname,omitempty" yaml:"hostname,omitempty"`
	Username string `mapstructure:"username,omitempty" json:"username,omitempty" yaml:"username,omitempty"`
	Env      string `mapstructure:"env,omitempty" json:"env,omitempty" yaml:"env,omitempty"`
	Ancestor string `mapstructure:"ancestor,omitempty" json:"ancestor,omitempty" yaml:"ancestor,omitempty"`
}

// allowProcConfig represents a allowed process configuration
type allowProcConfig struct {
	Procname string               `mapstructure:"procname,omitempty" json:"procname,omitempty" yaml:"procname,omitempty"`
	Arg      string               `mapstructure:"arg,omitempty" json:"arg,omitempty" yaml:"arg,omitempty"`
	Hostname string               `mapstructure:"hostname,omitempty" json:"hostname,omitempty" yaml:"hostname,omitempty"`
	Username string               `mapstructure:"username,omitempty" json:"username,omitempty" yaml:"username,omitempty"`
	Env      string               `mapstructure:"env,omitempty" json:"env,omitempty" yaml:"env,omitempty"`
	Ancestor string               `mapstructure:"ancestor,omitempty" json:"ancestor,omitempty" yaml:"ancestor,omitempty"`
	Config   libscope.ScopeConfig `mapstructure:"config" json:"config" yaml:"config"`
}

func getConfigFromStdin() (startConfig, error) {

	var startCfg startConfig
	var err error

	data, err := io.ReadAll(os.Stdin)

	if err != nil {
		return startCfg, err
	}

	err = yaml.Unmarshal(data, &startCfg)

	return startCfg, err
}

func (rc *Config) Start() error {
	startCfg, err := getConfigFromStdin()
	if err != nil {
		return err
	}

	// TODO Remove the Printf
	// fmt.Println("Start parsing Allow Process List")
	for _, alllowProc := range startCfg.AllowProc {
		// TODO Remove the Printf below
		// fmt.Println("Parsing Allow Process", i)
		// Attach
		allProcToAttach, err := util.ProcessesByName(alllowProc.Procname)
		if err != nil {
			fmt.Println("error in get pocess by name %w", err)
			continue
		}
		for _, procToAttach := range allProcToAttach {
			fmt.Println("Following Process", alllowProc.Procname, "PID", procToAttach.Pid, "will be attached")
			if !procToAttach.Scoped {
				err := rc.Attach([]string{strconv.Itoa(procToAttach.Pid)})
				if err == nil {
					fmt.Println("Process", alllowProc.Procname, "PID", procToAttach.Pid, "successfully attached")
				} else {
					fmt.Println("Process", alllowProc.Procname, "PID", procToAttach.Pid, "failed with attach", err)
				}
			} else {
				fmt.Println("Process", alllowProc.Procname, "PID", procToAttach.Pid, "attach skipped - already scoped")
			}
			// TODO Service

			// TODO Handle interactive process

		}
		// fmt.Printf("%+v\n", alllowProc)
	}
	// TODO Remove the Println below
	// fmt.Println("DEBUG End parsing Allow Process List")
	// fmt.Println("DEBUG Start Deny Process List")
	// for i, _ := range startCfg.DenyProc {
	// TODO Remove the Printf below
	// fmt.Println("Parsing Deny Process", i)
	// TODO Deservice ??
	// TODO Deattach ??
	// fmt.Printf("%+v\n", denyProc)
	// }
	// TODO Remove the Println below
	// fmt.Println("DEBUG End parsing Deny Process List")

	return nil
}
