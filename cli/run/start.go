package run

import (
	"errors"
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

func getConfigFromStdin() (startConfig, []byte, error) {

	var startCfg startConfig

	data, err := io.ReadAll(os.Stdin)

	if err != nil {
		return startCfg, data, err
	}

	err = yaml.Unmarshal(data, &startCfg)

	return startCfg, data, err
}

func (rc *Config) startAttachStage(process util.Process, cfgData []byte) error {

	if !process.Scoped {
		pid := strconv.Itoa(process.Pid)
		tmpFile, err := os.CreateTemp(".", pid)
		if err != nil {
			return err
		}

		if _, err := tmpFile.Write(cfgData); err != nil {
			os.Remove(tmpFile.Name())
			return err
		}

		if err := tmpFile.Close(); err != nil {
			os.Remove(tmpFile.Name())
			return err
		}

		rc.Subprocess = true
		rc.Passthrough = true
		os.Setenv("SCOPE_CONF_PATH", tmpFile.Name())
		err = rc.Attach([]string{pid})
		os.Unsetenv("SCOPE_CONF_PATH")
		os.Remove(tmpFile.Name())
		return err
	}
	return errAlreadyScope
}

func (rc *Config) startServiceStage(procName string, cfgData []byte) error {
	return rc.Service(procName, "root", true)
}

func startInterProcessStageHost(cfgData []byte) error {
	const ProfileScopeScript = "/etc/profile.d/scope.sh"
	// Setup "/etc/profile.d/scope.sh"
	if _, err := os.Stat(ProfileScopeScript); errors.Is(err, os.ErrNotExist) {
		// const LibLoc = "/tmp/libscope.so"
		// profileSetup := fmt.Sprintf("LD_PRELOAD=%s", LibLoc)
		const profileSetup = "Lorem Ipsum"
		if err := os.WriteFile(ProfileScopeScript, []byte(profileSetup), 0644); err != nil {
			return err
		}
	}

	return nil
}

func startSetupFilterCfgHost(cfgData []byte) error {

	const LibLoc = "/tmp/libscope.so"

	if _, err := os.Stat(LibLoc); errors.Is(err, os.ErrNotExist) {
		// Save main configuration file
		if err := os.WriteFile(LibLoc, cfgData, 0644); err != nil {
			return err
		}
	}
	const EdgeCfgLoc = "/tmp/scope_main.yml"
	return os.WriteFile(EdgeCfgLoc, cfgData, 0644)
}

func setupFilterCfgOnHost(cfgData []byte) error {
	return os.WriteFile("/tmp/scope_filter.yml", cfgData, 0644)
}

func (rc *Config) Start() error {
	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		return err
	}
	startCfg, startcfgData, err := getConfigFromStdin()
	if err != nil {
		return err
	}

	if err := createLdscope(); err != nil {
		return err
	}

	// Setup Filter file on Host
	if err := setupFilterCfgOnHost(startcfgData); err != nil {
		return err
	}

	// Setup Container namespace
	os.Setenv("SCOPE_FILTER_PATH", "/tmp/scope_filter.yml")
	// Iterate over all containers
	cPids, _ := util.GetDockerPids()
	// fmt.Println("Getting container PIDs status", err)
	for _, cPid := range cPids {
		// fmt.Println("Container Process Id", cPid)
		err = rc.SetupContainer(cPid)
		fmt.Println("Container Process Id Setup status", err)
	}
	os.Unsetenv("SCOPE_FILTER_PATH")

	// Host

	// Setup filter on the host

	// TODO Replace fmt.print with cli Log

	// Start from the Host

	// Setup Filter Stage
	// Description: Copy the filter file and libscope.so to well known location /tmp
	// Notes: The stage is required to iterate over containers and host
	//        TODO: need access to all container
	// err = startSetupFilterCfgHost(startcfgData)
	// fmt.Println("Setup Filter Host  Status:", err)

	// // Setup Interactive Process Stage Host
	// // Description: Setup the /etc/profile.d/scope.sh script
	// // Notes: The stage is required to iterate over containers and host
	// //        TODO: need access to all containers
	// err = startInterProcessStageHost(startcfgData)
	// fmt.Println("Interactive Process Host Setup Status", err)

	// Run SetupFilter
	// Run SetupIntePRocess
	// For each allow list process
	//    Run Service Stage	}
	//

	for _, alllowProc := range startCfg.AllowProc {

		cfgSingleProc, err := yaml.Marshal(alllowProc.Config)
		if err != nil {
			// Error return/continue ?
			fmt.Println("error Unmarshall configuration %w", err)
			continue
		}

		allProc, err := util.ProcessesByName(alllowProc.Procname)
		if err != nil {
			// Error return/continue ?
			fmt.Println("error in get process by name %w", err)
			continue
		}

		// Attach Stage
		// Description: Perform attach to all processes founded by specific name
		// Notes: The stage does not need to iterate over containers
		for _, process := range allProc {
			err := rc.startAttachStage(process, cfgSingleProc)
			fmt.Println("Attach Status", alllowProc.Procname, process.Pid, err)
		}

		// Service Stage Host
		// Description: Perform service to all processes founded by specific name
		// Notes: The stage is required to iterate over containers and host
		//        TODO: pass the libscope.so well known location
		//        TODO: need access to all containers
		//        TODO: quiet option output?
		err = rc.startServiceStage(alllowProc.Procname, cfgSingleProc)
		fmt.Println("Service Status", alllowProc.Procname, err)
	}

	// Deny list actions
	// Currently NOP
	// Detach ?
	// Deservice ?

	return nil
}
