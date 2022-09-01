package run

import (
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
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

// return status from start operation
var startErr error

// retrieve and unmarshall configuration passed in stdin
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

func setupHostCfg(cfgData []byte) error {
	const defaultLibLoc = "/tmp/libscope.so"
	const defaultFilterLoc = "/tmp/scope_filter.yml"
	const defaultProfileScript = "/etc/profile.d/scope.sh"

	asset, err := Asset("build/libscope.so")
	if err != nil {
		return err
	}

	// Setup libscope.so
	if err := os.WriteFile(defaultLibLoc, asset, 0775); err != nil {
		return err
	}
	// Setup filter file
	if err := os.WriteFile(defaultFilterLoc, cfgData, 0644); err != nil {
		return err
	}

	// Setup defualt Profile
	if err := os.WriteFile(defaultProfileScript, []byte(fmt.Sprintf("LoremIpsum=%s", defaultLibLoc)), 0644); err != nil {
		return err
	}

	// Enable this after last commit
	// if err := os.WriteFile(defaultProfileScript, []byte(fmt.Sprintf("LD_PRELOAD=%s", defaultLibLoc)), 0644); err != nil {
	// 	return err
	// }

	return nil
}

func SetupContainer(cfgData []byte, allowProcs []allowProcConfig) {
	os.Setenv("SCOPE_FILTER_PATH", "/tmp/scope_filter.yml")
	// Iterate over all containers
	cPids, _ := util.GetDockerPids()
	for _, cPid := range cPids {
		log.Debug().Msgf("Start set up namespace for Pid %d", cPid)
		// Setup /etc/profile.d
		// Extract Filter file
		// Extract libscope.so
		ld := loader.ScopeLoader{Path: ldscopePath()}
		if err := ld.Configure(cPid); err != nil {
			log.Error().Err(err).Msgf("Setup container namespace failed for PID %v", cPid)
			startErr = err
		} else {
			log.Debug().Msgf("Setup container succesfully PID: %v", cPid)
		}

		// Iterate over all allowed processses
		for _, process := range allowProcs {
			log.Debug().Msgf("Start setup service %v for container PID: %v", process.Procname, cPid)
			// Setup service
			if err := ld.ServiceContainer(process.Procname, cPid); err != nil {
				log.Error().Err(err).Msgf("Setup service %v failed for container PID: %v", process.Procname, cPid)
				startErr = err
			} else {
				log.Debug().Msgf("Setup service succesfully: %v for container PID: %v", process.Procname, cPid)
			}
		}
	}
	os.Unsetenv("SCOPE_FILTER_PATH")
}

func (rc *Config) Start() error {
	rc.setupWorkDir([]string{"start"}, true)
	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Fatal().Err(err)
		return err
	}

	startCfg, startcfgData, err := getConfigFromStdin()
	if err != nil {
		log.Fatal().Err(err)
		return err
	}

	if err := createLdscope(); err != nil {
		log.Fatal().Err(err)
		return err
	}

	// Setup Filter file on Host
	if err := setupHostCfg(startcfgData); err != nil {
		log.Fatal().Err(err)
		return err
	} else {
		log.Debug().Msg("Setup Filter on Host succeded.")
	}

	SetupContainer(startcfgData, startCfg.AllowProc)

	// Iterate over allowed process
	// Attach to services on Host and container
	// Setup service on Host
	for _, alllowProc := range startCfg.AllowProc {

		cfgSingleProc, err := yaml.Marshal(alllowProc.Config)
		if err != nil {
			log.Error().Err(err).Msgf("Serialize configuration %v", err)
			startErr = err
			continue
		}

		allProc, err := util.ProcessesByName(alllowProc.Procname)
		if err != nil {
			log.Error().Err(err).Msgf("Get Process name failed for %v", alllowProc.Procname)
			startErr = err
			continue
		}

		for _, process := range allProc {
			if err := rc.startAttachStage(process, cfgSingleProc); err != nil {
				log.Error().Err(err).Msgf("Attach failed for PID %v", process.Pid)
				startErr = err
			} else {
				log.Debug().Msgf("Attach succeded for PID %v", process.Pid)
			}

			// TODO: pass the libscope.so well known location and inform service about it
			// quiet status?
			// if err := rc.startServiceStage(alllowProc.Procname, cfgSingleProc); err != nil {
			// 	log.Error().Err(err).Msgf("Service setup failed for process: %v", alllowProc.Procname)
			// 	startErr = err
			// } else {
			// 	log.Debug().Msgf("Serivce setup succeded for process %v", alllowProc.Procname)
			// }
		}
	}

	// Deny list actions
	// Currently NOP
	// Detach ?
	// Deservice ?

	return startErr
}
