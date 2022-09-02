package run

import (
	"io"
	"io/ioutil"
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
		defer os.Remove(tmpFile.Name())

		if _, err := tmpFile.Write(cfgData); err != nil {
			return err
		}

		if err := tmpFile.Close(); err != nil {
			return err
		}

		env := append(os.Environ(), "SCOPE_CONF_PATH="+tmpFile.Name())
		ld := loader.ScopeLoader{Path: ldscopePath()}
		if err := ld.AttachSubProc([]string{pid}, env); err != nil {
			return err
		}
		os.Unsetenv("SCOPE_CONF_PATH")
		return err
	}
	return errAlreadyScope
}

func setupHost(filterData []byte) error {
	tempFile, err := ioutil.TempFile("/tmp", "temp_filter")
	if err != nil {
		return err
	} else {
		log.Debug().Msg("Create test_filter succeded.")
	}
	defer os.Remove(tempFile.Name())

	if _, err := tempFile.Write(filterData); err != nil {
		return err
	} else {
		log.Debug().Msg("Write test_filter succeded.")
	}

	ld := loader.ScopeLoader{Path: ldscopePath()}
	if err := ld.ConfigureHost(tempFile.Name()); err != nil {
		os.Remove(tempFile.Name())
		log.Debug().Msg("Setup on Host failed.")
		log.Fatal().Err(err)
		return err
	} else {
		log.Debug().Msg("Setup on Host succeded.")
	}

	return nil
}

func SetupContainer(allowProcs []allowProcConfig) error {
	// Iterate over all containers
	cPids, _ := util.GetDockerPids()
	for _, cPid := range cPids {
		log.Debug().Msgf("Start set up namespace for Pid %d", cPid)

		// Setup /etc/profile.d
		// Extract filter file
		// Extract libscope.so

		ld := loader.ScopeLoader{Path: ldscopePath()}
		if err := ld.ConfigureContainer("/tmp/scope_filter.yml", cPid); err != nil {
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
	return nil
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

	// Setup Host
	if err := setupHost(startcfgData); err != nil {
		log.Error().Err(err).Msgf("Setup host failed: %v", err)
		startErr = err
	} else {
		log.Debug().Msgf("Setup host succesfully")
	}

	if err := SetupContainer(startCfg.AllowProc); err != nil {
		log.Error().Err(err).Msgf("Setup container failed: %v", err)
		startErr = err
	} else {
		log.Debug().Msgf("Setup container succesfully")
	}

	// Iterate over allowed process
	// Attach to services on Host and container
	// Setup service on Host
	for _, alllowProc := range startCfg.AllowProc {

		// Setup service
		ld := loader.ScopeLoader{Path: ldscopePath()}
		if err := ld.ServiceHost(alllowProc.Procname); err != nil {
			log.Error().Err(err).Msgf("Setup service %v failed for host.", alllowProc.Procname)
			startErr = err
		} else {
			log.Debug().Msgf("Setup service succesfully: %v for host.", alllowProc.Procname)
		}

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
		}
	}

	// Deny list actions
	// Currently NOP
	// Detach ?
	// Deservice ?

	return startErr
}
