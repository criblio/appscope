package start

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v2"
)

// startConfig represents a processed configuration
type startConfig struct {
	AllowProc []allowProcConfig `mapstructure:"allow,omitempty" json:"allow,omitempty" yaml:"allow"`
	DenyProc  []denyProcConfig  `mapstructure:"deny,omitempty" json:"deny,omitempty" yaml:"deny"`
}

// denyProcConfig represents a denied process configuration
type denyProcConfig struct {
	Procname string `mapstructure:"procname,omitempty" json:"procname,omitempty" yaml:"procname,omitempty"`
	Arg      string `mapstructure:"arg,omitempty" json:"arg,omitempty" yaml:"arg,omitempty"`
}

// allowProcConfig represents a allowed process configuration
type allowProcConfig struct {
	Procname string               `mapstructure:"procname,omitempty" json:"procname,omitempty" yaml:"procname,omitempty"`
	Arg      string               `mapstructure:"arg,omitempty" json:"arg,omitempty" yaml:"arg,omitempty"`
	Config   libscope.ScopeConfig `mapstructure:"config" json:"config" yaml:"config"`
}

var (
	errMissingData = errors.New("missing filter data")
)

// TODO: the file creation steps here might not be needed because a filter file should be present
// and libscope.so should detect it (when that functionality is complete)
// startAttachSingleProcess attach to the the specific process identifed by pid with configuration pass in cfgData.
// It returns the status of operation.
func startAttachSingleProcess(pid string, cfgData []byte) error {
	tmpFile, err := os.Create(fmt.Sprintf("/tmp/tmpAttachCfg_%s.yml", pid))
	if err != nil {
		log.Error().
			Err(err).
			Str("pid", pid).
			Msgf("Attach PID %v failed. Create temporary attach configuration failed.", pid)
		return err
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(cfgData); err != nil {
		log.Error().
			Err(err).
			Str("pid", pid).
			Msgf("Attach PID %v failed. Write temporary attach configuration failed.", pid)
		return err
	}

	env := append(os.Environ(), "SCOPE_CONF_PATH="+tmpFile.Name())
	ld := loader.New()
	stdoutStderr, err := ld.AttachSubProc([]string{pid}, env)
	if err != nil {
		log.Error().
			Err(err).
			Str("pid", pid).
			Str("loaderDetails", stdoutStderr).
			Msgf("Attach PID %v failed.", pid)
		return err
	} else {
		log.Info().
			Str("pid", pid).
			Msgf("Attach PID %v success.", pid)
	}

	os.Unsetenv("SCOPE_CONF_PATH")
	return nil
}

// for the host
// for all containers
// for all allowed proc OR arg
func startAttach(allowProcs []allowProcConfig) error {
	// Iterate over all allowed processses
	for _, process := range allowProcs {
		pidsToAttach := make(util.PidScopeMapState)
		cfgSingleProc, err := yaml.Marshal(process.Config)
		if err != nil {
			log.Error().
				Err(err).
				Msg("Attach Failed. Serialize configuration failed.")
			return err
		}

		if process.Procname != "" {
			procsMap, err := util.PidScopeMapByProcessName(process.Procname)
			if err != nil {
				log.Error().
					Err(err).
					Str("process", process.Procname).
					Msgf("Attach Failed. Retrieve process name: %v failed.", process.Procname)
				return err
			}
			pidsToAttach = procsMap
		}
		if process.Arg != "" {
			procsMap, err := util.PidScopeMapByCmdLine(process.Arg)
			if err != nil {
				log.Error().
					Err(err).
					Str("process", process.Arg).
					Msgf("Attach Failed. Retrieve arg: %v failed.", process.Arg)
				return err
			}
			for k, v := range procsMap {
				pidsToAttach[k] = v
			}
		}

		// Exclude current scope process all Os threads from attach list
		scopeThreads, err := util.PidThreadsPids(os.Getpid())
		if err != nil {
			log.Warn().
				Str("pid", strconv.Itoa(os.Getpid())).
				Msgf("Exclude scope process %v from attach failed.", os.Getpid())
		}

		for _, scopePid := range scopeThreads {
			delete(pidsToAttach, scopePid)
		}

		for pid, scopeState := range pidsToAttach {
			if scopeState {
				log.Warn().
					Str("pid", strconv.Itoa(pid)).
					Msgf("Attach Failed. Process: %v is already scoped.", pid)
				continue
			}

			startAttachSingleProcess(strconv.Itoa(pid), cfgSingleProc)
		}
	}
	return nil
}

// for the host
func startConfigureHost(filename string) error {
	ld := loader.New()
	stdoutStderr, err := ld.ConfigureHost(filename)
	if err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msg("Setup host failed.")
	} else {
		log.Info().
			Msg("Setup host success.")
	}

	return nil
}

// for all containers
func startConfigureContainers(filename string, cPids []int) error {
	// Iterate over all containers
	for _, cPid := range cPids {
		ld := loader.New()
		stdoutStderr, err := ld.ConfigureContainer(filename, cPid)
		if err != nil {
			log.Warn().
				Err(err).
				Str("pid", strconv.Itoa(cPid)).
				Str("loaderDetails", stdoutStderr).
				Msgf("Configure containers failed. Configure container %v failed.", cPid)
		} else {
			log.Info().
				Str("pid", strconv.Itoa(cPid)).
				Msgf("Configure containers success. Configure container %v success.", cPid)
		}
	}

	return nil
}

// for the host
// for all allowed proc
func startServiceHost(allowProcs []allowProcConfig) error {
	ld := loader.New()

	// Iterate over all allowed processses
	for _, process := range allowProcs {
		if process.Procname != "" {
			stdoutStderr, err := ld.ServiceHost(process.Procname)
			if err == nil {
				log.Info().
					Str("service", process.Procname).
					Msgf("Service %v host success.", process.Procname)
			} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
				if ee.ExitCode() == 1 {
					log.Warn().
						Err(err).
						Str("service", process.Procname).
						Str("loaderDetails", stdoutStderr).
						Msgf("Service %v host failed.", process.Procname)
				} else {
					log.Warn().
						Str("service", process.Procname).
						Str("loaderDetails", stdoutStderr).
						Msgf("Service %v host failed.", process.Procname)
				}
			}
		}
	}
	return nil
}

// for all containers
// for all allowed proc
func startServiceContainers(allowProcs []allowProcConfig, cPids []int) error {
	// Iterate over all allowed processses
	for _, process := range allowProcs {
		if process.Procname != "" {
			// Iterate over all containers
			for _, cPid := range cPids {
				ld := loader.New()
				stdoutStderr, err := ld.ServiceContainer(process.Procname, cPid)
				if err == nil {
					log.Info().
						Str("service", process.Procname).
						Str("pid", strconv.Itoa(cPid)).
						Msgf("Setup containers. Service %v container %v success.", process.Procname, cPid)
				} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
					if ee.ExitCode() == 1 {
						log.Warn().
							Err(err).
							Str("service", process.Procname).
							Str("pid", strconv.Itoa(cPid)).
							Str("loaderDetails", stdoutStderr).
							Msgf("Setup containers failed. Service %v container %v failed.", process.Procname, cPid)
					} else {
						log.Warn().
							Str("service", process.Procname).
							Str("pid", strconv.Itoa(cPid)).
							Str("loaderDetails", stdoutStderr).
							Msgf("Setup containers failed. Service %v container %v failed.", process.Procname, cPid)
					}
				}
			}
		}
	}
	return nil
}

// getStartData reads data from Stdin
func getStartData() []byte {
	var startData []byte
	stdinFs, err := os.Stdin.Stat()
	if err != nil {
		return startData
	}

	// Avoid waiting for input from terminal when no data was provided
	if stdinFs.Mode()&os.ModeCharDevice != 0 && stdinFs.Size() == 0 {
		return startData
	}

	startData, _ = io.ReadAll(os.Stdin)
	return startData
}

// Start performs setup of scope in the host and containers
func Start() error {
	// Create a history directory for logs
	createWorkDir("start")

	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Error().
			Msg("Scope start requires administrator privileges")
		return err
	}

	cfgData := getStartData()
	if len(cfgData) == 0 {
		log.Error().
			Err(errMissingData).
			Msg("Missing filter data.")
		return errMissingData
	}

	var startCfg startConfig
	if err := yaml.Unmarshal(cfgData, &startCfg); err != nil {
		log.Error().
			Err(err).
			Msg("Read of filter file failed.")
		return err
	}

	filterPath, err := extractFilterFile(cfgData)
	if err != nil {
		return err
	}

	scopeDirVersion, err := getAppScopeVerDir()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Error Access AppScope version directory")
		return err
	}
	// If the `scope start` command is run inside a container, we should call `scope -z --starthost`
	// which will instead run `scope start` on the host
	// SCOPE_CLI_SKIP_HOST allows to run scope start in docker environment (integration tests)
	_, skipHostCfg := os.LookupEnv("SCOPE_CLI_SKIP_HOST")
	if util.InContainer() && !skipHostCfg {
		if err := extract(scopeDirVersion); err != nil {
			return err
		}

		ld := loader.New()
		stdoutStderr, err := ld.StartHost()
		if err == nil {
			log.Info().
				Msgf("Start host success.")
		} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
			if ee.ExitCode() == 1 {
				log.Error().
					Err(err).
					Str("loaderDetails", stdoutStderr).
					Msgf("Start host failed.")
				return err
			} else {
				log.Error().
					Str("loaderDetails", stdoutStderr).
					Msgf("Start host failed.")
				return err
			}
		}
		return nil
	}

	// Discover all container PIDs
	cPids := getContainersPids()

	if err := startConfigureHost(filterPath); err != nil {
		return err
	}

	if err := startConfigureContainers(filterPath, cPids); err != nil {
		return err
	}

	if err := startServiceHost(startCfg.AllowProc); err != nil {
		return err
	}

	if err := startServiceContainers(startCfg.AllowProc, cPids); err != nil {
		return err
	}

	if err := startAttach(startCfg.AllowProc); err != nil {
		return err
	}

	return nil
}
