package run

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
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
}

// allowProcConfig represents a allowed process configuration
type allowProcConfig struct {
	Procname string               `mapstructure:"procname,omitempty" json:"procname,omitempty" yaml:"procname,omitempty"`
	Arg      string               `mapstructure:"arg,omitempty" json:"arg,omitempty" yaml:"arg,omitempty"`
	Config   libscope.ScopeConfig `mapstructure:"config" json:"config" yaml:"config"`
}

// return status from start operation
var startErr error

var (
	errMissingData = errors.New("missing filter data")
)

// startConfigfromData loads a start configuration from data
// It returns the configuration.
func startConfigfromData(dataIn []byte) (startConfig, error) {

	var startCfg startConfig

	err := yaml.Unmarshal(dataIn, &startCfg)

	return startCfg, err
}

// startAttachSingleProcess attach to the the specific process identifed by pid with configuration pass in cfgData.
// It returns the status of operation.
func startAttachSingleProcess(pid string, cfgData []byte) error {
	tmpFile, err := os.Create(fmt.Sprintf("/tmp/tmpAttachCfg_%s.yml", pid))
	if err != nil {
		log.Error().
			Err(err).
			Str("pid", pid).
			Msgf("Attach PID %v Failed. Create temporary attach configuration failed.", pid)
		startErr = err
		return err
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(cfgData); err != nil {
		log.Error().
			Err(err).
			Str("pid", pid).
			Msgf("Attach PID %v Failed. Write temporary attach configuration failed.", pid)
		startErr = err
		return err
	}

	env := append(os.Environ(), "SCOPE_CONF_PATH="+tmpFile.Name())
	sL := loader.ScopeLoader{Path: ldscopePath()}
	stdoutStderr, err := sL.AttachSubProc([]string{pid}, env)
	if err != nil {
		log.Error().
			Err(err).
			Str("pid", pid).
			Str("loaderDetails", stdoutStderr).
			Msgf("Attach PID %v Failed. Write temporary attach configuration failed.", pid)
		startErr = err
	} else {
		log.Info().
			Str("pid", pid).
			Msgf("Attach PID %v success.", pid)
	}

	os.Unsetenv("SCOPE_CONF_PATH")
	return err
}

// startAttach attach to all allowed processes on the host and on the container
// It returns the status of operation.
func startAttach(allowProc allowProcConfig) error {
	var pidsToAttach util.PidScopeMapState
	cfgSingleProc, err := yaml.Marshal(allowProc.Config)
	if err != nil {
		log.Error().
			Err(err).
			Msg("Attach Failed. Serialize configuration failed.")
		startErr = err
		return err
	}

	if allowProc.Procname != "" {
		procsMap, err := util.PidScopeMapByProcessName(allowProc.Procname)
		if err != nil {
			log.Error().
				Err(err).
				Str("process", allowProc.Procname).
				Msgf("Attach Failed. Retrieve process name: %v failed.", allowProc.Procname)
			startErr = err
			return err
		}
		pidsToAttach = procsMap
	}
	if allowProc.Arg != "" {
		procsMap, err := util.PidScopeMapByCmdLine(allowProc.Arg)
		if err != nil {
			log.Error().
				Err(err).
				Str("process", allowProc.Arg).
				Msgf("Attach Failed. Retrieve arg: %v failed.", allowProc.Arg)
			startErr = err
			return err
		}
		for k, v := range procsMap {
			pidsToAttach[k] = v
		}
	}

	for pid, scopeState := range pidsToAttach {
		if scopeState {
			log.Info().
				Str("pid", strconv.Itoa(pid)).
				Msgf("Attach Failed. Process: %v is already scoped.", pid)
			continue
		}

		startAttachSingleProcess(strconv.Itoa(pid), cfgSingleProc)

	}

	return nil
}

// startConfigureHost setup the Host:
//
// - setup /etc/profile.d/scope.sh
//
// - extract filter into the filter file in /tmp/scope_filter.yml
//
// - extract libscope.so into /tmp/libscope.so
//
// It returns the status of operation.
func startConfigureHost(filterData []byte) error {
	tempFile, err := ioutil.TempFile("/tmp", "temp_filter")
	if err != nil {
		log.Error().
			Err(err).
			Msg("Setup host failed. Create temporary filter file.")
		startErr = err
		return err
	} else {
		log.Info().
			Msg("Setup host. Create test_filter success.")
	}
	defer os.Remove(tempFile.Name())

	if _, err := tempFile.Write(filterData); err != nil {
		log.Error().
			Err(err).
			Msg("Setup host failed. Write to temporary filter file.")
		startErr = err
		return err
	} else {
		log.Info().
			Msg("Setup host. Write test_filter success.")
	}

	sL := loader.ScopeLoader{Path: ldscopePath()}

	stdoutStderr, err := sL.ConfigureHost(tempFile.Name())
	if err != nil {
		log.Error().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msg("Setup host failed.")
		startErr = err
		return err
	} else {
		log.Info().
			Msg("Setup host success.")
	}

	return nil
}

// startServiceHost setup the Container:
//
// - setup service defined by the serviceName on the host.
//
// It returns the status of operation.
func startServiceHost(serviceName string) error {
	sL := loader.ScopeLoader{Path: ldscopePath()}
	stdoutStderr, err := sL.ServiceHost(serviceName)
	if err == nil {
		log.Info().
			Str("service", serviceName).
			Msgf("Service %v host success.", serviceName)
	} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
		if ee.ExitCode() == 1 {
			log.Error().
				Err(err).
				Str("service", serviceName).
				Str("loaderDetails", stdoutStderr).
				Msgf("Service %v host failed.", serviceName)
			startErr = err
		} else {
			log.Info().
				Str("service", serviceName).
				Str("loaderDetails", stdoutStderr).
				Msgf("Service %v host failed.", serviceName)
		}
	}
	return err
}

// startSetupContainer setup the Container:
//
// - setup /etc/profile.d/scope.sh in the container.
//
// - extract filter into the filter file in /tmp/scope_filter.yml in the container.
//
// - extract libscope.so into /tmp/libscope.so in the container.
//
// - service services defined by the allowProcs process name list in the container.
//
// It returns the status of operation.
func startSetupContainer(allowProcs []allowProcConfig) error {
	// Iterate over all containers
	cPids, err := util.GetDockerPids()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Setup containers failed.")
		startErr = err
	}

	for _, cPid := range cPids {
		ld := loader.ScopeLoader{Path: ldscopePath()}
		stdoutStderr, err := ld.ConfigureContainer("/tmp/scope_filter.yml", cPid)
		if err != nil {
			log.Error().
				Err(err).
				Str("pid", strconv.Itoa(cPid)).
				Str("loaderDetails", stdoutStderr).
				Msgf("Setup containers failed. Setup container %v failed.", cPid)
			startErr = err
		} else {
			log.Info().
				Str("pid", strconv.Itoa(cPid)).
				Msgf("Setup containers. Setup container %v success.", cPid)
		}

		// Iterate over all allowed processses
		for _, process := range allowProcs {
			// Setup service
			stdoutStderr, err := ld.ServiceContainer(process.Procname, cPid)
			if err == nil {
				log.Info().
					Str("service", process.Procname).
					Str("pid", strconv.Itoa(cPid)).
					Msgf("Setup containers. Service %v container %v success.", process.Procname, cPid)
			} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
				if ee.ExitCode() == 1 {
					log.Error().
						Err(err).
						Str("service", process.Procname).
						Str("pid", strconv.Itoa(cPid)).
						Str("loaderDetails", stdoutStderr).
						Msgf("Setup containers failed. Service %v container %v failed.", process.Procname, cPid)
					startErr = err
				} else {
					log.Info().
						Str("service", process.Procname).
						Str("pid", strconv.Itoa(cPid)).
						Str("loaderDetails", stdoutStderr).
						Msgf("Setup containers failed. Service %v container %v failed.", process.Procname, cPid)
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

	// Verify size of data for mode other than a pipe
	if stdinFs.Size() == 0 && stdinFs.Mode()&os.ModeNamedPipe == 0 {
		return startData
	}

	startData, _ = io.ReadAll(os.Stdin)
	return startData
}

// Start runs start command responsible for setup host and container env
// It returns the status of operation.
func (rc *Config) Start() error {
	rc.setupWorkDir([]string{"start"}, true)

	cfgData := getStartData()
	if len(cfgData) == 0 {
		startErr = errMissingData
		log.Error().
			Err(startErr).
			Msg("Missing filter data.")
		return startErr
	}

	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Error().
			Err(err).
			Msg("Verify root permission failed.")
		startErr = err
		return err
	}

	// Retrieve input data
	startCfg, err := startConfigfromData(cfgData)
	if err != nil {
		log.Error().
			Err(err).
			Msg("Read filter file failed.")
		startErr = err
		return err
	}

	if err := createLdscope(); err != nil {
		log.Error().
			Err(err).
			Msg("Create ldscope failed.")
		startErr = err
		return err
	}

	// Configure Host
	startConfigureHost(cfgData)

	// Setup Containers
	startSetupContainer(startCfg.AllowProc)

	// Iterate over allowed process
	// Setup service on Host
	// Attach to services on Host and container
	for _, allowProc := range startCfg.AllowProc {

		startServiceHost(allowProc.Procname)

		startAttach(allowProc)
	}

	// TODO: Deny list actions (Detach?/Deservice?)

	return startErr
}
