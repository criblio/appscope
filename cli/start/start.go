package start

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
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
	sL := loader.ScopeLoader{Path: run.LdscopePath()}
	stdoutStderr, err := sL.AttachSubProc([]string{pid}, env)
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
func startConfigureHost(filename string, noProfile bool) error {
	sL := loader.ScopeLoader{Path: run.LdscopePath()}
	stdoutStderr, err := sL.ConfigureHost(filename, noProfile)
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
func startConfigureContainers(filename string) error {
	// Discover all containers
	cPids, err := util.GetDockerPids()
	if err != nil {
		switch {
		case errors.Is(err, util.ErrDockerNotAvailable):
			log.Warn().
				Msgf("Configure containers skipped. Docker is not available")
		default:
			log.Error().
				Err(err).
				Msg("Configure containers failed.")
		}
		return nil
	}

	ld := loader.ScopeLoader{Path: run.LdscopePath()}

	// Iterate over all containers
	for _, cPid := range cPids {
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
	sL := loader.ScopeLoader{Path: run.LdscopePath()}

	// Iterate over all allowed processses
	for _, process := range allowProcs {
		if process.Procname != "" {
			stdoutStderr, err := sL.ServiceHost(process.Procname)
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
func startServiceContainers(allowProcs []allowProcConfig) error {
	// Discover all containers
	cPids, err := util.GetDockerPids()
	if err != nil {
		switch {
		case errors.Is(err, util.ErrDockerNotAvailable):
			log.Warn().
				Msgf("Setup container services skipped. Docker is not available")
		default:
			log.Warn().
				Err(err).
				Msg("Setup container services failed.")
		}
		return nil
	}

	ld := loader.ScopeLoader{Path: run.LdscopePath()}

	// Iterate over all allowed processses
	for _, process := range allowProcs {
		if process.Procname != "" {
			// Iterate over all containers
			for _, cPid := range cPids {
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

// extract extracts ldscope and scope to scope version directory
func extract(scopeDirVersion string) error {
	// Extract ldscope
	perms := os.FileMode(0755)
	b, err := run.Asset("build/ldscope")
	if err != nil {
		log.Error().
			Err(err).
			Msg("Error retrieving ldscope asset.")
		return err
	}

	err = os.MkdirAll(scopeDirVersion, perms)
	if err != nil {
		log.Error().
			Err(err).
			Msgf("Error creating %s directory.", scopeDirVersion)
		return err
	}

	if err = ioutil.WriteFile(filepath.Join(scopeDirVersion, "ldscope"), b, perms); err != nil {
		log.Error().
			Err(err).
			Msgf("Error writing ldscope to %s.", scopeDirVersion)
		return err
	}

	// Copy scope
	if _, err := util.CopyFile(os.Args[0], filepath.Join(scopeDirVersion, "scope")); err != nil {
		if err != os.ErrExist {
			log.Error().
				Err(err).
				Msgf("Error writing scope to %s.", scopeDirVersion)
			return err
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

// createFilterFile creates a filter file in /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
// It returns the filter file and status
func createFilterFile() (*os.File, error) {
	// Try to create AppScope directory in /usr/lib if it not exists yet
	if _, err := os.Stat("/usr/lib/appscope/"); os.IsNotExist(err) {
		os.MkdirAll("/usr/lib/appscope/", 0755)
	}

	f, err := os.OpenFile("/usr/lib/appscope/scope_filter", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		if _, err := os.Stat("/tmp/appscope"); os.IsNotExist(err) {
			os.MkdirAll("/tmp/appscope", 0755)
		}
		f, err = os.OpenFile("/tmp/appscope", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	}
	return f, err
}

// getAppScopeVerDir return the directory path which will be used for handling ldscope and scope files
// /usr/lib/appscope/<version>/
// or
// /tmp/appscope/<version>/
func getAppScopeVerDir() (string, error) {
	version := internal.GetNormalizedVersion()
	var appscopeVersionPath string

	// Check /usr/lib/appscope only for official version
	if !internal.IsVersionDev() {
		appscopeVersionPath = filepath.Join("/usr/lib/appscope/", version)
		if _, err := os.Stat(appscopeVersionPath); err != nil {
			err := os.MkdirAll(appscopeVersionPath, 0755)
			return appscopeVersionPath, err
		}
		return appscopeVersionPath, nil
	}

	appscopeVersionPath = filepath.Join("/tmp/appscope/", version)
	if _, err := os.Stat(appscopeVersionPath); err != nil {
		err := os.MkdirAll(appscopeVersionPath, 0755)
		return appscopeVersionPath, err
	}
	return appscopeVersionPath, nil

}

// extractFilterFile creates a filter file in /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
// It returns the filter file name and status
func extractFilterFile(cfgData []byte) (string, error) {
	f, err := createFilterFile()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Failed to create filter file.")
		return "", err
	}
	defer f.Close()

	if _, err = f.Write(cfgData); err != nil {
		log.Error().
			Err(err).
			Msg("Failed to write to filter file.")
		return "", err
	}
	return f.Name(), nil
}

// Start performs setup of scope in the host and containers
// optionally disable setup /etc/profile script
func Start(noProfile bool) error {
	// Create a history directory for logs
	createWorkDir()

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
	// If the `scope start` command is run inside a container, we should call `ldscope --starthost`
	// which will instead run `scope start` on the host
	// SCOPE_CLI_SKIP_START_HOST allows to run scope start in docker environment (integration tests)
	_, skipHostCfg := os.LookupEnv("SCOPE_CLI_SKIP_START_HOST")
	if util.InContainer() && !skipHostCfg {
		if err := extract(scopeDirVersion); err != nil {
			return err
		}

		ld := loader.ScopeLoader{Path: filepath.Join(scopeDirVersion, "ldscope")}
		stdoutStderr, err := ld.StartHost(noProfile)
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

	if err := run.CreateLdscope(); err != nil {
		log.Error().
			Err(err).
			Msg("Create ldscope failed.")
		return err
	}

	if err := startConfigureHost(filterPath, noProfile); err != nil {
		return err
	}

	if err := startConfigureContainers(filterPath); err != nil {
		return err
	}

	if err := startServiceHost(startCfg.AllowProc); err != nil {
		return err
	}

	if err := startServiceContainers(startCfg.AllowProc); err != nil {
		return err
	}

	if err := startAttach(startCfg.AllowProc); err != nil {
		return err
	}

	// TODO: Deny list actions (Detach?/Deservice?)
	return nil
}
