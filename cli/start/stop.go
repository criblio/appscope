package start

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"

	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

var (
	errRemovingFiles = errors.New("error removing start files. See logs for details")
)

// for the host
func stopConfigureHost() error {
	sL := loader.ScopeLoader{Path: run.LdscopePath()}

	stdoutStderr, err := sL.UnconfigureHost()
	if err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msg("Unconfigure host failed.")
	} else {
		log.Info().
			Msg("Unconfigure host success.")
	}

	return nil
}

// for all containers
func stopConfigureContainers() error {
	sL := loader.ScopeLoader{Path: run.LdscopePath()}

	// Discover all containers
	cPids, err := util.GetDockerPids()
	if err != nil {
		switch {
		case errors.Is(err, util.ErrDockerNotAvailable):
			log.Warn().
				Msgf("Unconfigure containers skipped. Docker is not available")
		default:
			log.Error().
				Err(err).
				Msg("Unconfigure containers failed.")
		}
		return nil
	}

	// Iterate over all containers
	for _, cPid := range cPids {
		stdoutStderr, err := sL.UnconfigureContainer(cPid)
		if err != nil {
			log.Warn().
				Err(err).
				Str("pid", strconv.Itoa(cPid)).
				Str("loaderDetails", stdoutStderr).
				Msgf("Unconfigure containers failed. Container %v failed.", cPid)
		} else {
			log.Info().
				Str("pid", strconv.Itoa(cPid)).
				Msgf("Unconfigure containers success. Container %v success.", cPid)
		}
	}

	return nil
}

// for the host
func stopServiceHost() error {
	sL := loader.ScopeLoader{Path: run.LdscopePath()}

	stdoutStderr, err := sL.UnserviceHost()
	if err == nil {
		log.Info().
			Msgf("Unservice %v host success.")
	} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
		if ee.ExitCode() == 1 {
			log.Warn().
				Err(err).
				Str("loaderDetails", stdoutStderr).
				Msgf("Unservice %v host failed.")
		} else {
			log.Warn().
				Str("loaderDetails", stdoutStderr).
				Msgf("Unervice %v host failed.")
		}
	}
	return nil
}

// for all containers
func stopServiceContainers() error {
	ld := loader.ScopeLoader{Path: run.LdscopePath()}

	// Discover all containers
	cPids, err := util.GetDockerPids()
	if err != nil {
		switch {
		case errors.Is(err, util.ErrDockerNotAvailable):
			log.Warn().
				Msgf("Unservice containers skipped. Docker is not available")
		default:
			log.Warn().
				Err(err).
				Msg("Unservice containers failed.")
		}
		return nil
	}

	// Iterate over all containers
	for _, cPid := range cPids {
		stdoutStderr, err := ld.UnserviceContainer(cPid)
		if err == nil {
			log.Info().
				Str("pid", strconv.Itoa(cPid)).
				Msgf("Unservice container %v success.", cPid)
		} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
			if ee.ExitCode() == 1 {
				log.Warn().
					Err(err).
					Str("pid", strconv.Itoa(cPid)).
					Str("loaderDetails", stdoutStderr).
					Msgf("Unservice container %v failed.", cPid)
			} else {
				log.Warn().
					Str("pid", strconv.Itoa(cPid)).
					Str("loaderDetails", stdoutStderr).
					Msgf("Unservice container %v failed", cPid)
			}
		}
	}
	return nil
}

func Stop() error {
	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Error().
			Msg("Scope stop requires administrator privileges")
		return err
	}

	scopeDirVersion, err := getAppScopeVerDir()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Error accessing AppScope version directory")
		return err
	}

	// If the `scope stop` command is run inside a container, we should call `ldscope --stophost`
	// which will also run `scope stop` on the host
	// SCOPE_CLI_SKIP_STOP_HOST allows to run scope stop in docker environment (integration tests)
	_, skipHostCfg := os.LookupEnv("SCOPE_CLI_SKIP_STOP_HOST")
	if util.InContainer() && !skipHostCfg {
		if err := extract(scopeDirVersion); err != nil {
			return err
		}

		ld := loader.ScopeLoader{Path: filepath.Join(scopeDirVersion, "ldscope")}
		stdoutStderr, err := ld.StopHost()
		if err == nil {
			log.Info().
				Msgf("Stop host success.")
		} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
			if ee.ExitCode() == 1 {
				log.Error().
					Err(err).
					Str("loaderDetails", stdoutStderr).
					Msgf("Stop host failed.")
				return err
			} else {
				log.Error().
					Str("loaderDetails", stdoutStderr).
					Msgf("Stop host failed.")
				return err
			}
		}
		return nil
	}

	rc := run.Config{
		Verbosity: 4, // Default
	}
	if err := rc.DetachAll([]string{}, false); err != nil {
		log.Error().
			Err(err).
			Msg("Error detaching from all Scoped processes")
	}

	if err := run.CreateLdscope(); err != nil {
		log.Error().
			Err(err).
			Msg("Create ldscope failed.")
		return err
	}

	if err := stopConfigureHost(); err != nil {
		return err
	}

	if err := stopConfigureContainers(); err != nil {
		return err
	}

	if err := stopServiceHost(); err != nil {
		return err
	}

	if err := stopServiceContainers(); err != nil {
		return err
	}

	return nil
}
