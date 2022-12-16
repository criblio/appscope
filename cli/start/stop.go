package start

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"

	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

// for the host
func stopConfigureHost() error {
	ld := loader.New()

	stdoutStderr, err := ld.UnconfigureHost()
	if err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msg("Unconfigure host failed.")
	} else {
		fmt.Print(stdoutStderr)
		log.Info().
			Msg("Unconfigure host success.")
	}

	return nil
}

// for all containers
func stopConfigureContainers(cPids []int) error {
	ld := loader.New()

	// Iterate over all containers
	for _, cPid := range cPids {
		stdoutStderr, err := ld.UnconfigureContainer(cPid)
		if err != nil {
			log.Warn().
				Err(err).
				Str("pid", strconv.Itoa(cPid)).
				Str("loaderDetails", stdoutStderr).
				Msgf("Unconfigure containers failed. Container %v failed.", cPid)
		} else {
			fmt.Print(stdoutStderr)
			log.Info().
				Str("pid", strconv.Itoa(cPid)).
				Msgf("Unconfigure containers success. Container %v success.", cPid)
		}
	}

	return nil
}

// for the host
func stopServiceHost() error {
	ld := loader.New()

	stdoutStderr, err := ld.UnserviceHost()
	if err == nil {
		fmt.Print(stdoutStderr)
		log.Info().
			Msgf("Unservice host success.")
	} else if ee := (&exec.ExitError{}); errors.As(err, &ee) {
		if ee.ExitCode() == 1 {
			log.Warn().
				Err(err).
				Str("loaderDetails", stdoutStderr).
				Msgf("Unservice host failed.")
		} else {
			log.Warn().
				Str("loaderDetails", stdoutStderr).
				Msgf("Unservice host failed.")
		}
	}
	return nil
}

// for all containers
func stopServiceContainers(cPids []int) error {
	ld := loader.New()

	// Iterate over all containers
	for _, cPid := range cPids {
		stdoutStderr, err := ld.UnserviceContainer(cPid)
		if err == nil {
			fmt.Print(stdoutStderr)
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
	// Create a history directory for logs
	createWorkDir("stop")

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

	// If the `scope stop` command is run inside a container, we should call `scope -z --stophost`
	// which will also run `scope stop` on the host
	// SCOPE_CLI_SKIP_HOST allows to run scope stop in docker environment (integration tests)
	_, skipHostCfg := os.LookupEnv("SCOPE_CLI_SKIP_HOST")
	if util.InContainer() && !skipHostCfg {
		if err := extract(scopeDirVersion); err != nil {
			return err
		}

		ld := loader.New()
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
		Subprocess: true,
		Verbosity:  4, // Default
	}
	if err := rc.DetachAll([]string{}, false); err != nil {
		log.Error().
			Err(err).
			Msg("Error detaching from all Scoped processes")
	}

	// Discover all container PIDs
	cPids := getContainersPids()

	if err := stopConfigureHost(); err != nil {
		return err
	}

	if err := stopConfigureContainers(cPids); err != nil {
		return err
	}

	if err := stopServiceHost(); err != nil {
		return err
	}

	if err := stopServiceContainers(cPids); err != nil {
		return err
	}

	return nil
}
