package start

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

var (
	errRemovingFiles = errors.New("error removing start files. See logs for details")
)

// remove /etc/profile.d/scope.sh
// remove /usr/lib/appscope/scope_filter
// remove /tmp/appscope/scope_filter
func removeStartFiles() error {

	errors := false

	if err := os.Remove("/etc/profile.d/scope.sh"); err != nil {
		log.Error().
			Err(err).
			Msg("Error removing /etc/profile.d/scope.sh")
		errors = true
	}

	if err := os.Remove("/usr/lib/appscope/scope_filter"); err != nil {
		log.Error().
			Err(err).
			Msg("Error removing /usr/lib/appscope/scope_filter")
		errors = true
	}

	if err := os.Remove("/tmp/appscope/scope_filter"); err != nil {
		log.Error().
			Err(err).
			Msg("Error removing /tmp/appscope/scope_filter")
		errors = true
	}

	if errors {
		return errRemovingFiles
	}
	return nil
}

// update service configs to remove LD_PRELOAD of libscope
func updateServiceConfigs() error {

	return nil
}

func Stop() error {
	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Error().
			Msg("Scope stop requires administrator privileges")
		return err
	}

	if err := removeStartFiles(); err != nil {
		log.Error().
			Err(err).
			Msg("Error removing start files")
	}

	if err := updateServiceConfigs(); err != nil {
		log.Error().
			Err(err).
			Msg("Error updating service configurations")
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

	return nil
}
