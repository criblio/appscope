package start

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

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

	return nil
}
