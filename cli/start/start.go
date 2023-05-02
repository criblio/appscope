package start

import (
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

// Start installs libscope
func Start(prefix string) error {
	// Create a history directory for logs
	run.CreateWorkDirBasic("start")

	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Error().
			Msg("Scope start requires administrator privileges")
		return err
	}

	// Instantiate the loader
	ld := loader.New()

	if prefix == "" {
		stdoutStderr, err := ld.Install()
		if err != nil {
			log.Warn().
				Err(err).
				Str("loaderDetails", stdoutStderr).
				Msg("Install library failed.")
		} else {
			log.Info().
				Msg("Install library success.")
		}
	} else {
		// TODO get cpid
		cpid := 1

		stdoutStderr, err := ld.InstallNamespace(cpid)
		if err != nil {
			log.Warn().
				Err(err).
				Str("loaderDetails", stdoutStderr).
				Msg("Install library failed.")
		} else {
			log.Info().
				Msg("Install library success.")
		}
	}

	return nil
}
