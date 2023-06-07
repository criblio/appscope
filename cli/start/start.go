package start

import (
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/rs/zerolog/log"
)

// Start installs libscope
func Start(rootdir string) error {
	// Create a history directory for logs
	run.CreateWorkDirBasic("start")

	// Instantiate the loader
	ld := loader.New()

	if rootdir == "" {
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
		stdoutStderr, err := ld.InstallNamespace(rootdir)

		if err != nil {
			log.Warn().
				Err(err).
				Str("loaderDetails", stdoutStderr).
				Msgf("Install library in %s namespace failed.", rootdir)
		} else {
			log.Info().
				Msg("Install library success.")
		}
	}

	return nil
}
