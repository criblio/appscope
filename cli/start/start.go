package start

import (
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/rs/zerolog/log"
)

// Start installs libscope
func Start(rootdir string) error {
	// Create a history directory for logs
	rc := run.Config{}
	rc.CreateWorkDirBasic("start")

	// Instantiate the loader
	ld := loader.New()

	stdoutStderr, err := ld.Install(rootdir)
	if err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msg("Install library failed.")
		return err
	} else {
		log.Info().
			Msg("Install library success.")
	}

	return nil
}
