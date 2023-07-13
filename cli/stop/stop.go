package stop

import (
	"errors"
	"os"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v2"
)

var (
	errDetachingMultiple = errors.New("at least one error found when detaching from more than 1 process. See logs")
)

func Stop(rc *run.Config) error {
	// Create a history directory for logs
	rc.CreateWorkDirBasic("stop")

	// Validate user has root permissions
	if err := util.UserVerifyRootPerm(); err != nil {
		log.Error().
			Msg("Scope stop requires administrator privileges")
		return err
	}

	// Instantiate the loader
	ld := loader.New()

	////////////////////////////////////////////
	// Empty the global rules file
	////////////////////////////////////////////

	var rulesFile libscope.Rules

	// Write the rules contents to a temporary path
	rulesFilePath := "/tmp/scope_rules"
	file, err := os.OpenFile(rulesFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		log.Warn().Err(err).Msgf("Error creating/opening %s", rulesFilePath)
		return err
	}
	defer file.Close()

	data, err := yaml.Marshal(&rulesFile)
	if err != nil {
		util.Warn("Error marshaling YAML:%v", err)
		return err
	}

	_, err = file.Write(data)
	if err != nil {
		util.Warn("Error writing to file:%v", err)
		return err
	}

	// Ask the loader to update the rules file
	if stdoutStderr, err := ld.Rules(rulesFilePath, rc.Rootdir); err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msgf("Install library in %s namespace failed.", rc.Rootdir)
		return err
	}

	////////////////////////////////////////////
	// Unset ld.so.preload if it is set.
	////////////////////////////////////////////

	if stdoutStderr, err := ld.Preload("off", rc.Rootdir); err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msgf("Install library in %s namespace failed.", rc.Rootdir)
		return err
	}

	////////////////////////////////////////////
	// Detach from all scoped processes
	////////////////////////////////////////////

	procs, err := util.HandleInputArg("", "", rc.Rootdir, false, false, false, false)
	if err != nil {
		return err
	}
	// Note: if len(procs) == 0 do nothing
	if len(procs) == 1 {
		if err = rc.Detach(procs[0].Pid); err != nil {
			return err
		}
	} else if len(procs) > 1 {
		errors := false
		for _, proc := range procs {
			if err = rc.Detach(proc.Pid); err != nil {
				log.Error().Err(err)
				errors = true
			}
		}
		if errors {
			return errDetachingMultiple
		}
	}

	return nil
}
