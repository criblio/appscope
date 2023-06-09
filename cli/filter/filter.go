package filter

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
	"gopkg.in/yaml.v2"
)

var (
	errNoScopedProcs     = errors.New("no scoped processes found")
	errDetachingMultiple = errors.New("at least one error found when detaching from more than 1 process. See logs")
	errAttachingMultiple = errors.New("at least one error found when attaching to more than 1 process. See logs")
)

func Retrieve(rootdir string) ([]byte, libscope.Filter, error) {
	filePath := "/usr/lib/appscope/scope_filter"
	if rootdir != "" {
		filePath = rootdir + filePath
	}

	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, nil, err
	}

	// Read yaml in
	var filterFile libscope.Filter
	err = yaml.Unmarshal(data, &filterFile)
	if err != nil {
		fmt.Println("Error unmarshaling YAML:", err)
		return nil, nil, err
	}

	return data, filterFile, nil
}

// Add a process to the scope filter
func Add(filterFile libscope.Filter, addProc, rootdir string, rc *run.Config) error {

	// Instantiate the loader
	ld := loader.New()

	// Install libscope if not already installed
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

	// Modify the filter contents
	newEntry := libscope.Entry{
		ProcName: addProc,
		ProcArg:  "",
		// TODO Configuration:
	}
	filterFile["allow"] = append(filterFile["allow"], newEntry)

	// Write the filter contents to a temporary path
	// TODO write to a temporary file. give it a unique id
	filterFilePath := "/tmp/filter"

	// Update the global filter file
	if stdoutStderr, err := ld.Filter(filterFilePath, rc.Rootdir); err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msgf("Install library in %s namespace failed.", rootdir)
		return err
	}

	// Set ld.so.preload if it is not already set.
	if stdoutStderr, err := ld.SetPreload(true, rc.Rootdir); err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msgf("Install library in %s namespace failed.", rootdir)
		return err
	}

	// Perform a scope attach to all matching processes that are not already scoped (will re-attach to update existing procs)
	if rootdir != "" {
		util.Warn("It can take up to 1 minute to attach to processes in a parent namespace")
	}

	rc.Subprocess = true
	procs, err := util.HandleInputArg(addProc, rc.Rootdir, false, false, true, true)
	if err != nil {
		return err
	}
	// Note: if len(procs) == 0 do nothing
	if len(procs) == 1 {
		if err = rc.Attach(procs[0].Pid); err != nil {
			return err
		}
	} else if len(procs) > 1 {
		errors := false
		for _, proc := range procs {
			if err = rc.Attach(proc.Pid); err != nil {
				log.Error().Err(err)
				errors = true
			}
		}
		if errors {
			return errAttachingMultiple
		}
	}

	// TBC ? perform a scope detach to all processes that do not match anything in the filter file

	return nil
}

// Remove a process from the scope filter
func Remove(filterFile libscope.Filter, remProc, rootdir string, rc *run.Config) error {

	// Instantiate the loader
	ld := loader.New()

	// Modify the filter contents
	// TODO remove the entry from the filterFile slice

	// Write the filter contents to a temporary path
	// TODO write to a temporary file. give it a unique id
	filterFilePath := "/tmp/filter"

	// Update the global filter file
	if stdoutStderr, err := ld.Filter(filterFilePath, rc.Rootdir); err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msgf("Install library in %s namespace failed.", rootdir)
		return err
	}

	// Unset ld.so.preload if it is already set.
	if stdoutStderr, err := ld.SetPreload(false, rc.Rootdir); err != nil {
		log.Warn().
			Err(err).
			Str("loaderDetails", stdoutStderr).
			Msgf("Install library in %s namespace failed.", rootdir)
		return err
	}

	// Perform a scope detach to all matching, scoped processes
	rc.Subprocess = true
	procs, err := util.HandleInputArg(remProc, rc.Rootdir, false, false, false, true)
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
