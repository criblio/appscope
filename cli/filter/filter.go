package filter

import (
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
	if rootdir == "" {
		stdoutStderr, err := ld.Install()
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
	} else {
		stdoutStderr, err := ld.InstallNamespace(rootdir)
		if err != nil {
			log.Warn().
				Err(err).
				Str("loaderDetails", stdoutStderr).
				Msgf("Install library in %s namespace failed.", rootdir)
			return err
		} else {
			log.Info().
				Msg("Install library success.")
		}
	}

	/*
		TODO
		add proc to the filter file - create a filter file if it does not exist.
			new loader command for this: maybe a scope --ldfilter <newfile>
			add an entry to the filter file. if process is already in the filter list, update the entry.
			requires a namespace switch so need a loader command
		    maybe we read it here, manipulate it with the cli yaml library, and then provide a file to write back

		set ld.so.preload if it is not already set.
		   	new loader command for this?
			scope --ldpreload true (--rootdir x)
	*/

	// Perform a scope attach to all matching processes that are not already scoped (will re-attach to update existing procs)
	if rootdir != "" {
		util.Warn("It can take up to 1 minute to attach to a process in a parent namespace")
	}
	if _, err := rc.Attach(addProc, false); err != nil {
		return err
	}

	// TBC ? TODO perform a scope detach to all processes that do not match anything in the filter file

	return nil
}

// Remove a process from the scope filter
func Remove(filterFile libscope.Filter, remProc, rootdir string, rc *run.Config) error {

	/*
		TODO
		remove proc from the filter file
		    new loader command for this: maybe a scope --ldfilter <newfile>
			requires a namespace switch so need a loader command to modify this file
			maybe we read it here, manipulate it with the cli yaml library, and then provide a file to write back

		if it's the last entry in the filter file
			unset ld.so.preload
				new loader command for this?
				scope --ldpreload false (--rootdir x)

			remove the empty filter file
				requires change namespace so use loader command:
				loader command: scope stop --rootdir hostfs

	*/

	// Perform a scope detach to all matching, scoped processes
	if err := rc.Detach(remProc, false, false); err != nil {
		return err
	}

	return nil
}
