package filter

import (
	"fmt"
	"io/ioutil"
	"os"

	"github.com/criblio/scope/libscope"
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
func Add(addProc string) error {
	/*
		   check the library is installed. if not, install it.
		   		requires a namespace switch so use existing loader command
				loader command: scope install

		   add proc to the filter file - create a filter file if it does not exist.
			    new loader command for this: maybe a scope --ldfilter <newfile>
			    add an entry to the filter file. if process is already in the filter list, update the entry.
				requires a namespace switch so need a loader command
		        maybe we read it here, manipulate it with the cli yaml library, and then provide a file to write back

		   set ld.so.preload if it is not already set.
		   		new loader command for this?
				scope --ldpreload true (--rootdir x)

		   perform a scope update to all matching processes that are already scoped
		   perform a scope attach to all matching processes that are not already scoped
		   perform a scope detach to all processes that do not match anything in the filter file.
				read in the filter file, read all process names to a list
				perform a scope ps, read all process names to a list
				create a list of everything in scope ps thats not in filter file list
					perform a detach for each of these (pass rootdir)
				create a list of everything that matches addProc that is already scoped
					perform an update for each of these (pass rootdir)
				create a list of everything that matches addProc that is not already scoped
					perform an attach for each of these (pass rootdir)


	*/
	return nil
}

// Remove a process from the scope filter
func Remove(remProc string) error {

	/*
		remove proc from the filter file
		    new loader command for this: maybe a scope --ldfilter <newfile>
			requires a namespace switch so need a loader command to modify this file
			maybe we read it here, manipulate it with the cli yaml library, and then provide a file to write back

		--- cli work ---
		detach from all matching procs
			rc.Detach (remProc) // ensure you catch all of them

		perform a scope detach to all processes that do not match anything in the filter file.
			read in the filter file, read all process names to a list
				if no matches, err
			perform a scope ps, read all process names to a list
			create a list of everything in scope ps thats not in filter file list
			perform a detach for each of these

		if it's the last entry in the filter file
			unset ld.so.preload
				new loader command for this?
				scope --ldpreload false (--rootdir x)

			remove the empty filter file
				requires change namespace so use loader command:
				loader command: scope stop --rootdir hostfs

	*/

	return nil
}

//if rc.Rootdir != "" {
//	util.Warn("Attaching to process %d", pid)
//	util.Warn("It can take up to 1 minute to attach to a process in a parent namespace")
//}
