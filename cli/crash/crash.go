package crash

import "github.com/rs/zerolog/log"

// GenFiles creates the following files for a given pid
// - snapshot
// - info (where available)
// - coredump (where available)
// - cfg (where available)
// - backtrace (where available)
func GenFiles(pid, sig, errno uint32) error {

	// where are we putting this stuff
	// if session directory exists, write to sessiondir/snapshot/
	// if not, write to /tmp/appscope/pid/
	dir := "/tmp"

	// retrieve info file

	// retrieve coredump file

	// retrieve cfg file

	// retrieve backtrace file

	// create snapshot file
	if err := GenSnapshotFile(pid, dir); err != nil {
		log.Error().Err(err).Msgf("error generating snapshot file")
		return err
	}

	return nil
}

// GenSnapshotFile generates the snapshot file for a given pid
func GenSnapshotFile(pid uint32, dir string) error {

	//Debug Information 	MVP
	//Time of Snapshot 	cli
	//AppScope Lib Version 	lib
	//AppScope Cli Version 	cli
	//Process Name 	cli
	//Process Arguments 	cli
	//PID, PPID 	cli
	//User ID/ Group ID 	cli
	//Username/ Groupname 	cli
	//
	//AppScope configuration 	lib
	//Environment Variables 	cli
	//
	//Signal number 	cli (eBPF)
	//Signal handler 	cli (eBPF)
	//Error number 	cli (eBPF)
	//
	//Machine Arch 	cli
	//Distro, Distro version 	cli
	//Kernel version 	cli
	//Hostname 	cli
	//Namespace Id's 	cli
	//Container Impl (docker, podman...) 	-
	//Container Name/Version(?) 	-
	//SELinux or AppArmor enforcing? 	-
	//Unix Capabilities... PTRACE?... 	-
	//
	//AppScope Log Output 	if possible
	//scope ps output 	cli
	//scope history output 	cli
	//
	//Backtrace (offending thread) 	if possible
	//Backtraces (all threads) 	lib
	//Memory (stacks, heap) 	lib
	//Registers 	lib
	//Application and .so elf files 	-
	//
	//Application version(?) 	-
	//JRE version (if java) 	cli
	//Go version (if go) 	cli
	//Static or Dynamically linked(?) 	cli
	//
	//Network interface status? 	-
	//ownership/permissions on pertinent files/unix sockets 	-
	//dns on pertinent host names 	-

	//write file to dir

	return nil
}
