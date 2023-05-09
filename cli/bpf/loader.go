package bpf

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/cilium/ebpf"
	"github.com/criblio/scope/pidfd"
	"golang.org/x/sys/unix"
)

type BpfLoader struct {
	pidFd pidfd.PidFd
	links []int
}

var (
	errReadProc            = errors.New("failed to read /proc")
	errMissingLoader       = errors.New("failed to find eBPF loader process")
	errMissingBpfLink      = errors.New("failed to find eBPF link object")
	errMissingMapInProgram = errors.New("failed to find eBPF map in program object")
	errMultipleMaps        = errors.New("only single eBPF map is supported")
	errMissingSigdel       = errors.New("failed to find sigdel eBPF link object")
	errNotSupportedProgram = errors.New("founded eBPF program is not supported")
)

// Create Bpf Loader object
func NewLoader(loaderName string) (BpfLoader, error) {
	var bl BpfLoader
	// Read the list of directories in /proc
	procDirs, err := os.ReadDir("/proc")
	if err != nil {
		return bl, errReadProc
	}

	// Iterate through the directories and check cmdline
	for _, dir := range procDirs {
		if dir.IsDir() {
			pid, err := strconv.Atoi(dir.Name())
			if err == nil {
				// Read the cmdline file for the process
				cmdline, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
				if err == nil {
					// Convert cmdline to string and remove null terminator
					cmdlineStr := strings.TrimSuffix(string(cmdline), "\x00")
					if strings.Contains(cmdlineStr, loaderName) {
						links, err := findeBPFLinks(pid)
						if err != nil {
							continue
						}
						pidFd, err := pidfd.Open(pid)
						if err != nil {
							continue
						}
						bl.pidFd = pidFd
						bl.links = links
						return bl, nil
					}
				}
			}
		}
	}

	return bl, errMissingLoader
}

// Terminate Bpf Loader
func (bl BpfLoader) Terminate() error {
	defer bl.pidFd.Close()
	return bl.pidFd.SendSignal(unix.SIGUSR1)
}

type validEbpfScope struct {
	ebpfProgName string
	ebpfProgType ebpf.ProgramType
	ebpfLinkType linkType
	ebpfMapType  ebpf.MapType
}

// validBPfProgram verifies if the eBPF objects are supported in AppScope
func validBPfProgram(linkInfo LinkInformation, progInfo *ebpf.ProgramInfo, mapInfo *ebpf.MapInfo) bool {
	expected := validEbpfScope{
		ebpfProgName: "sig_deliver",
		ebpfProgType: ebpf.TracePoint,
		ebpfLinkType: BPF_LINK_TYPE_PERF_EVENT,
		ebpfMapType:  ebpf.PerfEventArray,
	}

	if expected.ebpfProgName == progInfo.Name &&
		expected.ebpfProgType == progInfo.Type &&
		expected.ebpfLinkType == linkInfo.Type &&
		expected.ebpfMapType == mapInfo.Type {
		return true
	}

	return true
}

// getMapFromLink retrieves the map object associated with specified eBPF link file descriptor
func (bl BpfLoader) getMapFromLink(linkfd int) (*ebpf.Map, error) {

	var mapObj *ebpf.Map

	// Obtain eBPF link information
	linkInfo, err := LinkGetInfo(linkfd)
	if err != nil {
		return mapObj, err
	}

	// Obtain eBPF program file descriptor from id
	progFd, err := ProgFdFromId(linkInfo.ProgId)
	if err != nil {
		return mapObj, err
	}

	// Create eBPF program from file descriptor
	prog, err := ebpf.NewProgramFromFD(progFd)
	if err != nil {
		return mapObj, err
	}

	// Retrieve eBPF program information
	progInfo, err := prog.Info()
	if err != nil {
		prog.Close()
		return mapObj, err
	}

	// Retrieve eBPF map(s) Id(s)from eBPF program information
	maps, mapPresent := progInfo.MapIDs()
	if !mapPresent {
		prog.Close()
		return mapObj, errMissingMapInProgram
	}

	// TODO: fix me
	if len(maps) != 1 {
		prog.Close()
		return mapObj, errMultipleMaps
	}

	// Create eBPF map from eBPF map Id
	mapObj, err = ebpf.NewMapFromID(maps[0])
	if err != nil {
		mapObj.Close()
		prog.Close()
		return mapObj, err
	}

	// Retrieve eBPF map information
	mapInfo, err := mapObj.Info()
	if err != nil {
		mapObj.Close()
		prog.Close()
		return mapObj, err
	}

	res := validBPfProgram(linkInfo, progInfo, mapInfo)
	if !res {
		mapObj.Close()
		prog.Close()
		return nil, errNotSupportedProgram
	}

	return mapObj, nil
}

// Sigdel object
func (bl BpfLoader) NewSigdel() (SigDel, error) {
	var sd SigDel

	for _, linkId := range bl.links {
		// Copy the link file descriptor from loader
		dupLinkFd, err := bl.pidFd.GetFd(linkId)
		if err != nil {
			continue
		}
		newMap, err := bl.getMapFromLink(dupLinkFd)
		if err == nil {
			sd.bpfLinkFd = dupLinkFd
			sd.bpfMap = newMap
			return sd, nil
		}
		unix.Close(dupLinkFd)
	}

	// How to handle this one found all found nothing found?
	return sd, errMissingSigdel
}

// findeBPFLinks retrieves list of eBPF link objects in specified process described by targetPid
func findeBPFLinks(targetPid int) ([]int, error) {
	linkFds := make([]int, 0)

	// Read the files in the file descriptor directory.
	fdDir := path.Join("/proc", strconv.Itoa(targetPid), "fd")
	files, err := os.ReadDir(fdDir)
	if err != nil {
		return linkFds, fmt.Errorf("failed to read %s: %v", fdDir, err)
	}

	// Iterate over the files in the directory.
	for _, file := range files {
		// Resolve the filename to the link target.
		resolvedFileName, err := os.Readlink(filepath.Join(fdDir, file.Name()))
		if err != nil {
			fmt.Println("Readlink failed ", err)
			continue
		}

		// Check if the link target contains "bpf_link".
		if strings.Contains(resolvedFileName, "bpf_link") {
			// Convert the file name to an integer file descriptor.
			linkFd, err := strconv.Atoi(file.Name())
			if err != nil {
				continue
			}
			// Add the file descriptor to the list of eBPF link objects.
			linkFds = append(linkFds, linkFd)
		}
	}

	// Check if any eBPF links were found.
	if len(linkFds) == 0 {
		return linkFds, errMissingBpfLink
	}

	return linkFds, nil
}
