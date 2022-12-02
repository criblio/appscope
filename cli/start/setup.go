package start

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

// get the list of PID(s) related to containers
func getContainersPids() []int {
	cPids := []int{}

	ctrDPids, err := util.GetContainerDPids()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Discover ContainerD containers failed.")
	}
	if ctrDPids != nil {
		cPids = append(cPids, ctrDPids...)
	}

	podmanPids, err := util.GetPodmanPids()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Discover Podman containers failed.")
	}
	if podmanPids != nil {
		cPids = append(cPids, podmanPids...)
	}

	lxcPids, err := util.GetLXCPids()
	if err != nil {
		switch {
		case errors.Is(err, util.ErrLXDSocketNotAvailable):
			log.Warn().
				Msgf("Discover LXC containers skipped. LXD is not available")
		default:
			log.Error().
				Err(err).
				Msg("Discover LXC containers failed.")
		}
	}
	if lxcPids != nil {
		cPids = append(cPids, lxcPids...)
	}

	return cPids
}

// extract extracts ldscope and scope to scope version directory
func extract(scopeDirVersion string) error {
	// Extract ldscope
	perms := os.FileMode(0755)
	b, err := run.Asset("build/ldscope")
	if err != nil {
		log.Error().
			Err(err).
			Msg("Error retrieving ldscope asset.")
		return err
	}

	if err = os.WriteFile(filepath.Join(scopeDirVersion, "ldscope"), b, perms); err != nil {
		log.Error().
			Err(err).
			Msgf("Error writing ldscope to %s.", scopeDirVersion)
		return err
	}

	// Copy scope
	exPath, err := os.Executable()
	if err != nil {
		log.Error().
			Err(err).
			Msgf("Error getting executable path")
		return err
	}
	if _, err := util.CopyFile(exPath, filepath.Join(scopeDirVersion, "scope"), perms); err != nil {
		if err != os.ErrExist {
			log.Error().
				Err(err).
				Msgf("Error writing scope to %s.", scopeDirVersion)
			return err
		}
	}

	return nil
}

// createFilterFile creates a filter file in /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
// It returns the filter file and status
func createFilterFile() (*os.File, error) {
	// Try to create AppScope directory in /usr/lib if it not exists yet
	if _, err := os.Stat("/usr/lib/appscope/"); os.IsNotExist(err) {
		os.MkdirAll("/usr/lib/appscope/", 0755)
		if err := os.Chmod("/usr/lib/appscope/", 0755); err != nil {
			return nil, err
		}
	}

	f, err := os.OpenFile("/usr/lib/appscope/scope_filter", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	if err != nil {
		if _, err := os.Stat("/tmp/appscope"); os.IsNotExist(err) {
			os.MkdirAll("/tmp/appscope", 0777)
			if err := os.Chmod("/tmp/appscope/", 0777); err != nil {
				return f, err
			}
		}
		f, err = os.OpenFile("/tmp/appscope/scope_filter", os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0755)
	}
	return f, err
}

// getAppScopeVerDir return the directory path which will be used for handling ldscope and scope files
// /usr/lib/appscope/<version>/
// or
// /tmp/appscope/<version>/
func getAppScopeVerDir() (string, error) {
	version := internal.GetNormalizedVersion()
	var appscopeVersionPath string

	// Check /usr/lib/appscope only for official version
	if !internal.IsVersionDev() {
		appscopeVersionPath = filepath.Join("/usr/lib/appscope/", version)
		if _, err := os.Stat(appscopeVersionPath); err != nil {
			if err := os.MkdirAll(appscopeVersionPath, 0755); err != nil {
				return appscopeVersionPath, err
			}
			err := os.Chmod(appscopeVersionPath, 0755)
			return appscopeVersionPath, err
		}
		return appscopeVersionPath, nil
	}

	appscopeVersionPath = filepath.Join("/tmp/appscope/", version)
	if _, err := os.Stat(appscopeVersionPath); err != nil {
		if err := os.MkdirAll(appscopeVersionPath, 0777); err != nil {
			return appscopeVersionPath, err
		}
		err := os.Chmod(appscopeVersionPath, 0777)
		return appscopeVersionPath, err
	}
	return appscopeVersionPath, nil

}

// extractFilterFile creates a filter file in /usr/lib/appscope/scope_filter or /tmp/appscope/scope_filter
// It returns the filter file name and status
func extractFilterFile(cfgData []byte) (string, error) {
	f, err := createFilterFile()
	if err != nil {
		log.Error().
			Err(err).
			Msg("Failed to create filter file.")
		return "", err
	}
	defer f.Close()

	if _, err = f.Write(cfgData); err != nil {
		log.Error().
			Err(err).
			Msg("Failed to write to filter file.")
		return "", err
	}
	return f.Name(), nil
}

func createWorkDir(cmd string) {
	// Directories named CMD_SESSIONID_PID_TIMESTAMP
	ts := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	pid := strconv.Itoa(os.Getpid())
	sessionID := run.GetSessionID()
	tmpDirName := path.Base(cmd + "_" + sessionID + "_" + pid + "_" + ts)

	// Create History directory
	histDir := run.HistoryDir()
	err := os.MkdirAll(histDir, 0755)
	util.CheckErrSprintf(err, "error creating history dir: %v", err)

	// Sudo user warning
	if _, present := os.LookupEnv("SUDO_USER"); present {
		fmt.Printf("WARNING: Session logs will be stored in %s and owned by root\n", histDir)
	}

	// Create working directory in history/
	WorkDir := filepath.Join(run.HistoryDir(), tmpDirName)
	err = os.Mkdir(WorkDir, 0755)
	util.CheckErrSprintf(err, "error creating workdir dir: %v", err)

	// Populate working directory
	// Create Log file
	filePerms := os.FileMode(0644)
	internal.CreateLogFile(filepath.Join(WorkDir, "scope.log"), filePerms)
	internal.SetDebug()
}
