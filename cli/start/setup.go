package start

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
)

func createWorkDir() {
	// Directories named CMD_SESSIONID_PID_TIMESTAMP
	ts := strconv.FormatInt(time.Now().UTC().UnixNano(), 10)
	pid := strconv.Itoa(os.Getpid())
	sessionID := run.GetSessionID()
	tmpDirName := path.Base("start" + "_" + sessionID + "_" + pid + "_" + ts)

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
