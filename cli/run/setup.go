package run

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

func ldscopePath() string {
	return filepath.Join(util.ScopeHome(), "ldscope")
}

// createLdscope creates ldscope in $SCOPE_HOME
func createLdscope() error {
	ldscope := ldscopePath()
	ldscopeInfo, _ := AssetInfo("build/ldscope")

	// If it exists, don't create
	if stat, err := os.Stat(ldscope); err == nil {
		if stat.ModTime() == ldscopeInfo.ModTime() {
			return nil
		}
	}

	b, err := Asset("build/ldscope")
	if err != nil {
		return err
	}

	err = os.MkdirAll(util.ScopeHome(), 0755)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(ldscope, b, 0755)
	if err != nil {
		return err
	}
	err = os.Chtimes(ldscope, ldscopeInfo.ModTime(), ldscopeInfo.ModTime())
	if err != nil {
		return err
	}
	log.Info().Str("ldscope", ldscope).Msg("created ldscope")
	return nil
}

func environNoScope() []string {
	env := []string{}
	for _, envVar := range os.Environ() {
		if !strings.HasPrefix(envVar, "SCOPE") {
			env = append(env, envVar)
		}
	}
	return env
}

// CreateAll outputs all bundled files to a given path
func CreateAll(path string) error {
	files := []string{"ldscope", "libscope.so", "scope.yml", "scope_protocol.yml"}
	perms := []os.FileMode{0755, 0755, 0644, 0644}
	for i, f := range files {
		b, err := Asset(fmt.Sprintf("build/%s", f))
		if err != nil {
			return err
		}
		err = ioutil.WriteFile(filepath.Join(path, f), b, perms[i])
		if err != nil {
			return err
		}
	}
	return nil
}

// createWorkDir creates a working directory
func (rc *Config) createWorkDir(cmd string, attach bool) {
	filePerms := os.FileMode(0644)
	dirPerms := os.FileMode(0755)
	if attach {
		filePerms = 0777
		dirPerms = 0777
		oldmask := syscall.Umask(0)
		defer syscall.Umask(oldmask)
	}
	if rc.WorkDir != "" {
		if util.CheckDirExists(rc.WorkDir) {
			// Directory exists, exit
			return
		}
	}
	if rc.now == nil {
		rc.now = time.Now
	}

	// Directories named CMD_SESSIONID_PID_TIMESTAMP
	ts := strconv.FormatInt(rc.now().UTC().UnixNano(), 10)
	pid := strconv.Itoa(os.Getpid())
	sessionID := getSessionID()
	tmpDirName := path.Base(cmd) + "_" + sessionID + "_" + pid + "_" + ts

	// History directory
	histDir := HistoryDir()
	err := os.MkdirAll(histDir, 0755)
	util.CheckErrSprintf(err, "error creating history dir: %v", err)

	// Working directory
	if attach {
		// Validate /tmp exists
		if !util.CheckDirExists("/tmp") {
			util.ErrAndExit("/tmp directory does not exist")
		}

		// Create working directory in /tmp (0777 permissions)
		rc.WorkDir = filepath.Join("/tmp", tmpDirName)
		err := os.Mkdir(rc.WorkDir, filePerms)
		util.CheckErrSprintf(err, "error creating workdir dir: %v", err)

		// Symbolic link between /tmp/tmpDirName and /history/tmpDirName
		rootHistDir := filepath.Join(histDir, tmpDirName)
		os.Symlink(rc.WorkDir, rootHistDir)
	} else {
		// Create working directory in history/
		rc.WorkDir = filepath.Join(HistoryDir(), tmpDirName)
		err = os.Mkdir(rc.WorkDir, 0755)
		util.CheckErrSprintf(err, "error creating workdir dir: %v", err)
	}

	// Cmd directory
	cmdDir := filepath.Join(rc.WorkDir, "cmd")
	err = os.Mkdir(cmdDir, dirPerms)
	util.CheckErrSprintf(err, "error creating cmd dir: %v", err)

	// Payloads directory
	payloadsDir := filepath.Join(rc.WorkDir, "payloads")
	err = os.MkdirAll(payloadsDir, dirPerms)
	util.CheckErrSprintf(err, "error creating payloads dir: %v", err)

	// Log file
	internal.CreateLogFile(filepath.Join(rc.WorkDir, "scope.log"), filePerms)

	// Metrics file
	if rc.MetricsDest != "" || rc.CriblDest != "" {
		metricsDest := rc.MetricsDest
		if metricsDest == "" {
			metricsDest = rc.CriblDest
		}
		err = ioutil.WriteFile(filepath.Join(rc.WorkDir, "metric_dest"), []byte(metricsDest), filePerms)
		util.CheckErrSprintf(err, "error writing metric_dest: %v", err)
	}
	if rc.MetricsFormat != "" {
		err = ioutil.WriteFile(filepath.Join(rc.WorkDir, "metric_format"), []byte(rc.MetricsFormat), filePerms)
		util.CheckErrSprintf(err, "error writing metric_format: %v", err)
	}

	// Events file
	if rc.EventsDest != "" || rc.CriblDest != "" {
		eventsDest := rc.EventsDest
		if eventsDest == "" {
			eventsDest = rc.CriblDest
		}
		err = ioutil.WriteFile(filepath.Join(rc.WorkDir, "event_dest"), []byte(eventsDest), filePerms)
		util.CheckErrSprintf(err, "error writing event_dest: %v", err)
	}

	log.Info().Str("workDir", rc.WorkDir).Msg("created working directory")
}

// Open to race conditions, but having a duplicate ID is only a UX bug rather than breaking
// so keeping it simple and avoiding locking etc
func getSessionID() string {
	countFile := filepath.Join(util.ScopeHome(), "count")
	lastSessionID := 0
	lastSessionBytes, err := ioutil.ReadFile(countFile)
	if err == nil {
		lastSessionID, err = strconv.Atoi(string(lastSessionBytes))
		if err != nil {
			lastSessionID = 0
		}
	}
	sessionID := strconv.Itoa(lastSessionID + 1)
	_ = ioutil.WriteFile(countFile, []byte(sessionID), 0644)
	return sessionID
}

// HistoryDir returns the history directory
func HistoryDir() string {
	return filepath.Join(util.ScopeHome(), "history")
}

// setupWorkDir sets up a working directory for a given set of args
func (rc *Config) setupWorkDir(args []string, attach bool) {
	filePerms := os.FileMode(0644)
	if attach {
		filePerms = 0777
		oldmask := syscall.Umask(0)
		defer syscall.Umask(oldmask)
	}

	cmd := path.Base(args[0])
	rc.createWorkDir(cmd, attach)

	err := rc.configFromRunOpts()
	util.CheckErrSprintf(err, "%v", err)

	if rc.sc.Metric.Transport.TransportType == "file" {
		newPath, err := filepath.Abs(rc.sc.Metric.Transport.Path)
		util.CheckErrSprintf(err, "error getting absolute path for %s: %v", rc.sc.Metric.Transport.Path, err)
		rc.sc.Metric.Transport.Path = newPath
		f, err := os.OpenFile(rc.sc.Metric.Transport.Path, os.O_CREATE, filePerms)
		if err != nil && !os.IsExist(err) {
			util.ErrAndExit("cannot create metric file %s: %v", rc.sc.Metric.Transport.Path, err)
		}
		f.Close()
	}

	if rc.sc.Event.Transport.TransportType == "file" {
		newPath, err := filepath.Abs(rc.sc.Event.Transport.Path)
		util.CheckErrSprintf(err, "error getting absolute path for %s: %v", rc.sc.Event.Transport.Path, err)
		rc.sc.Event.Transport.Path = newPath
		f, err := os.OpenFile(rc.sc.Event.Transport.Path, os.O_CREATE, filePerms)
		if err != nil && !os.IsExist(err) {
			util.ErrAndExit("cannot create metric file %s: %v", rc.sc.Event.Transport.Path, err)
		}
		f.Close()
	}

	scYamlPath := filepath.Join(rc.WorkDir, "scope.yml")
	err = rc.WriteScopeConfig(scYamlPath, filePerms)
	util.CheckErrSprintf(err, "%v", err)

	argsJSONPath := filepath.Join(rc.WorkDir, "args.json")
	argsBytes, err := json.Marshal(args)
	util.CheckErrSprintf(err, "error marshaling JSON: %v", err)
	err = ioutil.WriteFile(argsJSONPath, argsBytes, filePerms)
}
