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
	files := []string{"ldscope", "libscope.so", "scope.yml"}
	perms := []os.FileMode{0755, 0755, 0644}
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

// setupWorkDir sets up a working directory for a given set of args
func (rc *Config) setupWorkDir(args []string, attach bool) {

	// Override to CriblDest if specified
	if rc.CriblDest != "" {
		rc.EventsDest = rc.CriblDest
		rc.MetricsDest = rc.CriblDest
	}

	// Create session directory
	rc.createWorkDir(args, attach)

	// Build or load config
	if rc.UserConfig == "" {
		err := rc.configFromRunOpts()
		util.CheckErrSprintf(err, "%v", err)
	} else {
		err := rc.ConfigFromFile()
		util.CheckErrSprintf(err, "%v", err)
	}

	// Update paths to absolute for file transports
	if rc.sc.Metric.Transport.TransportType == "file" {
		newPath, err := filepath.Abs(rc.sc.Metric.Transport.Path)
		util.CheckErrSprintf(err, "error getting absolute path for %s: %v", rc.sc.Metric.Transport.Path, err)
		rc.sc.Metric.Transport.Path = newPath
	}
	if rc.sc.Event.Transport.TransportType == "file" {
		newPath, err := filepath.Abs(rc.sc.Event.Transport.Path)
		util.CheckErrSprintf(err, "error getting absolute path for %s: %v", rc.sc.Event.Transport.Path, err)
		rc.sc.Event.Transport.Path = newPath
	}

	// Populate session directory
	rc.populateWorkDir(args, attach)
}

// createWorkDir creates a working directory
func (rc *Config) createWorkDir(args []string, attach bool) {
	dirPerms := os.FileMode(0755)
	if attach {
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
	tmpDirName := path.Base(args[0]) + "_" + sessionID + "_" + pid + "_" + ts

	// Create History directory
	histDir := HistoryDir()
	err := os.MkdirAll(histDir, 0755)
	util.CheckErrSprintf(err, "error creating history dir: %v", err)

	// Sudo user warning
	if _, present := os.LookupEnv("SUDO_USER"); present {
		fmt.Printf("WARNING: Session history will be stored in %s and owned by root\n", histDir)
	}

	// Create Working directory
	if attach {
		// Validate /tmp exists
		if !util.CheckDirExists("/tmp") {
			util.ErrAndExit("/tmp directory does not exist")
		}

		// Create working directory in /tmp (0777 permissions)
		rc.WorkDir = filepath.Join("/tmp", tmpDirName)
		err := os.Mkdir(rc.WorkDir, dirPerms)
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

	// Create Cmd directory
	cmdDir := filepath.Join(rc.WorkDir, "cmd")
	err = os.Mkdir(cmdDir, dirPerms)
	util.CheckErrSprintf(err, "error creating cmd dir: %v", err)

	// Create Payloads directory
	payloadsDir := filepath.Join(rc.WorkDir, "payloads")
	err = os.MkdirAll(payloadsDir, dirPerms)
	util.CheckErrSprintf(err, "error creating payloads dir: %v", err)

	log.Info().Str("workDir", rc.WorkDir).Msg("created working directory")
}

func (rc *Config) populateWorkDir(args []string, attach bool) {
	filePerms := os.FileMode(0644)
	if attach {
		filePerms = 0777
		oldmask := syscall.Umask(0)
		defer syscall.Umask(oldmask)
	}

	// Create Log file
	internal.CreateLogFile(filepath.Join(rc.WorkDir, "scope.log"), filePerms)

	// Create Metrics file
	if rc.sc.Metric.Transport.TransportType == "file" {
		f, err := os.OpenFile(rc.sc.Metric.Transport.Path, os.O_CREATE, filePerms)
		if err != nil && !os.IsExist(err) {
			util.ErrAndExit("cannot create metric file %s: %v", rc.sc.Metric.Transport.Path, err)
		}
		f.Close()
	}

	// Create Events file
	if rc.sc.Event.Transport.TransportType == "file" {
		f, err := os.OpenFile(rc.sc.Event.Transport.Path, os.O_CREATE, filePerms)
		if err != nil && !os.IsExist(err) {
			util.ErrAndExit("cannot create metric file %s: %v", rc.sc.Event.Transport.Path, err)
		}
		f.Close()
	}

	// Create event_dest file
	err := ioutil.WriteFile(filepath.Join(rc.WorkDir, "event_dest"), []byte(rc.buildEventsDest()), filePerms)
	util.CheckErrSprintf(err, "error writing event_dest: %v", err)

	// Create metrics_dest file
	err = ioutil.WriteFile(filepath.Join(rc.WorkDir, "metric_dest"), []byte(rc.buildMetricsDest()), filePerms)
	util.CheckErrSprintf(err, "error writing metric_dest: %v", err)

	// Create metrics_format file
	if rc.MetricsFormat != "" {
		err = ioutil.WriteFile(filepath.Join(rc.WorkDir, "metric_format"), []byte(rc.MetricsFormat), filePerms)
		util.CheckErrSprintf(err, "error writing metric_format: %v", err)
	}

	// Create config file
	scYamlPath := filepath.Join(rc.WorkDir, "scope.yml")
	if rc.UserConfig == "" {
		err := rc.WriteScopeConfig(scYamlPath, filePerms)
		util.CheckErrSprintf(err, "%v", err)
	} else {
		input, err := ioutil.ReadFile(rc.UserConfig)
		if err != nil {
			util.ErrAndExit("cannot read file %s: %v", rc.UserConfig, err)
		}
		if err = ioutil.WriteFile(scYamlPath, input, 0644); err != nil {
			util.ErrAndExit("failed to write file to %s: %v", scYamlPath, err)
		}
	}

	// Create args.json file
	argsJSONPath := filepath.Join(rc.WorkDir, "args.json")
	argsBytes, err := json.Marshal(args)
	util.CheckErrSprintf(err, "error marshaling JSON: %v", err)
	err = ioutil.WriteFile(argsJSONPath, argsBytes, filePerms)
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

func (rc *Config) buildMetricsDest() string {
	var dest string
	if rc.sc.Cribl.Enable == "true" {
		if rc.sc.Cribl.Transport.TransportType == "unix" || rc.sc.Cribl.Transport.TransportType == "file" {
			dest = rc.sc.Cribl.Transport.TransportType + "://" + rc.sc.Cribl.Transport.Path
		} else {
			dest = rc.sc.Cribl.Transport.TransportType + "://" + rc.sc.Cribl.Transport.Host + ":" + fmt.Sprint(rc.sc.Cribl.Transport.Port)
		}
	} else {
		if rc.sc.Metric.Transport.TransportType == "unix" || rc.sc.Metric.Transport.TransportType == "file" {
			dest = rc.sc.Metric.Transport.TransportType + "://" + rc.sc.Metric.Transport.Path
		} else {
			dest = rc.sc.Metric.Transport.TransportType + "://" + rc.sc.Metric.Transport.Host + ":" + fmt.Sprint(rc.sc.Metric.Transport.Port)
		}
	}
	return dest
}

func (rc *Config) buildEventsDest() string {
	var dest string
	if rc.sc.Cribl.Enable == "true" {
		if rc.sc.Cribl.Transport.TransportType == "unix" || rc.sc.Cribl.Transport.TransportType == "file" {
			dest = rc.sc.Cribl.Transport.TransportType + "://" + rc.sc.Cribl.Transport.Path
		} else {
			dest = rc.sc.Cribl.Transport.TransportType + "://" + rc.sc.Cribl.Transport.Host + ":" + fmt.Sprint(rc.sc.Cribl.Transport.Port)
		}
	} else {
		if rc.sc.Event.Transport.TransportType == "unix" || rc.sc.Event.Transport.TransportType == "file" {
			dest = rc.sc.Event.Transport.TransportType + "://" + rc.sc.Event.Transport.Path
		} else {
			dest = rc.sc.Event.Transport.TransportType + "://" + rc.sc.Event.Transport.Host + ":" + fmt.Sprint(rc.sc.Event.Transport.Port)
		}
	}
	return dest
}
