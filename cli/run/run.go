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

// Config represents options to change how we run scope
type Config struct {
	Passthrough bool
	Verbosity   int
	Payloads    bool

	workDir string
	now     func() time.Time
}

// Run executes a scoped command
func (rc *Config) Run(args []string) {
	if err := CreateLdscope(); err != nil {
		util.ErrAndExit("error creating ldscope: %v", err)
	}
	// Normal operational, not passthrough, create directory for this run
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	env := os.Environ()
	if !rc.Passthrough {
		rc.SetupWorkDir(args)
		env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.workDir, "scope.yml"))
		log.Info().Bool("passthrough", rc.Passthrough).Strs("args", args).Msg("calling syscall.Exec")
	}
	syscall.Exec(ldscopePath(), append([]string{"ldscope"}, args...), env)
}

func ldscopePath() string {
	return filepath.Join(util.ScopeHome(), "ldscope")
}

// CreateLdscope creates ldscope in $SCOPE_HOME
func CreateLdscope() error {
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

// CreateWorkDir creates a working directory
func (rc *Config) CreateWorkDir(cmd string) {
	if rc.workDir != "" {
		if util.CheckDirExists(rc.workDir) {
			// Directory exists, exit
			return
		}
	}
	if rc.now == nil {
		rc.now = time.Now
	}
	ts := strconv.FormatInt(rc.now().UTC().UnixNano(), 10)
	pid := strconv.Itoa(os.Getpid())
	histDir := HistoryDir()
	err := os.MkdirAll(histDir, 0755)
	util.CheckErrSprintf(err, "error creating history dir: %v", err)
	sessionID := getSessionID()
	// Directories named CMD_SESSIONID_PID_TIMESTAMP
	tmpDirName := path.Base(cmd) + "_" + sessionID + "_" + pid + "_" + ts
	rc.workDir = filepath.Join(HistoryDir(), tmpDirName)
	err = os.Mkdir(rc.workDir, 0755)
	util.CheckErrSprintf(err, "error creating workdir dir: %v", err)
	cmdDir := filepath.Join(rc.workDir, "cmd")
	err = os.Mkdir(cmdDir, 0755)
	util.CheckErrSprintf(err, "error creating cmd dir: %v", err)
	payloadsDir := filepath.Join(rc.workDir, "payloads")
	err = os.MkdirAll(payloadsDir, 0755)
	util.CheckErrSprintf(err, "error creating payloads dir: %v", err)
	internal.SetLogFile(filepath.Join(rc.workDir, "scope.log"))
	log.Info().Str("workDir", rc.workDir).Msg("created working directory")
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

// SetupWorkDir sets up a working directory for a given set of args
func (rc *Config) SetupWorkDir(args []string) {
	cmd := path.Base(args[0])
	rc.CreateWorkDir(cmd)

	rc.SetupScopeYaml()

	argsJSONPath := filepath.Join(rc.workDir, "args.json")
	argsBytes, err := json.Marshal(args)
	util.CheckErrSprintf(err, "error marshaling JSON: %v", err)
	err = ioutil.WriteFile(argsJSONPath, argsBytes, 0644)
}

// SetupScopeYaml writes the scope configuration to the workdir
func (rc *Config) SetupScopeYaml() {
	sc := GetDefaultScopeConfig(rc.workDir)

	if rc.Verbosity != 0 {
		sc.Metric.Format.Verbosity = rc.Verbosity
	}

	if rc.Payloads {
		sc.Payload = ScopePayloadConfig{
			Enable: true,
			Dir:    filepath.Join(rc.workDir, "payloads"),
		}
	}

	scYamlPath := filepath.Join(rc.workDir, "scope.yml")
	err := WriteScopeConfig(sc, scYamlPath)
	util.CheckErrSprintf(err, "%v", err)
}
