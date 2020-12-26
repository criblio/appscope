package run

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	log "github.com/criblio/scope/logger"
	"github.com/criblio/scope/util"
	"github.com/spf13/viper"
)

var now func() time.Time

func init() {
	now = time.Now
	if os.Getenv("SCOPE_TEST") != "" {
		now = func() time.Time { return time.Unix(0, 0) }
	}
}

// Run executes a scoped command
func Run(args []string) {
	if err := CreateCScope(); err != nil {
		util.ErrAndExit("error creating cscope: %v", err)
	}
	var env []string
	// Normal operational, not passthrough, create directory for this run
	// Directory contains scope.yml which is configured to output to that
	// directory and has a command directory configured in that directory.
	if !viper.GetBool("passthrough") {
		env = environNoScope()
		SetupWorkDir(args)
	} else {
		env = os.Environ()
	}
	syscall.Exec(cscopePath(), append([]string{"cscope"}, args...), env)
}

func cscopePath() string {
	return filepath.Join(util.ScopeHome(), "cscope")
}

// CreateCScope creates libwrap.so in BOLTON_TMPDIR
func CreateCScope() error {
	cscope := cscopePath()

	// If it exists, don't create
	if _, err := os.Stat(cscope); err == nil {
		return nil
	}

	b, err := buildCscopeBytes()
	if err != nil {
		return err
	}

	err = os.MkdirAll(util.ScopeHome(), 0755)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(cscope, b, 0755)
	if err != nil {
		return err
	}
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

// workDir represents our working directory
var workDir string

// CreateWorkDir creates a working directory
func CreateWorkDir(cmd string) {
	if workDir != "" {
		if util.CheckFileExists(workDir) {
			// Directory exists, exit
			return
		}
	}
	ts := strconv.FormatInt(now().UTC().UnixNano(), 10)
	pid := strconv.Itoa(os.Getpid())
	// Directories named CMD_PID_TIMESTAMP
	tmpDirName := path.Base(cmd) + "_" + pid + "_" + ts
	workDir = filepath.Join(HistoryDir(), tmpDirName)
	cmdDir := filepath.Join(workDir, "cmd")
	err := os.MkdirAll(cmdDir, 0755)
	if err != nil {
		util.ErrAndExit("error creating runDir: %v", err)
	}
	log.WithFields(log.Fields{"workDir": workDir}).Infof("creating working directory")
}

// HistoryDir returns the history directory
func HistoryDir() string {
	return filepath.Join(util.ScopeHome(), "history")
}

// SetupWorkDir sets up a working directory for a given set of args
func SetupWorkDir(args []string) {
	cmd := path.Base(args[0])
	CreateWorkDir(cmd)

	sc := GetDefaultScopeConfig(workDir)

	scYamlPath := filepath.Join(workDir, "scope.yml")
	err := WriteScopeConfig(sc, scYamlPath)
	if err != nil {
		util.ErrAndExit("%v", err)
	}
	argsJSONPath := filepath.Join(workDir, "args.json")
	argsBytes, err := json.Marshal(args)
	if err != nil {
		util.ErrAndExit("error marshaling JSON: %v", err)
	}
	err = ioutil.WriteFile(argsJSONPath, argsBytes, 0644)
}
