package run

import (
	"os"
	"path/filepath"
	"time"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/loader"
	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

// Config represents options to change how we run scope
type Config struct {
	WorkDir       string
	Passthrough   bool
	Verbosity     int
	Payloads      bool
	MetricsDest   string
	EventsDest    string
	MetricsFormat string
	CriblDest     string
	Subprocess    bool
	Loglevel      string
	LogDest       string
	LibraryPath   string
	NoBreaker     bool
	AuthToken     string
	UserConfig    string
	CommandDir    string

	now func() time.Time
	sc  *libscope.ScopeConfig
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
	// Disable detection of a scope filter file with this command
	env = append(env, "SCOPE_FILTER=false")

	if rc.NoBreaker {
		env = append(env, "SCOPE_CRIBL_NO_BREAKER=true")
	}
	if !rc.Passthrough {
		rc.setupWorkDir(args, false)
		env = append(env, "SCOPE_CONF_PATH="+filepath.Join(rc.WorkDir, "scope.yml"))
		log.Info().Bool("passthrough", rc.Passthrough).Strs("args", args).Msg("calling syscall.Exec")
	}
	if len(rc.LibraryPath) > 0 {
		// Validate path exists
		if !util.CheckDirExists(rc.LibraryPath) {
			util.ErrAndExit("Library Path does not exist: \"%s\"", rc.LibraryPath)
		}
		// Prepend "-f" [PATH] to args
		args = append([]string{"-f", rc.LibraryPath}, args...)
	}
	ld := loader.New()
	if !rc.Subprocess {
		ld.Run(args, env)
	}
	ld.RunSubProc(args, env)
}
