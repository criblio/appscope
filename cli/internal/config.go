package internal

import (
	"io"
	"os"
	"path/filepath"

	"github.com/criblio/scope/util"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// InitConfig reads in config file and ENV variables if set.
func InitConfig() {
	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	CreateLogFile(filepath.Join(util.ScopeHome(), "scope.log"))
}

// GetConfigMap returns configuration as a map
// func GetConfigMap() (map[string]interface{}, error) {
// 	conf, err := GetConfig()
// 	return structs.Map(conf), err
// }

// CreateLogFile sets log output to a particular file path
func CreateLogFile(path string) {
	err := os.MkdirAll(filepath.Dir(path), 0755)
	util.CheckErrSprintf(err, "could not create path to log file %s: %v", path, err)
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	util.CheckErrSprintf(err, "could not open log file %s: %v", path, err)
	SetLogWriter(f)
}

// SetLogWriter sets log to output to a given writer
func SetLogWriter(w io.Writer) {
	log.Logger = zerolog.New(w).With().Timestamp().Caller().Logger()
}

// SetDebug sets logging to debug
func SetDebug() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}
