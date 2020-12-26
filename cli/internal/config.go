package internal

import (
	"os"
	"path/filepath"

	"github.com/criblio/scope/util"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/spf13/viper"
)

// InitConfig reads in config file and ENV variables if set.
func InitConfig(cfgFile string) {
	viper.SetConfigFile(cfgFile)
	viper.SetEnvPrefix("scope")
	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	viper.ReadInConfig()
	// viper.WatchConfig() // Watch config for changes
	// viper.OnConfigChange(func(e fsnotify.Event) {
	// 	log.Debugf("Config file changed: %s", e.Name)
	// })

	zerolog.SetGlobalLevel(zerolog.InfoLevel)
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	if viper.GetInt("verbose") > 0 {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
	SetLogFile(filepath.Join(util.ScopeHome(), "scope.log"))

	c := viper.AllSettings()
	log.Debug().Interface("config", c).Msg("config")
}

// GetConfigMap returns configuration as a map
// func GetConfigMap() (map[string]interface{}, error) {
// 	conf, err := GetConfig()
// 	return structs.Map(conf), err
// }

// SetLogFile sets log output to a particular file path
func SetLogFile(path string) {
	err := os.MkdirAll(filepath.Dir(path), 0755)
	if err != nil {
		util.ErrAndExit("could not create path to log file %s: %v", path, err)
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
	if err != nil {
		util.ErrAndExit("could not open log file %s: %v", path, err)
	}
	log.Logger = zerolog.New(f).With().Timestamp().Caller().Logger()
}
