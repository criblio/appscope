package internal

import (
	"path/filepath"

	log "github.com/criblio/scope/logger"
	"github.com/criblio/scope/util"
	"github.com/spf13/viper"
)

// InitConfig reads in config file and ENV variables if set.
func InitConfig(cfgFile string) {
	log.SetLogFile(filepath.Join(util.ScopeHome(), "scope.log"))
	viper.SetConfigFile(cfgFile)
	viper.SetEnvPrefix("scope")
	viper.AutomaticEnv() // read in environment variables that match

	// If a config file is found, read it in.
	viper.ReadInConfig()
	// viper.WatchConfig() // Watch config for changes
	// viper.OnConfigChange(func(e fsnotify.Event) {
	// 	log.Debugf("Config file changed: %s", e.Name)
	// })
	if viper.GetInt("verbose") > 0 {
		log.SetDebug()
	}

	c := viper.AllSettings()
	log.WithFields(log.Fields{"config": c}).Debugf("config")
}

// GetConfigMap returns configuration as a map
// func GetConfigMap() (map[string]interface{}, error) {
// 	conf, err := GetConfig()
// 	return structs.Map(conf), err
// }
