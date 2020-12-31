package cmd

import (
	"fmt"
	"os"
	"strings"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var cfgFile string

// RootCmd represents the base command when called without any subcommands
var RootCmd = &cobra.Command{
	Use:   "scope",
	Short: "command line interface for working with Cribl AppScope",
}

// Execute adds all child commands to the root command and sets flags appropriately.
// This is called by main.main(). It only needs to happen once to the rootCmd.
func Execute() {
	RootCmd.SilenceErrors = true
	if err := RootCmd.Execute(); err != nil {
		if strings.HasPrefix(err.Error(), "unknown command") {
			for _, cmd := range RootCmd.Commands() {
				if os.Args[1] == cmd.Name() {
					cmd.PrintErrln("Error:", err.Error())
					cmd.PrintErrf("Run '%v --help' for usage.\n", cmd.CommandPath())
					os.Exit(1)
				}
			}

			// If we're not a known command, exec cscope
			internal.InitConfig(util.GetConfigPath())
			run.Run(os.Args[1:])
		}
		fmt.Println(err)
		os.Exit(1)
	}
}

func viperPersistentStringFlag(name string, viperName string, shorthand string, value string, usage string) {
	RootCmd.PersistentFlags().StringP(name, shorthand, value, usage)
	viper.BindPFlag(viperName, RootCmd.PersistentFlags().Lookup(name))
}

func init() {
	cobra.OnInitialize(initConfig)

	RootCmd.PersistentFlags().CountP("verbose", "v", "set verbosity level")
	RootCmd.PersistentFlags().StringVar(&cfgFile, "config", util.GetConfigPath(), "config file")
}

func initConfig() {

	viper.BindPFlag("verbose", RootCmd.Flags().Lookup("verbose"))
	internal.InitConfig(cfgFile)
}
