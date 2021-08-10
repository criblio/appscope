package cmd

import (
	"fmt"
	"io/ioutil"
	"os"
	"syscall"
	"github.com/criblio/scope/util"
	"github.com/criblio/scope/run"
	"github.com/spf13/cobra"
)

// servichCmd represents the service command
var serviceCmd = &cobra.Command{
	Use:   "service SERVICE [flags]",
	Short: "Configure a systemd service to be scoped",
	Long: `the "scope service" command adjusts the configuration for the named systemd service so it is scoped when it starts.`,
Example: `scope service  cribl -c tls://in.my-instance.cribl.cloud:10090`,
	Args: cobra.MinimumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// TODO confirm running as root

		// The service name is the first and only argument
		if len(args) < 1 {
			util.ErrAndExit("error: missing required SERVICE argument")
		}
		if len(args) > 1 {
			os.Stderr.WriteString("warn: ignoring extra arguments\n")
		}
		serviceName := args[0]

		// the service name must exist
		serviceFile := fmt.Sprintf("/etc/systemd/system/%s.service", serviceName)
		if _, err := os.Stat(serviceFile); err != nil {
			serviceFile := fmt.Sprintf("/etc/systemd/user/%s.service", serviceName)
			if _, err := os.Stat(serviceFile); err != nil {
				util.ErrAndExit("error: didn't find \"" + serviceName + ".service\" under /etc/systemd")
			}
		}

		// TODO ask "Are you sure?" here

		// extract the library
		// TODO calculate the "-linux-gnu" bits
		utsname := syscall.Utsname{}
		err := syscall.Uname(&utsname)
		util.CheckErrSprintf(err, "error: syscall.Uname failed; %v", err)
		machine := make ([]byte,0,64)
		for _, v := range utsname.Machine[:] {
			if v == 0 { break }
			machine = append(machine, uint8(v))
		}
		libraryDir := fmt.Sprintf("/usr/lib/%s-linux-gnu/cribl", machine)
		if _, err := os.Stat(libraryDir); err != nil {
			err := os.Mkdir(libraryDir, 0755)
			util.CheckErrSprintf(err, "error: failed to create library directory; %v", err)
			//os.Stdout.WriteString("info: created library directory; " + libraryDir + "\n")
		} else {
			//os.Stdout.WriteString("info: library directory exists; " + libraryDir + "\n")
		}
		libraryPath := fmt.Sprintf("/usr/lib/%s-linux-gnu/cribl/libscope.so", machine)
		if _, err := os.Stat(libraryPath); err != nil {
			asset, err := run.Asset("build/libscope.so")
			util.CheckErrSprintf(err, "error: failed to find libscope.so asset; %v", err)
			err = ioutil.WriteFile(libraryPath, asset, 0755)
			util.CheckErrSprintf(err, "error: failed to extract library; %v", err)
			//os.Stdout.WriteString("info: extracted library; " + libraryPath + "\n")
		} else {
			//os.Stdout.WriteString("info: library exists; " + libraryPath + "\n")
		}

		// create the config directory
		configBaseDir := fmt.Sprintf("/etc/scope")
		if _, err := os.Stat(configBaseDir); err != nil {
            err := os.Mkdir(configBaseDir, 0755)
            util.CheckErrSprintf(err, "error: failed to config directory; %v", err)
            //os.Stdout.WriteString("info: created config base directory; " + configBaseDir + "\n")
        } else {
            //os.Stdout.WriteString("info: config base directory exists; " + configBaseDir + "\n")
        }
		configDir := fmt.Sprintf("/etc/scope/%s", serviceName)
		if _, err := os.Stat(configDir); err != nil {
            err := os.Mkdir(configDir, 0755)
            util.CheckErrSprintf(err, "error: failed to config directory; %v", err)
            //os.Stdout.WriteString("info: created config directory; " + configDir + "\n")
        } else {
            //os.Stdout.WriteString("info: config directory exists; " + configDir + "\n")
        }

		// extract scope.yml
		configPath := fmt.Sprintf("/etc/scope/%s/scope.yml", serviceName)
		if _, err := os.Stat(configPath); err == nil {
			util.ErrAndExit("error: scope.yml already exists; " + configPath)
		}
		asset, err := run.Asset("build/scope.yml")
		util.CheckErrSprintf(err, "error: failed to find scope.yml asset; %v", err)
		err = ioutil.WriteFile(configPath, asset, 0644)
		util.CheckErrSprintf(err, "error: failed to extract scope.yml; %v", err)
		if rc.MetricsDest != "" || rc.EventsDest != "" || rc.CriblDest != "" {
			examplePath := fmt.Sprintf("/etc/scope/%s/scope_example.yml", serviceName)
			err = os.Rename(configPath, examplePath)
			util.CheckErrSprintf(err, "error: failed to move scope.yml to soppe_example.yml; %v", err)
			rc.WorkDir = configDir
			err = rc.WriteScopeConfig(configPath, 0644)
			util.CheckErrSprintf(err, "error: failed to create scope.yml: %v", err)
		}

		// extract scope_protocol.yml
		protocolPath := fmt.Sprintf("/etc/scope/%s/scope_protocol.yml", serviceName)
		if _, err := os.Stat(protocolPath); err == nil {
			util.ErrAndExit("error: scope_protocol.yml already exists; " + protocolPath)
		}
		asset, err = run.Asset("build/scope_protocol.yml")
		util.CheckErrSprintf(err, "error: failed to find scope_protocol.yml asset; %v", err)
		err = ioutil.WriteFile(protocolPath, asset, 0644)
		util.CheckErrSprintf(err, "error: failed to extract scope_protocol.yml; %v", err)

		// TODO display what to do next; edit configs and restart service
	},
}

func init() {
	metricAndEventDestFlags(serviceCmd, rc)

	RootCmd.AddCommand(serviceCmd)
}
