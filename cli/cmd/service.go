package cmd

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"syscall"

	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

var forceFlag bool
var serviceUser string

// serviceCmd represents the service command
var serviceCmd = &cobra.Command{
	Use:     "service SERVICE [flags]",
	Short:   "Configure a systemd service to be scoped",
	Long:    `Configures the specified systemd service to be scoped upon starting.`,
	Example: `scope service  cribl -c tls://in.my-instance.cribl.cloud:10090`,
	Args:    cobra.ExactArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		// must be root
		if 0 != os.Getuid() {
			util.ErrAndExit("error: must be run as root")
		}

		// service name is the first and only argument
		serviceName := args[0]

		// get uname pieces
		utsname := syscall.Utsname{}
		err := syscall.Uname(&utsname)
		util.CheckErrSprintf(err, "error: syscall.Uname failed; %v", err)
		buf := make([]byte, 0, 64)
		for _, v := range utsname.Machine[:] {
			if v == 0 {
				break
			}
			buf = append(buf, uint8(v))
		}
		unameMachine := string(buf)
		buf = make([]byte, 0, 64)
		for _, v := range utsname.Sysname[:] {
			if v == 0 {
				break
			}
			buf = append(buf, uint8(v))
		}
		unameSysname := strings.ToLower(string(buf))

		// TODO get libc name
		libcName := "gnu"

		// Systemd
		if _, err := os.Stat("/etc/systemd"); err == nil {
			installSystemd(serviceName, unameMachine, unameSysname, libcName)
			os.Exit(0)
		}

		// Init.d
		if _, err := os.Stat("/etc/init.d"); err == nil {
			installInitd(serviceName, unameMachine, unameSysname, libcName)
			os.Exit(0)
		}

		// Unknown
		util.ErrAndExit("error: unknown boot system\n")
	},
}

func init() {
	metricAndEventDestFlags(serviceCmd, rc)
	serviceCmd.Flags().BoolVar(&forceFlag, "force", false, "Bypass confirmation prompt")
	serviceCmd.Flags().StringVarP(&serviceUser, "user", "u", "", "Specify owner username")
	RootCmd.AddCommand(serviceCmd)
}

func confirm(s string) bool {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("%s [y/n]: ", s)
		resp, err := reader.ReadString('\n')
		util.CheckErrSprintf(err, "error: confirm failed; %v", err)
		resp = strings.ToLower(strings.TrimSpace(resp))
		if resp == "y" || resp == "yes" {
			return true
		} else if resp == "n" || resp == "no" {
			return false
		}
	}
}

func installScope(serviceName string, unameMachine string, unameSysname string, libcName string) string {
	// determine the library directory
	libraryDir := fmt.Sprintf("/usr/lib/%s-%s-%s", unameMachine, unameSysname, libcName)
	if _, err := os.Stat(libraryDir); err != nil {
		libraryDir = "/usr/lib64"
		if _, err := os.Stat(libraryDir); err != nil {
			util.ErrAndExit("error: failed to determine library directory")
		}
	}

	// extract the library
	libraryDir = libraryDir + "/cribl"
	if _, err := os.Stat(libraryDir); err != nil {
		err := os.Mkdir(libraryDir, 0755)
		util.CheckErrSprintf(err, "error: failed to create library directory; %v", err)
	}
	libraryPath := libraryDir + "/libscope.so"
	if _, err := os.Stat(libraryPath); err != nil {
		asset, err := run.Asset("build/libscope.so")
		util.CheckErrSprintf(err, "error: failed to find libscope.so asset; %v", err)
		err = ioutil.WriteFile(libraryPath, asset, 0755)
		util.CheckErrSprintf(err, "error: failed to extract library; %v", err)
	}

	// create the config directory
	configBaseDir := fmt.Sprintf("/etc/scope")
	if _, err := os.Stat(configBaseDir); err != nil {
		err := os.Mkdir(configBaseDir, 0755)
		util.CheckErrSprintf(err, "error: failed to create config base directory; %v", err)
	}
	configDir := fmt.Sprintf("/etc/scope/%s", serviceName)
	if _, err := os.Stat(configDir); err != nil {
		err := os.Mkdir(configDir, 0755)
		util.CheckErrSprintf(err, "error: failed to create config directory; %v", err)
	}

	// create the log directory
	logDir := fmt.Sprintf("/var/log/scope")
	if _, err := os.Stat(logDir); err != nil {
		err := os.Mkdir(logDir, 0755) // TODO chown/chgrp to who?
		util.CheckErrSprintf(err, "error: failed to create log directory; %v", err)
	}

	// create the run directory
	runDir := fmt.Sprintf("/var/run/scope")
	if _, err := os.Stat(runDir); err != nil {
		err := os.Mkdir(runDir, 0755) // TODO chown/chgrp to who?
		util.CheckErrSprintf(err, "error: failed to create run directory; %v", err)
	}

	// extract scope.yml
	configPath := fmt.Sprintf("/etc/scope/%s/scope.yml", serviceName)
	asset, err := run.Asset("build/scope.yml")
	util.CheckErrSprintf(err, "error: failed to find scope.yml asset; %v", err)
	err = ioutil.WriteFile(configPath, asset, 0644)
	util.CheckErrSprintf(err, "error: failed to extract scope.yml; %v", err)
	examplePath := fmt.Sprintf("/etc/scope/%s/scope_example.yml", serviceName)
	err = os.Rename(configPath, examplePath)
	util.CheckErrSprintf(err, "error: failed to move scope.yml to scope_example.yml; %v", err)
	rc.WorkDir = configDir
	rc.CommandDir = runDir
	rc.LogDest = "file://" + logDir + "/cribl.log"
	err = rc.WriteScopeConfig(configPath, 0644)
	util.CheckErrSprintf(err, "error: failed to create scope.yml: %v", err)

	return libraryPath
}

func locateService(serviceName string) bool {
	servicePrefixDirs := []string{"/etc/systemd/system/", "/lib/systemd/system/", "/run/systemd/system/", "/usr/lib/systemd/system/"}
	for _, prefixDir := range servicePrefixDirs {
		if util.CheckFileExists(prefixDir + serviceName + ".service") {
			return true
		}
	}
	return false
}

func installSystemd(serviceName string, unameMachine string, unameSysname string, libcName string) {
	if !locateService(serviceName) {
		util.ErrAndExit("error: didn't find service file; " + serviceName + ".service")
	}
	serviceDir := "/etc/systemd/system/" + serviceName + ".service.d"
	if !util.CheckDirExists(serviceDir) {
		err := os.MkdirAll(serviceDir, 0655)
		util.CheckErrSprintf(err, "error: failed to create serviceDir directory; %v", err)
	}
	serviceEnvPath := serviceDir + "/env.conf"

	if !forceFlag {
		fmt.Printf("\nThis command will make the following changes if not found already:\n")
		fmt.Printf("  - install libscope.so into /usr/lib/%s-%s-%s/cribl/\n", unameMachine, unameSysname, libcName)
		fmt.Printf("  - create %s override\n", serviceEnvPath)
		fmt.Printf("  - create /etc/scope/%s/scope.yml\n", serviceName)
		fmt.Printf("  - create /var/log/scope/\n")
		fmt.Printf("  - create /var/run/scope/\n")
		if !confirm("Ready to proceed?") {
			util.ErrAndExit("info: canceled")
		}
	}

	libraryPath := installScope(serviceName, unameMachine, unameSysname, libcName)

	if !util.CheckFileExists(serviceEnvPath) {
		content := fmt.Sprintf("[Service]\nEnvironment=LD_PRELOAD=%s\nEnvironment=SCOPE_HOME=/etc/scope/%s\n", libraryPath, serviceName)
		err := ioutil.WriteFile(serviceEnvPath, []byte(content), 0644)
		util.CheckErrSprintf(err, "error: failed to create override file; %v", err)
	}

	fmt.Printf("\nThe %s service has been updated to run with AppScope.\n", serviceName)
	fmt.Printf("\nRun `systemctl daemon-reload` to reload units.\n")
	fmt.Printf("\nPlease review the configs in /etc/scope/%s/scope.yml and check their\npermissions to ensure the scoped service can read them.\n", serviceName)
	fmt.Printf("\nAlso, please review permissions on /var/log/scope and /var/run/scope to ensure\nthe scoped service can write there.\n")
	fmt.Printf("\nRestart the service with `systemctl restart %s` so the changes take effect.\n", serviceName)
}

func installInitd(serviceName string, unameMachine string, unameSysname string, libcName string) {
	initScript := "/etc/init.d/" + serviceName
	if _, err := os.Stat(initScript); err != nil {
		util.ErrAndExit("error: didn't find service script; " + initScript)
	}

	if !forceFlag {
		fmt.Printf("\nThis command will make the following changes if not found already:\n")
		fmt.Printf("  - install libscope.so into /usr/lib/%s-%s-%s/cribl/\n", unameMachine, unameSysname, libcName)
		fmt.Printf("  - create /etc/sysconfig/%s\n", serviceName)
		fmt.Printf("  - create /etc/scope/%s/scope.yml\n", serviceName)
		fmt.Printf("  - create /var/log/scope/\n")
		fmt.Printf("  - create /var/run/scope/\n")
		if !confirm("Ready to proceed?") {
			util.ErrAndExit("info: canceled")
		}
	}

	libraryPath := installScope(serviceName, unameMachine, unameSysname, libcName)

	sysconfigFile := "/etc/sysconfig/" + serviceName
	if _, err := os.Stat(sysconfigFile); err != nil {
		content := fmt.Sprintf("LD_PRELOAD=%s\nSCOPE_HOME=/etc/scope/%s\n", libraryPath, serviceName)
		err := ioutil.WriteFile(sysconfigFile, []byte(content), 0644)
		util.CheckErrSprintf(err, "error: failed to create sysconfig file; %v", err)
	}

	fmt.Printf("\nThe %s service has been updated to run with AppScope.\n", serviceName)
	fmt.Printf("\nPlease review the configs in /etc/scope/%s/scope.yml and check their\npermissions to ensure the scoped service can read them.\n", serviceName)
	fmt.Printf("\nAlso, please review permissions on /var/log/scope and /var/run/scope to ensure\nthe scoped service can write there.\n")
	fmt.Printf("\nRestart the service with `service %s restart` so the changes take effect.\n", serviceName)
}
