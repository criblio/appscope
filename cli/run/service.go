package run

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strings"
	"syscall"

	"github.com/criblio/scope/util"
)

var (
	ErrRoot                = errors.New("must be run as a root")
	ErrSyscallUname        = errors.New("syscall.Uname failed")
	ErrUnkownBootSystem    = errors.New("unknown boot system")
	ErrConfirmFailed       = errors.New("confirm failed")
	ErrCreateLibDir        = errors.New("failed to create library directory")
	ErrCreateScopeYml      = errors.New("failed to create scope.yml")
	ErrCreateServiceDir    = errors.New("failed to create service directory")
	ErrAssetLibscope       = errors.New("failed to find libscope.so asset")
	ErrAssetLdscope        = errors.New("failed to find ldscope asset")
	ErrAssetScopeYml       = errors.New("failed to find scope.yml asset")
	ErrExtractLib          = errors.New("failed to extract library")
	ErrExtractLoader       = errors.New("failed to extract loader")
	ErrExtractScopeYml     = errors.New("failed to extract scope.yml")
	ErrMoveScopeYml        = errors.New("failed to move scope.yml to scope_example.yml")
	ErrRemoveLoader        = errors.New("failed to remove loader")
	ErrCreateConfigBaseDir = errors.New("failed to create config base directory")
	ErrCreateConfigDir     = errors.New("failed to create config directory")
	ErrCreateRunDir        = errors.New("failed to create run directory")
	ErrCreateLogDir        = errors.New("failed to create log directory")
	ErrMissingService      = errors.New("missing service file")
	ErrCancel              = errors.New("info: canceled")
	ErrOverrideFileCreate  = errors.New("failed to create override file")
	ErrOverrideFileRead    = errors.New("failed to read override file")
	ErrOverrideFileUpdate  = errors.New("failed to update override file")
	ErrSysConfigFileCreate = errors.New("failed to create sysconfig file")
	ErrSysConfigFileOpen   = errors.New("failed to open sysconfig file")
	ErrSysConfigFileUpdate = errors.New("failed to update sysconfig file")
	ErrSysConfigFileRead   = errors.New("failed to read sysconfig file")
)

func (rc *Config) Service(serviceName string, user string, force bool) error {
	// TODO: move it to separate package
	// must be root
	if os.Getuid() != 0 {
		return ErrRoot
	}

	// get uname pieces
	utsname := syscall.Utsname{}
	err := syscall.Uname(&utsname)
	if err != nil {
		return ErrSyscallUname
	}
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

	if util.CheckFileExists("/etc/rc.conf") {
		// OpenRC
		return rc.installOpenRc(serviceName, unameMachine, unameSysname, libcName, force)
	} else if util.CheckDirExists("/etc/systemd") {
		// Systemd
		return rc.installSystemd(serviceName, unameMachine, unameSysname, libcName, force)
	} else if util.CheckDirExists("/etc/init.d") {
		// Initd
		return rc.installInitd(serviceName, unameMachine, unameSysname, libcName, force)
	}
	return ErrUnkownBootSystem
}

func confirm(s string) (bool, error) {
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("%s [y/n]: ", s)
		resp, err := reader.ReadString('\n')
		if err != nil {
			return false, ErrConfirmFailed
		}
		resp = strings.ToLower(strings.TrimSpace(resp))
		if resp == "y" || resp == "yes" {
			return true, nil
		} else if resp == "n" || resp == "no" {
			return false, nil
		}
	}
}

func (rc *Config) installScope(serviceName string, unameMachine string, unameSysname string, libcName string) (string, error) {
	// determine the library directory
	libraryDir := fmt.Sprintf("/usr/lib/%s-%s-%s", unameMachine, unameSysname, libcName)
	if !util.CheckDirExists(libraryDir) {
		if err := os.MkdirAll(libraryDir, 0755); err != nil {
			return "", ErrCreateLibDir
		}
	}

	// extract the library
	libraryDir = libraryDir + "/cribl"
	if !util.CheckDirExists(libraryDir) {
		if err := os.Mkdir(libraryDir, 0755); err != nil {
			return "", ErrCreateLibDir
		}
	}
	libraryPath := libraryDir + "/libscope.so"
	if !util.CheckFileExists(libraryPath) {
		asset, err := Asset("build/libscope.so")
		if err != nil {
			return "", ErrAssetLibscope
		}

		if err := os.WriteFile(libraryPath, asset, 0755); err != nil {
			return "", ErrExtractLib
		}
	}

	// patch the library
	asset, err := Asset("build/ldscope")
	if err != nil {
		return "", ErrAssetLdscope
	}
	loaderPath := libraryDir + "/ldscope"
	if err := os.WriteFile(loaderPath, asset, 0755); err != nil {
		return "", ErrExtractLoader
	}
	rc.Patch(libraryDir)
	if err = os.Remove(loaderPath); err != nil {
		return "", ErrRemoveLoader
	}

	// create the config directory
	configBaseDir := "/etc/scope"
	if !util.CheckDirExists(configBaseDir) {
		if err := os.Mkdir(configBaseDir, 0755); err != nil {
			return "", ErrCreateConfigBaseDir
		}
	}
	configDir := fmt.Sprintf("/etc/scope/%s", serviceName)
	if !util.CheckDirExists(configDir) {
		if err := os.Mkdir(configDir, 0755); err != nil {
			return "", ErrCreateConfigDir
		}
	}

	// create the log directory
	logDir := "/var/log/scope"
	if !util.CheckDirExists(logDir) {
		// TODO chown/chgrp to who?
		if err := os.Mkdir(logDir, 0755); err != nil {
			return "", ErrCreateLogDir
		}
	}

	// create the run directory
	runDir := "/var/run/scope"
	if !util.CheckDirExists(runDir) {
		// TODO chown/chgrp to who?
		if err := os.Mkdir(runDir, 0755); err != nil {
			return "", ErrCreateRunDir
		}

	}

	// extract scope.yml
	configPath := fmt.Sprintf("/etc/scope/%s/scope.yml", serviceName)
	asset, err = Asset("build/scope.yml")
	if err != nil {
		return "", ErrAssetScopeYml
	}

	if err := os.WriteFile(configPath, asset, 0644); err != nil {
		return "", ErrExtractScopeYml
	}
	examplePath := fmt.Sprintf("/etc/scope/%s/scope_example.yml", serviceName)
	if err := os.Rename(configPath, examplePath); err != nil {
		return "", ErrMoveScopeYml
	}
	rc.WorkDir = configDir
	rc.CommandDir = runDir
	rc.LogDest = "file://" + logDir + "/cribl.log"
	if err := rc.WriteScopeConfig(configPath, 0644); err != nil {
		return "", ErrCreateScopeYml
	}

	return libraryPath, nil
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

func (rc *Config) installSystemd(serviceName string, unameMachine string, unameSysname string, libcName string, force bool) error {
	if !locateService(serviceName) {
		return fmt.Errorf("%v %v.service", ErrMissingService, serviceName)
	}
	serviceDir := "/etc/systemd/system/" + serviceName + ".service.d"
	if !util.CheckDirExists(serviceDir) {
		if err := os.MkdirAll(serviceDir, 0655); err != nil {
			return ErrCreateServiceDir
		}
	}
	serviceEnvPath := serviceDir + "/env.conf"

	if !force {
		fmt.Printf("\nThis command will make the following changes if not found already:\n")
		fmt.Printf("  - install libscope.so into /usr/lib/%s-%s-%s/cribl/\n", unameMachine, unameSysname, libcName)
		fmt.Printf("  - create/update %s override\n", serviceEnvPath)
		fmt.Printf("  - create /etc/scope/%s/scope.yml\n", serviceName)
		fmt.Printf("  - create /var/log/scope/\n")
		fmt.Printf("  - create /var/run/scope/\n")
		status, err := confirm("Ready to proceed?")
		if err != nil || !status {
			return ErrCancel
		}
	}

	libraryPath, err := rc.installScope(serviceName, unameMachine, unameSysname, libcName)
	if err != nil {
		return err
	}
	if !util.CheckFileExists(serviceEnvPath) {
		// Create env.conf
		content := fmt.Sprintf("[Service]\nEnvironment=LD_PRELOAD=%s\nEnvironment=SCOPE_HOME=/etc/scope/%s\n", libraryPath, serviceName)
		if err := os.WriteFile(serviceEnvPath, []byte(content), 0644); err != nil {
			return ErrOverrideFileCreate
		}
	} else {
		// Modify env.conf
		data, err := os.ReadFile(serviceEnvPath)
		if err != nil {
			return ErrOverrideFileRead
		}
		strData := string(data)
		if !strings.Contains(strData, "[Service]\n") {
			// Create Service section
			content := fmt.Sprintf("[Service]\nEnvironment=LD_PRELOAD=%s\nEnvironment=SCOPE_HOME=/etc/scope/%s\n", libraryPath, serviceName)
			f, err := os.OpenFile(serviceEnvPath, os.O_APPEND|os.O_WRONLY, 0644)
			if err != nil {
				return ErrSysConfigFileOpen
			}
			defer f.Close()
			_, err = f.WriteString(content)
			if err != nil {
				return ErrOverrideFileUpdate
			}
		} else {
			// Update Service section
			oldData := "[Service]\n"
			if !strings.Contains(strData, "Environment=LD_PRELOAD=") {
				newData := oldData + fmt.Sprintf("Environment=LD_PRELOAD=%s\n", libraryPath)
				strData = strings.Replace(strData, oldData, newData, 1)
				oldData = newData
			}
			if !strings.Contains(strData, "Environment=SCOPE_HOME=") {
				newData := oldData + fmt.Sprintf("Environment=SCOPE_HOME=/etc/scope/%s\n", serviceName)
				strData = strings.Replace(strData, oldData, newData, 1)
			}
			if err := os.WriteFile(serviceEnvPath, []byte(strData), 0644); err != nil {
				return ErrOverrideFileCreate
			}
		}
	}

	fmt.Printf("\nThe %s service has been updated to run with AppScope.\n", serviceName)
	fmt.Printf("\nRun `systemctl daemon-reload` to reload units.\n")
	fmt.Printf("\nPlease review the configs in /etc/scope/%s/scope.yml and check their\npermissions to ensure the scoped service can read them.\n", serviceName)
	fmt.Printf("\nAlso, please review permissions on /var/log/scope and /var/run/scope to ensure\nthe scoped service can write there.\n")
	fmt.Printf("\nRestart the service with `systemctl restart %s` so the changes take effect.\n", serviceName)
	return nil
}

func (rc *Config) installInitd(serviceName string, unameMachine string, unameSysname string, libcName string, force bool) error {
	initScript := "/etc/init.d/" + serviceName
	if !util.CheckFileExists(initScript) {
		return fmt.Errorf("%v %v", ErrMissingService, initScript)
	}

	sysconfigFile := "/etc/sysconfig/" + serviceName

	if !force {
		fmt.Printf("\nThis command will make the following changes if not found already:\n")
		fmt.Printf("  - install libscope.so into /usr/lib/%s-%s-%s/cribl/\n", unameMachine, unameSysname, libcName)
		fmt.Printf("  - create/update %s\n", sysconfigFile)
		fmt.Printf("  - create /etc/scope/%s/scope.yml\n", serviceName)
		fmt.Printf("  - create /var/log/scope/\n")
		fmt.Printf("  - create /var/run/scope/\n")
		status, err := confirm("Ready to proceed?")
		if err != nil || !status {
			return ErrCancel
		}
	}

	libraryPath, err := rc.installScope(serviceName, unameMachine, unameSysname, libcName)
	if err != nil {
		return err
	}

	if !util.CheckFileExists(sysconfigFile) {
		content := fmt.Sprintf("LD_PRELOAD=%s\nSCOPE_HOME=/etc/scope/%s\n", libraryPath, serviceName)
		if err := os.WriteFile(sysconfigFile, []byte(content), 0644); err != nil {
			return ErrSysConfigFileCreate
		}
	} else {
		data, err := os.ReadFile(sysconfigFile)
		if err != nil {
			return ErrSysConfigFileRead
		}
		strData := string(data)
		if !strings.Contains(strData, "LD_PRELOAD=") {
			content := fmt.Sprintf("LD_PRELOAD=%s\n", libraryPath)
			f, err := os.OpenFile(sysconfigFile, os.O_APPEND|os.O_WRONLY, 0644)
			if err != nil {
				return ErrSysConfigFileOpen
			}
			defer f.Close()
			if _, err := f.WriteString(content); err != nil {
				return ErrSysConfigFileUpdate
			}
		}
		if !strings.Contains(strData, "SCOPE_HOME=") {
			content := fmt.Sprintf("SCOPE_HOME=/etc/scope/%s\n", serviceName)
			f, err := os.OpenFile(sysconfigFile, os.O_APPEND|os.O_WRONLY, 0644)
			if err != nil {
				return ErrSysConfigFileOpen
			}
			defer f.Close()
			if _, err = f.WriteString(content); err != nil {
				return ErrSysConfigFileUpdate
			}
		}
	}

	fmt.Printf("\nThe %s service has been updated to run with AppScope.\n", serviceName)
	fmt.Printf("\nPlease review the configs in /etc/scope/%s/scope.yml and check their\npermissions to ensure the scoped service can read them.\n", serviceName)
	fmt.Printf("\nAlso, please review permissions on /var/log/scope and /var/run/scope to ensure\nthe scoped service can write there.\n")
	fmt.Printf("\nRestart the service with `service %s restart` so the changes take effect.\n", serviceName)
	return nil
}

func (rc *Config) installOpenRc(serviceName string, unameMachine string, unameSysname string, libcName string, force bool) error {
	initScript := "/etc/init.d/" + serviceName
	if !util.CheckFileExists(initScript) {
		return fmt.Errorf("%v %v", ErrMissingService, initScript)
	}

	initConfigScript := "/etc/conf.d/" + serviceName

	if !force {
		fmt.Printf("\nThis command will make the following changes if not found already:\n")
		fmt.Printf("  - install libscope.so into /usr/lib/%s-%s-%s/cribl/\n", unameMachine, unameSysname, libcName)
		fmt.Printf("  - create/update %s\n", initConfigScript)
		fmt.Printf("  - create /etc/scope/%s/scope.yml\n", serviceName)
		fmt.Printf("  - create /var/log/scope/\n")
		fmt.Printf("  - create /var/run/scope/\n")
		status, err := confirm("Ready to proceed?")
		if err != nil || !status {
			return ErrCancel
		}
	}

	libraryPath, err := rc.installScope(serviceName, unameMachine, unameSysname, libcName)
	if err != nil {
		return err
	}

	if !util.CheckFileExists(initConfigScript) {
		fmt.Printf("Write to a file %s", initConfigScript)
		content := fmt.Sprintf("export LD_PRELOAD=%s\nexport SCOPE_HOME=/etc/scope/%s\n", libraryPath, serviceName)
		if err := os.WriteFile(initConfigScript, []byte(content), 0644); err != nil {
			return ErrSysConfigFileUpdate
		}
	} else {
		data, err := os.ReadFile(initConfigScript)
		if err != nil {
			return ErrSysConfigFileRead
		}
		strData := string(data)
		if !strings.Contains(strData, "export LD_PRELOAD=") {
			content := fmt.Sprintf("export LD_PRELOAD=%s\n", libraryPath)
			f, err := os.OpenFile(initConfigScript, os.O_APPEND|os.O_WRONLY, 0644)
			if err != nil {
				return ErrSysConfigFileOpen
			}
			defer f.Close()
			if _, err := f.WriteString(content); err != nil {
				return ErrSysConfigFileUpdate
			}
		}
		if !strings.Contains(strData, "export SCOPE_HOME=") {
			content := fmt.Sprintf("export SCOPE_HOME=/etc/scope/%s\n", serviceName)
			f, err := os.OpenFile(initConfigScript, os.O_APPEND|os.O_WRONLY, 0644)
			if err != nil {
				return ErrSysConfigFileOpen
			}
			defer f.Close()
			if _, err = f.WriteString(content); err != nil {
				return ErrSysConfigFileUpdate
			}
		}
	}

	fmt.Printf("\nThe %s service has been updated to run with AppScope.\n", serviceName)
	fmt.Printf("\nPlease review the configs in /etc/scope/%s/scope.yml and check their\npermissions to ensure the scoped service can read them.\n", serviceName)
	fmt.Printf("\nAlso, please review permissions on /var/log/scope and /var/run/scope to ensure\nthe scoped service can write there.\n")
	fmt.Printf("\nRestart the service with `rc-service %s restart` so the changes take effect.\n", serviceName)
	return nil
}
