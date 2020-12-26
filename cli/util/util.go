package util

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

func init() {
	if os.Getenv("SCOPE_NOTRAND") != "" {
		rand.Seed(0)
	} else {
		rand.Seed(time.Now().UnixNano())
	}
}

var letterRunes = []rune("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

// RandString returns a random string of length n
func RandString(n int) string {
	b := make([]rune, n)
	for i := range b {
		b[i] = letterRunes[rand.Intn(len(letterRunes))]
	}
	return string(b)
}

// Cache home value
var scopeHome string

// ScopeHome returns the scope home directory, default $HOME/.scope
func ScopeHome() string {
	if scopeHome != "" {
		return scopeHome
	}
	base, match := os.LookupEnv("SCOPE_HOME")
	if match {
		scopeHome = base
		return scopeHome
	}
	home, match := os.LookupEnv("HOME")
	if !match {
		home, match = os.LookupEnv("TMPDIR")
		if !match {
			home = "/tmp"
		}
	}
	scopeHome = filepath.Join(home, ".scope")
	return scopeHome
}

// GetConfigPath returns path to our default config file
func GetConfigPath() string {
	return filepath.Join(ScopeHome(), "config.yml")
}

// ErrAndExit writes a format string to stderr and then exits with a status code of 1
func ErrAndExit(format string, a ...interface{}) {
	out := fmt.Sprintf(format, a...)
	if out[len(out)-1:] != "\n" {
		out += "\n"
	}
	os.Stderr.WriteString(out)
	os.Exit(1)
}

// CheckFileExists checks if a file exists on the filesystem
func CheckFileExists(filePath string) bool {
	if _, err := os.Stat(os.ExpandEnv(filePath)); err == nil { // File exists, no errors
		return true
	}
	return false
}
