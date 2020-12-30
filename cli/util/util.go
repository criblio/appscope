package util

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/fatih/structs"
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
		if absPath, err := filepath.Abs(base); err == nil {
			base = absPath
		}
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

// CheckErrSprintf writes a format string to stderr and then exits with a status code of 1 if there is an error
func CheckErrSprintf(err error, format string, a ...interface{}) {
	if err != nil {
		ErrAndExit(format, a...)
	}
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

// GetValue returns a reflect.Value to either the object itself or indirects its pointer first
func GetValue(obj interface{}) reflect.Value {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		return reflect.Indirect(v)
	}
	return v
}

// GetJSONField returns a given struct field given a JSON tag value
func GetJSONField(obj interface{}, field string) *structs.Field {
	fields := structs.Fields(obj)
	for _, f := range fields {
		if strings.Contains(f.Tag("json"), field) {
			return f
		}
	}
	return nil
}

// GetAgo returns a human readable duration from now
func GetAgo(t time.Time) string {
	since := time.Now().Sub(t)
	if since.Hours() > 24*7 {
		days := math.Floor(since.Hours() / 24)
		return fmt.Sprintf("%.fd", days)
	} else if since.Hours() > 24 {
		days := math.Floor(since.Hours() / 24)
		hours := int(math.Floor(since.Hours())) % 24
		return fmt.Sprintf("%.fd%dh", days, hours)
	} else if since.Hours() >= 1 {
		hours := math.Floor(since.Hours())
		mins := int(math.Floor(since.Minutes())) % 60
		return fmt.Sprintf("%.fh%dm", hours, mins)
	} else if since.Minutes() >= 1 {
		mins := math.Floor(since.Minutes())
		secs := int(math.Floor(since.Seconds())) % 60
		return fmt.Sprintf("%.fm%ds", mins, secs)
	} else {
		secs := math.Floor(since.Seconds())
		return fmt.Sprintf("%.fs", secs)
	}
}

// TruncWithElipsis truncates a string to a specified length including the elipsis
func TruncWithElipsis(s string, l int) string {
	if len(s) > l {
		return fmt.Sprintf("%."+strconv.Itoa(l-1)+"s…", s)
	}
	return s
}
