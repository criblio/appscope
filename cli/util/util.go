package util

import (
	"bytes"
	"fmt"
	"io"
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

type ReadSeekCloser interface {
	io.ReadSeeker
	io.Closer
}

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

// CheckDirExists checks if a directory exists on the filesystem
func CheckDirExists(filePath string) bool {
	stat, err := os.Stat(os.ExpandEnv(filePath))
	if err != nil {
		return false
	}
	if !stat.IsDir() {
		return false
	}
	return true
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

// GetHumanDuration returns a human readable duration
func GetHumanDuration(since time.Duration) string {
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
	} else if since.Seconds() >= 1 {
		secs := math.Floor(since.Seconds())
		return fmt.Sprintf("%.fs", secs)
	} else {
		return fmt.Sprintf("%dms", since.Milliseconds())
	}
}

// TruncWithElipsis truncates a string to a specified length including the elipsis
func TruncWithElipsis(s string, l int) string {
	if len(s) > l {
		return fmt.Sprintf("%."+strconv.Itoa(l-1)+"s…", s)
	}
	return s
}

// Trunc truncates a string to a specified length
func Trunc(s string, l int) string {
	if len(s) > l {
		return fmt.Sprintf("%."+strconv.Itoa(l)+"s", s)
	}
	return s
}

// CountingReader counts the bytes read from an io.Reader
type CountingReader struct {
	io.Reader
	BytesRead int
}

func (r *CountingReader) Read(p []byte) (n int, err error) {
	n, err = r.Reader.Read(p)
	r.BytesRead += n
	return n, err
}

// TailReader implements a simple polling based file tailer
type TailReader struct {
	ReadSeekCloser
}

func (t TailReader) Read(b []byte) (int, error) {
	for {
		n, err := t.ReadSeekCloser.Read(b)
		if n > 0 {
			return n, nil
		} else if err != io.EOF {
			return n, err
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// Seek moves the file cursor to the given offset
func (t TailReader) Seek(offset int64, whence int) (int64, error) {
	return t.ReadSeekCloser.Seek(offset, whence)
}

// Close closes the file
func (t TailReader) Close() error {
	return t.ReadSeekCloser.Close()
}

// NewTailReader creates a new tailing reader for fileName
func NewTailReader(f ReadSeekCloser) TailReader {
	return TailReader{f}
}

// FormatTimestamp prints a human readable timestamp from scope's secs.millisecs format.
func FormatTimestamp(timeFp float64) string {
	return ParseEventTime(timeFp).Format(time.Stamp)
}

// ParseEventTime parses an event time in secs.msecs format
func ParseEventTime(timeFp float64) time.Time {
	secs := int64(math.Floor(timeFp))
	nSecs := int64(math.Mod(timeFp, 1) * 1000000000)
	return time.Unix(secs, nSecs)
}

// CountLines returns the number of lines in a given file
func CountLines(path string) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	buf := make([]byte, 32*1024)
	count := 0
	lineSep := []byte{'\n'}

	for {
		c, err := file.Read(buf)
		count += bytes.Count(buf[:c], lineSep)

		switch {
		case err == io.EOF:
			return count, nil

		case err != nil:
			return count, err
		}
	}
}

// Copy pasta from https://codereview.stackexchange.com/questions/71272/convert-int64-to-custom-base64-number-string
var codes = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%^"

func EncodeOffset(offset int64) string {
	str := make([]byte, 0, 12)
	if offset == 0 {
		return "0"
	}
	for offset > 0 {
		ch := codes[offset%64]
		str = append(str, byte(ch))
		offset /= 64
	}
	return string(str)
}

func DecodeOffset(encoded string) (int64, error) {
	res := int64(0)

	for i := len(encoded); i > 0; i-- {
		ch := encoded[i-1]
		res *= 64
		mod := strings.IndexRune(codes, rune(ch))
		if mod == -1 {
			return -1, fmt.Errorf("Invalid encoded character: '%c'", ch)
		}
		res += int64(mod)
	}
	return res, nil
}

// copy pasta from https://yourbasic.org/golang/formatting-byte-size-to-human-readable-format/
func ByteCountSI(b int64) string {
	const unit = 1000
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB",
		float64(b)/float64(div), "kMGTPE"[exp])
}
