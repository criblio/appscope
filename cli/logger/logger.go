package logging

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	logrus "github.com/sirupsen/logrus"
	prefixed "github.com/x-cray/logrus-prefixed-formatter"

	"path"
	"runtime"
	"strings"
)

// Logger allows us to create an interface out of our logging functions
type Logger struct{}

var logger *Logger

// GetLogger returns a logger
func GetLogger() *Logger { return logger }

// Debug logs a message at Debug level
func (l Logger) Debug(v ...interface{}) { Debug(v...) }

// Debugf logs a message at Debug level
func (l Logger) Debugf(format string, v ...interface{}) { Debugf(format, v...) }

// Info logs a message at Info level
func (l Logger) Info(v ...interface{}) { Info(v...) }

// Infof logs a message at Info level
func (l Logger) Infof(format string, v ...interface{}) { Infof(format, v...) }

// Warning logs a message at Warning level
func (l Logger) Warning(v ...interface{}) { Warning(v...) }

// Warningf logs a message at Warning level
func (l Logger) Warningf(format string, v ...interface{}) { Warningf(format, v...) }

// Error logs a message at Error level
func (l Logger) Error(v ...interface{}) { Error(v...) }

// Errorf logs a message at Error level
func (l Logger) Errorf(format string, v ...interface{}) { Errorf(format, v...) }

// DefaultLogLevel sets the default log level
var DefaultLogLevel logrus.Level

// outFile contains our file description for writing log entries
var outFile *os.File

// Fields allows passing key value pairs to Logrus
type Fields map[string]interface{}

// ContextHook provides an interface for Logrus Hook for callbacks
type ContextHook struct{}

// Levels returns all available logging levels, required for Logrus Hook implementation
func (hook ContextHook) Levels() []logrus.Level {
	return logrus.AllLevels
}

// Fire is a callback issued for every log message, allowing us to modify the output, required for Logrus Hook implementation
func (hook ContextHook) Fire(entry *logrus.Entry) error {
	pc := make([]uintptr, 5, 5)
	cnt := runtime.Callers(6, pc)

	for i := 0; i < cnt; i++ {
		fu := runtime.FuncForPC(pc[i] - 2)
		name := fu.Name()
		if !strings.Contains(name, "github.com/sirupsen/logrus") &&
			!strings.Contains(name, "github.com/criblio/scope/logger") {
			file, line := fu.FileLine(pc[i] - 2)
			entry.Data["file"] = path.Base(file)
			entry.Data["func"] = path.Base(name)
			entry.Data["line"] = line
			break
		}
	}
	return nil
}

// FileHook also outputs logrus data to a file
type FileHook struct {
	file      *os.File
	flag      int
	chmod     os.FileMode
	formatter *logrus.TextFormatter
}

// NewFileHook also writes logrus output to a file
func NewFileHook(file string, flag int, chmod os.FileMode) (*FileHook, error) {
	plainFormatter := &logrus.TextFormatter{DisableColors: true}
	logFile, err := os.OpenFile(file, flag, chmod)
	if err != nil {
		return nil, fmt.Errorf("unable to write file on filehook %v", err)
	}

	return &FileHook{logFile, flag, chmod, plainFormatter}, err
}

// Fire event
func (hook *FileHook) Fire(entry *logrus.Entry) error {
	plainformat, err := hook.formatter.Format(entry)
	line := string(plainformat)
	_, err = hook.file.WriteString(line)
	if err != nil {
		return fmt.Errorf("unable to write file on filehook(entry.String)%v", err)
	}
	return nil
}

// Levels returns logrus levels for FileHook
func (hook *FileHook) Levels() []logrus.Level {
	return []logrus.Level{
		logrus.PanicLevel,
		logrus.FatalLevel,
		logrus.ErrorLevel,
		logrus.WarnLevel,
		logrus.InfoLevel,
		logrus.DebugLevel,
	}
}

// SetLogFile sets a file to also output to
func SetLogFile(filename string) {
	fileHook, err := NewFileHook(filename, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0666)
	if err == nil {
		logrus.AddHook(fileHook)
	}
}

func init() {
	logger = new(Logger)
	DefaultLogLevel = logrus.InfoLevel
	logrus.SetFormatter(&prefixed.TextFormatter{TimestampFormat: "Jan 02 03:04:05.000", FullTimestamp: true})
	logrus.AddHook(ContextHook{})
	logrus.SetLevel(DefaultLogLevel)
}

// WithField adds a field to the logrus entry
func WithField(key string, value interface{}) *logrus.Entry {
	return logrus.WithField(key, value)
}

// WithFields add fields to the logrus entry
func WithFields(fields Fields) *logrus.Entry {
	sendfields := make(logrus.Fields)
	for k, v := range fields {
		sendfields[k] = v
	}
	return logrus.WithFields(sendfields)
}

// WithError adds an error field to the logrus entry
func WithError(err error) *logrus.Entry {
	return logrus.WithError(err)
}

// Debugf logs a message at Debug level
func Debugf(format string, v ...interface{}) {
	logrus.Debugf(format, v...)
}

// Debug logs a message at Debug level
func Debug(v ...interface{}) {
	logrus.Debug(v...)
}

// Infof logs a message at Info level
func Infof(format string, v ...interface{}) {
	logrus.Infof(format, v...)
}

// Warningf logs a message at Warning level
func Warningf(format string, v ...interface{}) {
	logrus.Warningf(format, v...)
}

// Errorf logs a message at Error level
func Errorf(format string, v ...interface{}) {
	logrus.Errorf(format, v...)
}

// Error logs a message at Error level
func Error(v ...interface{}) {
	logrus.Error(v...)
}

// Warning logs a message at Warning level
func Warning(v ...interface{}) {
	logrus.Warning(v...)
}

// Info logs a message at Info level
func Info(v ...interface{}) {
	logrus.Info(v...)
}

// Panic logs a message at Panic and then exits
func Panic(v ...interface{}) {
	logrus.Panic(v...)
}

// Panicf logs a message at Panic and then exits
func Panicf(format string, v ...interface{}) {
	logrus.Panicf(format, v...)
}

// Fatal logs a message and then exits
func Fatal(v ...interface{}) {
	logrus.Fatal(v...)
}

// Fatalf logs a message and then exits
func Fatalf(format string, v ...interface{}) {
	logrus.Fatalf(format, v...)
}

// EnableJSONOutput sets the Logrus formatter to output in JSON
func EnableJSONOutput() {
	logrus.SetFormatter(&logrus.JSONFormatter{})
	logrus.AddHook(ContextHook{})
}

// EnableWebOutput enables output to our web endpoint
func EnableWebOutput(wh logrus.Hook) {
	logrus.AddHook(wh)
}

// EnableTextOutput sets the Logrus formatter to output as plain text
func EnableTextOutput() {
	logrus.SetFormatter(&logrus.TextFormatter{FullTimestamp: true})
}

// SetOutput sets output a specified file name
func SetOutput(name string) {
	out, err := os.OpenFile(name, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		logrus.SetOutput(os.Stderr)
	}
	logrus.SetOutput(out)
}

// SetNull sets the output to DevNull
func SetNull() {
	logrus.SetOutput(ioutil.Discard)
}

// SetDebug the log level to Debug
func SetDebug() {
	logrus.SetLevel(logrus.DebugLevel)
}

// SetInfo sets the log level to Info
func SetInfo() {
	logrus.SetLevel(logrus.InfoLevel)
}

// SetWarn sets the log level to Warning
func SetWarn() {
	logrus.SetLevel(logrus.WarnLevel)
}

// SetError sets the log level to Error
func SetError() {
	logrus.SetLevel(logrus.ErrorLevel)
}

// GetStdLogger returns a log.Logger compataible interface
func GetStdLogger() *log.Logger {
	w := logrus.StandardLogger().Writer()
	return log.New(w, "", 0)
}
