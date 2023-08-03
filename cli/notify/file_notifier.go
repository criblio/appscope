package notify

import (
	"fmt"
	"os"
)

type fileNotifier struct {
	file *os.File
}

// NewFileNotifier
func NewFileNotifier(f *os.File) fileNotifier {
	return fileNotifier{
		file: f,
	}
}

// Perform notification via post message using specific file
func (f fileNotifier) Notify(msg string) {
	fmt.Fprintln(f.file, msg)
}
