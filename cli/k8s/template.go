package k8s

import (
	"html/template"
	"io"
)

// Options represent the configurable parameters of a k8s deployment
type Options struct {
	App       string
	Namespace string
}

// PrintConfig will output a k8s config to the specified writer
func PrintConfig(w io.Writer, opt Options) error {
	t := template.New("k8s")
	t1 := template.Must(t.Parse(webhook))
	return t1.Execute(w, opt)
}
