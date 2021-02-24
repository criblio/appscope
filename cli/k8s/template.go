package k8s

import (
	"html/template"
	"io"
)

// Options represent the configurable parameters of a k8s deployment
type Options struct {
	App          string
	Namespace    string
	Version      string
	MetricDest   string
	MetricFormat string
	EventDest    string
	CertFile     string
	KeyFile      string
	Port         int
	Debug        bool
}

// PrintConfig will output a k8s config to the specified writer
func PrintConfig(w io.Writer, opt Options) error {
	t1, _ := template.New("webhook").Parse(webhook)
	err := t1.Execute(w, opt)
	if err != nil {
		return err
	}
	t2, _ := template.New("csr").Parse(csr)
	err = t2.Execute(w, opt)
	if err != nil {
		return err
	}
	t3, _ := template.New("deployment").Parse(webhookDeployment)
	err = t3.Execute(w, opt)
	return err
}
