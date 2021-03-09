package k8s

import (
	"html/template"
	"io"

	"github.com/Masterminds/sprig"
)

// Options represent the configurable parameters of a k8s deployment
type Options struct {
	App             string
	Namespace       string
	Version         string
	CriblDest       string
	MetricDest      string
	MetricFormat    string
	EventDest       string
	CertFile        string
	KeyFile         string
	Port            int
	Debug           bool
	ScopeConfigYaml []byte
}

// PrintConfig will output a k8s config to the specified writer
func (opt Options) PrintConfig(w io.Writer) error {
	t1, _ := template.New("webhook").Funcs(sprig.FuncMap()).Parse(webhook)
	err := t1.Execute(w, opt)
	if err != nil {
		return err
	}
	t2, _ := template.New("csr").Funcs(sprig.FuncMap()).Parse(csr)
	err = t2.Execute(w, opt)
	if err != nil {
		return err
	}
	t3, _ := template.New("deployment").Funcs(sprig.FuncMap()).Parse(webhookDeployment)
	err = t3.Execute(w, opt)
	return err
}
