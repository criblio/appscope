module github.com/criblio/scope

go 1.15

require (
	github.com/Masterminds/goutils v1.1.1 // indirect
	github.com/Masterminds/semver v1.5.0 // indirect
	github.com/Masterminds/sprig v2.22.0+incompatible
	github.com/ahmetb/go-linq/v3 v3.2.0
	github.com/c9s/goprocinfo v0.0.0-20210130143923-c95fcf8c64a8
	github.com/dlclark/regexp2 v1.4.0 // indirect
	github.com/dop251/goja v0.0.0-20201221183957-6b6d5e2b5d80
	github.com/fatih/color v1.7.0
	github.com/fatih/structs v1.1.0
	github.com/go-sourcemap/sourcemap v2.1.3+incompatible // indirect
	github.com/guptarohit/asciigraph v0.5.1
	github.com/huandu/xstrings v1.3.2 // indirect
	github.com/imdario/mergo v0.3.11 // indirect
	github.com/kballard/go-shellquote v0.0.0-20180428030007-95032a82bc51
	github.com/mitchellh/copystructure v1.1.1 // indirect
	github.com/mitchellh/mapstructure v1.4.1
	github.com/mum4k/termdash v0.14.0
	github.com/olekukonko/tablewriter v0.0.4
	github.com/rs/zerolog v1.20.0
	github.com/sirupsen/logrus v1.2.0
	github.com/spf13/cobra v1.1.1
	github.com/stretchr/testify v1.6.1
	github.com/syndtr/gocapability v0.0.0-20200815063812-42c35b437635
	golang.org/x/crypto v0.0.0-20201002170205-7f63de1d35b0
	golang.org/x/sync v0.0.0-20190911185100-cd5d95a43a6e
	golang.org/x/sys v0.0.0-20201223074533-0d417f636930 // indirect
	gopkg.in/yaml.v2 v2.4.0
	k8s.io/api v0.20.2
	k8s.io/apimachinery v0.20.2
	k8s.io/client-go v0.20.2
)

replace github.com/olekukonko/tablewriter => github.com/cockroachdb/tablewriter v0.0.5-0.20200105123400-bd15540e8847
