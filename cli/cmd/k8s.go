package cmd

import (
	"os"

	"github.com/criblio/scope/internal"
	"github.com/criblio/scope/k8s"
	"github.com/criblio/scope/util"
	"github.com/spf13/cobra"
)

var opt k8s.Options

// k8sCmd represents the k8s command
var k8sCmd = &cobra.Command{
	Use:   "k8s",
	Short: "Install scope in kubernetes",
	Long: `Prints configurations to pass to kubectl, which then automatically instruments newly-launched containers. This installs a mutating admission webhook, which adds an initContainer to each pod. The webhook also sets environment variables that install AppScope for all processes in that container.

The --*dest flags accept file names like /tmp/scope.log; URLs like file:///tmp/scope.log; or sockets specified with the pattern unix:///var/run/mysock, tcp://hostname:port, udp://hostname:port, or tls://hostname:port.`,
	Example: `scope k8s --metricdest tcp://some.host:8125 --eventdest tcp://other.host:10070 | kubectl apply -f -
kubectl label namespace default scope=enabled`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		if rc.CriblDest == "" && rc.MetricsDest == "" {
			util.ErrAndExit("CriblDest and MetricDest not set. Must set one.")
		}
		if rc.MetricsDest != "" && rc.EventsDest == "" {
			util.ErrAndExit("MetricDest set but EventDest is not")
		}
		if opt.Version == "" {
			opt.Version = internal.GetVersion()
		}
		opt.MetricDest = rc.MetricsDest
		opt.MetricFormat = rc.MetricsFormat
		opt.EventDest = rc.EventsDest
		opt.CriblDest = rc.CriblDest
		rc.WorkDir = "/scope"
		var err error
		opt.ScopeConfigYaml, err = rc.ScopeConfigYaml()
		util.CheckErrSprintf(err, "%v", err)
		server, _ := cmd.Flags().GetBool("server")
		if !server {
			opt.PrintConfig(os.Stdout)
			os.Exit(0)
		}
		internal.SetLogWriter(os.Stdout)
		if opt.Debug {
			internal.SetDebug()
		}
		err = opt.StartServer()
		util.CheckErrSprintf(err, "%v", err)
	},
}

func init() {
	RootCmd.AddCommand(k8sCmd)
	k8sCmd.Flags().StringVar(&opt.App, "app", "scope", "Name of the app in Kubernetes")
	k8sCmd.Flags().StringVar(&opt.Namespace, "namespace", "default", "Name of the namespace in which to install")
	k8sCmd.Flags().StringVar(&opt.Version, "version", "", "Version of scope to deploy")
	k8sCmd.Flags().StringVar(&opt.CertFile, "certfile", "/etc/certs/tls.crt", "Certificate file for TLS in the container (mounted secret)")
	k8sCmd.Flags().StringVar(&opt.KeyFile, "keyfile", "/etc/certs/tls.key", "Private key file for TLS in the container (mounted secret)")
	k8sCmd.Flags().IntVar(&opt.Port, "port", 4443, "Port to listen on")
	k8sCmd.Flags().Bool("server", false, "Run Webhook server")
	k8sCmd.Flags().BoolVar(&opt.Debug, "debug", false, "Turn on debug logging in the scope webhook container")
	metricAndEventDestFlags(k8sCmd, rc)
}
