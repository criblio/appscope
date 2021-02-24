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
	Long: `The k8s command prints configurations to be passed to kubectl which will automatically instrument newly launched containers. This installs
a mutating admission webhook which adds an initContainer to each pod along with some environment variables which installs scope for all
all processes in that container.`,
	Example: `scope k8s --metricdest tcp://some.host:8125 --eventdest tcp://other.host:10070 | kubectl apply -f -
kubectl label namespace default scope=enabled`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		if opt.Version == "" {
			opt.Version = internal.GetVersion()
		}
		opt.MetricDest = rc.MetricsDest
		opt.MetricFormat = rc.MetricsFormat
		opt.EventDest = rc.EventsDest
		server, _ := cmd.Flags().GetBool("server")
		if !server {
			k8s.PrintConfig(os.Stdout, opt)
			os.Exit(0)
		}
		internal.SetLogWriter(os.Stdout)
		if opt.Debug {
			internal.SetDebug()
		}
		err := k8s.StartServer(opt)
		util.CheckErrSprintf(err, "%v", err)
	},
}

func init() {
	RootCmd.AddCommand(k8sCmd)
	k8sCmd.Flags().StringVar(&opt.App, "app", "scope", "Name of the app in Kubernetes")
	k8sCmd.Flags().StringVar(&opt.Namespace, "namespace", "default", "Name of the namespace to install in")
	k8sCmd.Flags().StringVar(&opt.Version, "version", "", "Version of scope to deploy")
	k8sCmd.Flags().StringVar(&opt.CertFile, "certfile", "/etc/certs/tls.crt", "Certificate file for TLS in the container (mounted secret)")
	k8sCmd.Flags().StringVar(&opt.KeyFile, "keyfile", "/etc/certs/tls.key", "Private key file for TLS in the container (mounted secret)")
	k8sCmd.Flags().IntVar(&opt.Port, "port", 4443, "Port to listen on")
	k8sCmd.Flags().Bool("server", false, "Run Webhook server")
	k8sCmd.Flags().BoolVar(&opt.Debug, "debug", false, "Turn on debug logging")
	metricAndEventDestFlags(k8sCmd, rc)
}
