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
	Use:     "k8s",
	Short:   "Prints k8s configurations to install scope to automatically instrument pods",
	Long:    `Prints k8s configurations to install scope to automatically instrument pods`,
	Example: `scope k8s`,
	Args:    cobra.NoArgs,
	Hidden:  true,
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
