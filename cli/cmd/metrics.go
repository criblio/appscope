package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/metrics"
	"github.com/criblio/scope/ps"
	"github.com/criblio/scope/util"
	"github.com/guptarohit/asciigraph"
	"github.com/spf13/cobra"
)

// metricsCmd represents the metrics command
var metricsCmd = &cobra.Command{
	Use:     "metrics [flags]",
	Short:   "Outputs metrics for a session",
	Long:    `Outputs metrics for a session`,
	Example: `scope metrics`,
	Args:    cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		name, _ := cmd.Flags().GetString("name")
		graph, _ := cmd.Flags().GetString("graph")
		cols, _ := cmd.Flags().GetStringSlice("cols")

		var sessions ps.SessionList
		if id == -1 {
			sessions = ps.GetSessions().Last(1)
		} else {
			sessions = ps.GetSessions().ID(id)
		}
		sessionCount := len(sessions)
		if sessionCount != 1 {
			util.ErrAndExit("error expected a single session, saw: %d", sessionCount)
		}
		allm := metrics.ReadMetricsFromFile(sessions[0].MetricsPath)

		// Filter metrics to match metrics
		mm := []metrics.Metric{}
		values := []float64{}
		metricCols := []map[string]interface{}{}
		if name != "" || graph != "" || len(cols) > 0 {
			for _, m := range allm {
				if m.Name == "proc.cpu" {
					metricCols = append(metricCols, map[string]interface{}{})
				}
				if name != "" && m.Name == name {
					mm = append(mm, m)
				}
				if graph != "" && m.Name == graph {
					values = append(values, m.Value)
				}
				for _, col := range cols {
					// fmt.Printf("col: %s\n", col)
					if m.Name == col {
						metricCols[len(metricCols)-1][m.Name] = m.Value
					}
				}

			}
		} else {
			mm = allm
		}

		if graph != "" {
			fmt.Println(asciigraph.Plot(values, asciigraph.Height(20)))
			os.Exit(0)
		}

		if len(cols) > 0 {
			ofCols := []util.ObjField{}
			for _, col := range cols {
				ofCols = append(ofCols, util.ObjField{Name: col, Field: col})
			}
			util.PrintObj(ofCols, metricCols)
			os.Exit(0)
		}

		util.PrintObj([]util.ObjField{
			{Name: "Name", Field: "name"},
			{Name: "Value", Field: "value"},
			{Name: "Type", Field: "type"},
			{Name: "Unit", Field: "unit"},
			{Name: "Tags", Field: "tags", Transform: func(obj interface{}) string {
				ret := ""
				for _, t := range obj.([]metrics.MetricTag) {
					ret += fmt.Sprintf("%s: %s,", t.Name, t.Value)
				}
				if len(ret) > 0 {
					ret = ret[:len(ret)-1]
				}
				return ret
			}},
		}, mm)
	},
}

func init() {
	metricsCmd.Flags().Int("id", -1, "Display metrics from session ID")
	metricsCmd.Flags().StringP("name", "n", "", "Returns matching metrics")
	metricsCmd.Flags().String("graph", "", "Graph this metric")
	metricsCmd.Flags().StringSlice("cols", []string{}, "Display supplied metrics as columns")
	RootCmd.AddCommand(metricsCmd)
}
