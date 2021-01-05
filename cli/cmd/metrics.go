package cmd

import (
	"fmt"
	"os"

	"github.com/criblio/scope/metrics"
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
		names, _ := cmd.Flags().GetStringSlice("metric")
		graph, _ := cmd.Flags().GetBool("graph")
		cols, _ := cmd.Flags().GetBool("cols")

		sessions := sessionByID(id)

		if graph && len(names) == 0 {
			util.ErrAndExit("must specify a metric to graph")
		}

		file, err := os.Open(sessions[0].MetricsPath)
		util.CheckErrSprintf(err, "Invalid session. Command likely exited without capturing metric data.\nerror opening metrics file: %v", err)
		in := make(chan metrics.Metric)
		offsetChan := make(chan int)
		filters := []util.MatchFunc{}
		if len(names) > 0 {
			for _, n := range names {
				filters = append(filters, util.MatchString(n))
			}
		} else {
			filters = []util.MatchFunc{util.MatchAlways}
		}
		go func() {
			offset, err := metrics.Reader(file, util.MatchAny(filters...), in)
			util.CheckErrSprintf(err, "error reading metrics: %v", err)
			offsetChan <- offset
		}()
		util.CheckErrSprintf(err, "error reading metrics from file: %v", err)

		// Filter metrics to match metrics
		values := []float64{}
		metricCols := []map[string]interface{}{}
		mm := []metrics.Metric{}

		for m := range in {
			if cols && m.Name == "proc.cpu" { // use proc.cpu as a breaker for batches of metrics
				metricCols = append(metricCols, map[string]interface{}{})
			}
			if len(names) > 0 {
				for _, name := range names {
					if m.Name == name {
						if graph {
							values = append(values, m.Value)
						} else if cols {
							metricCols[len(metricCols)-1][m.Name] = m.Value
						} else {
							mm = append(mm, m)
						}
					}
				}
			} else {
				mm = append(mm, m)
			}
		}

		if graph {
			fmt.Println(asciigraph.Plot(values, asciigraph.Height(20)))
			os.Exit(0)
		}

		if cols {
			ofCols := []util.ObjField{}
			for _, col := range names {
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
	metricsCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
	metricsCmd.Flags().StringSliceP("metric", "m", []string{}, "Display only supplied metrics")
	metricsCmd.Flags().BoolP("graph", "g", false, "Graph this metric")
	metricsCmd.Flags().BoolP("cols", "c", false, "Display metrics as columns")
	RootCmd.AddCommand(metricsCmd)
}
