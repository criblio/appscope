package cmd

import (
	"bufio"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"os"
	"sort"
	"strings"
	"sync"

	linq "github.com/ahmetb/go-linq/v3"
	"github.com/criblio/scope/metrics"
	"github.com/criblio/scope/util"
	"github.com/guptarohit/asciigraph"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh/terminal"
)

/* Args Matrix (X disallows)
 *          id metric graph cols uniq
 * id       -
 * metric      -
 * graph              -     X    X
 * cols               X     -
 * uniq               X          -
 */

// metricsCmd represents the metrics command
var metricsCmd = &cobra.Command{
	Use:   "metrics [flags]",
	Short: "Outputs metrics for a session",
	Long:  `Outputs metrics for a session.`,
	Example: `scope metrics
scope metrics -m net.error,fs.error
scope metrics -m net.tx -g
	`,
	Args: cobra.NoArgs,
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		names, _ := cmd.Flags().GetStringSlice("metric")
		graph, _ := cmd.Flags().GetBool("graph")
		cols, _ := cmd.Flags().GetBool("cols")
		uniq, _ := cmd.Flags().GetBool("uniq")

		// Disallow bad argument combinations (see Arg Matrix at top of file)
		if graph && cols {
			helpErrAndExit(cmd, "Cannot specify --graph and --cols")
		} else if graph && uniq {
			helpErrAndExit(cmd, "Cannot specify --graph and --uniq")
		} else if graph && len(names) == 0 {
			helpErrAndExit(cmd, "Must specify metric names with --graph")
		} else if cols && len(names) == 0 {
			helpErrAndExit(cmd, "Must specify metric names with --cols")
		}

		sessions := sessionByID(id)

		file, err := os.Open(sessions[0].MetricsPath)
		if err != nil && strings.Contains(err.Error(), "metrics.json: no such file or directory") {
			if util.CheckFileExists(sessions[0].MetricsDestPath) {
				dest, _ := ioutil.ReadFile(sessions[0].MetricsDestPath)
				fmt.Printf("Metrics were output to %s\n", dest)
				os.Exit(0)
			}
			promptClean(sessions[0:1])
		}
		defer file.Close()

		// Check for empty metrics file
		fstat, err := file.Stat()
		if err != nil {
			util.ErrAndExit("Couldn't stat file: %s", file.Name())
		}
		if fstat.Size() == 0 {
			fmt.Println("No metrics available")
			os.Exit(0)
		}

		in := make(chan metrics.Metric)
		// offsetChan := make(chan int)
		filters := []util.MatchFunc{}
		if len(names) > 0 {
			for _, n := range names {
				filters = append(filters, util.MatchString(n))
			}
		} else {
			filters = []util.MatchFunc{util.MatchAlways}
		}

		go func() {
			_, err := metrics.Reader(file, util.MatchAny(filters...), in)
			util.CheckErrSprintf(err, "error reading metrics: %v", err)
			// offsetChan <- offset
		}()
		util.CheckErrSprintf(err, "error reading metrics from file: %v", err)

		// Filter metrics to match metrics
		values := []float64{}
		metricCols := []map[string]interface{}{}
		mm := []metrics.Metric{}

		for m := range in { // all metrics
			if len(names) > 0 { // if -m is used
				for _, name := range names {
					if m.Name == name { // filter out metrics we arent concerned with
						if graph {
							values = append(values, m.Value)
						} else if cols { // if -c is used
							inserted := false
							for i, column := range metricCols { // place metric value in correct place
								if _, exists := column[m.Name]; !exists {
									metricCols[i][m.Name] = m.Value
									inserted = true
									break
								}
							}
							if !inserted {
								metricCols = append(metricCols, map[string]interface{}{m.Name: m.Value})
							}
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
			if len(values) == 0 {
				util.ErrAndExit("Valid metric names required with --graph")
			}

			termWidth, _, err := terminal.GetSize(0)
			if err != nil {
				// If we cannot get the terminal size, we are dealing with redirected stdin
				// as opposed to an actual terminal, so we will assume terminal width is
				// 160, to show all columns.
				termWidth = 160
			}

			q := linq.From(values)
			max := q.Max().(float64)
			legendSize := len(fmt.Sprintf("%.0f", max)) + 4
			maxValues := termWidth - legendSize
			sampleRate := int(math.Round(float64(len(values)) / float64(maxValues)))
			if sampleRate == 0 {
				sampleRate = 1
			}

			var newValues []float64
			q.WhereIndexed(
				func(idx int, _ interface{}) bool {
					return idx%sampleRate == 0
				},
			).ToSlice(&newValues)
			if maxValues > len(newValues) {
				maxValues = len(newValues)
			}
			fmt.Println(asciigraph.Plot(newValues[:maxValues], asciigraph.Height(20)))
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

		if uniq {
			linq.From(mm).DistinctBy(func(item interface{}) interface{} {
				return item.(metrics.Metric).Name
			}).ToSlice(&mm)
		}

		// Goroutine to intercept stdout to perform truncation and tab replacement
		r, w := io.Pipe()
		util.SetOut(w)
		scanner := bufio.NewScanner(r)
		termWidth, _, err := terminal.GetSize(0)
		if err != nil {
			// If we cannot get the terminal size, we are dealing with redirected stdin
			// as opposed to an actual terminal, so we will assume terminal width is
			// 160, to show all columns.
			termWidth = 160
		}

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			first := true
			for scanner.Scan() {
				line := strings.ReplaceAll(scanner.Text(), "\t", "    ")
				if first {
					line = util.Trunc(line, termWidth)
				} else {
					line = util.TruncWithEllipsis(line, termWidth)
				}
				fmt.Printf("%s\n", line)
				first = false
			}
		}()

		util.PrintObj([]util.ObjField{
			{Name: "Name", Field: "name"},
			{Name: "Value", Field: "value"},
			{Name: "Type", Field: "type"},
			{Name: "Unit", Field: "unit"},
			{Name: "PID", Field: "pid"},
			{Name: "Tags", Field: "tags", Transform: func(obj interface{}) string {
				ret := ""
				tags := obj.([]metrics.MetricTag)
				sort.Slice(tags, func(i, j int) bool { return tags[i].Name < tags[j].Name })
				for _, t := range tags {
					ret += fmt.Sprintf("%s: %s,", t.Name, t.Value)
				}
				if len(ret) > 0 {
					ret = ret[:len(ret)-1]
				}
				return ret
			}},
		}, mm)

		// Tell goroutine we're done then wait for it
		// otherwise you might exit before some output is printed
		w.Close() // Send EOF to scanner, so it ends for loop
		wg.Wait() // Wait for wg done
	},
}

func init() {
	metricsCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
	metricsCmd.Flags().StringSliceP("metric", "m", []string{}, "Display for specified metrics only (comma-separated)")
	metricsCmd.Flags().BoolP("graph", "g", false, "Graph this metric. Must be combined with -m")
	metricsCmd.Flags().BoolP("cols", "c", false, "Display metrics as columns. Must be combined with -m")
	metricsCmd.Flags().BoolP("uniq", "u", false, "Display first instance of each unique metric")
	RootCmd.AddCommand(metricsCmd)
}
