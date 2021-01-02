package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/criblio/scope/events"
	"github.com/criblio/scope/util"
	"github.com/fatih/color"
	"github.com/spf13/cobra"
)

var nc = color.New

type colorOpts map[string]*color.Color

var darkColorOpts = colorOpts{
	"key":     nc(color.FgGreen),
	"val":     nc(color.FgHiWhite),
	"_time":   nc(color.FgWhite),
	"data":    nc(color.FgHiWhite),
	"proc":    nc(color.FgHiMagenta),
	"source":  nc(color.FgHiBlack),
	"default": nc(color.FgWhite),
}

var lightColorOpts = colorOpts{
	"key":     nc(color.FgGreen),
	"val":     nc(color.FgBlack),
	"_time":   nc(color.FgBlack),
	"data":    nc(color.FgHiBlack),
	"proc":    nc(color.FgHiMagenta),
	"source":  nc(color.FgHiWhite),
	"default": nc(color.FgBlack),
}

var sourcetypeColors = colorOpts{
	"console": nc(color.FgHiGreen),
	"http":    nc(color.FgHiBlue),
	"default": nc(color.FgYellow),
}

var sourcetypeFields = map[string][]string{
	"http": {"http.host", "http.method", "http.request_content_length", "http.scheme", "http.target", "http.response_content_length", "http.status", "duration", "req_per_sec"},
}

// color returns the matching color or the default
func (co colorOpts) color(color string) *color.Color {
	c := co[color]
	if c == nil {
		c = co["default"]
	}
	return c
}

var co colorOpts

// eventsCmd represents the events command
var eventsCmd = &cobra.Command{
	Use:     "events [flags] ([id])",
	Short:   "Outputs events for a session",
	Long:    `Outputs events for a session`,
	Example: `scope events`,
	Args:    cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		sources, _ := cmd.Flags().GetStringSlice("source")
		sourcetypes, _ := cmd.Flags().GetStringSlice("sourcetype")
		lastN, _ := cmd.Flags().GetInt("last")
		follow, _ := cmd.Flags().GetBool("follow")
		forceColor, _ := cmd.Flags().GetBool("color")

		if forceColor {
			color.NoColor = false
		}

		sessions := sessionByID(id)

		// Count events to figure out our starting place if we want the last N events or a specific event
		count, err := util.CountLines(sessions[0].EventsPath)
		util.CheckErrSprintf(err, "Invalid session. Command likely exited without capturing event data.\nerror opening events file: %v", err)
		skipEvents := 0
		eventID := 0
		if len(args) > 0 {
			var err error
			eventID, err = strconv.Atoi(args[0])
			util.CheckErrSprintf(err, "could not parse %s as event ID (integer): %v", args[0], err)
			skipEvents = eventID
		} else if lastN > -1 {
			skipEvents = count - lastN - 1
		}
		// How many events do we have? Determines width of the id field
		idChars := len(fmt.Sprintf("%d", count))

		// EventMatch contains parameters to match again, used in the filter function
		// sent to events.Reader
		em := eventMatch{
			sources:     sources,
			sourcetypes: sourcetypes,
			skipEvents:  skipEvents,
		}

		// Open events.json file
		file, err := os.Open(sessions[0].EventsPath)
		util.CheckErrSprintf(err, "error opening events file: %v", err)
		// Read Events
		in := make(chan map[string]interface{})
		offsetChan := make(chan int)
		go func() {
			offset, err := events.Reader(file, em.filter(), in)
			util.CheckErrSprintf(err, "error reading events: %v", err)
			offsetChan <- offset
		}()

		// If we were passed an EventID, read that one event, print and exit
		if eventID > 0 {
			printEvent(cmd, in)
			os.Exit(0)
		}

		// Print events from events.json
		printEvents(cmd, idChars, in)
		file.Close()

		if follow {
			offset := <-offsetChan
			tailFile, err := util.NewTailReader(sessions[0].EventsPath, offset)
			util.CheckErrSprintf(err, "error opening events file for tailing: %v", err)
			in = make(chan map[string]interface{})
			em.skipEvents = 0
			go events.Reader(tailFile, em.filter(), in)
			printEvents(cmd, idChars, in)
		}
	},
}

func printEvent(cmd *cobra.Command, in chan map[string]interface{}) {
	jsonOut, _ := cmd.Flags().GetBool("json")
	enc := json.NewEncoder(os.Stdout)
	e := <-in
	if jsonOut {
		enc.Encode(e)
	} else {
		fields := []util.ObjField{
			{Name: "ID", Field: "id"},
			{Name: "Proc", Field: "proc"},
			{Name: "Pid", Field: "pid"},
			{Name: "Time", Field: "_time", Transform: func(obj interface{}) string { return util.FormatTimestamp(obj.(float64)) }},
			{Name: "Command", Field: "cmd"},
			{Name: "Source", Field: "source"},
			{Name: "Sourcetype", Field: "sourcetype"},
			{Name: "Host", Field: "host"},
			{Name: "Channel", Field: "_channel"},
			{Name: "Data", Field: "data"},
		}
		util.PrintObj(fields, e)
	}
}

var lastID = 0 // HACK until we can get some real IDs

func printEvents(cmd *cobra.Command, idChars int, in chan map[string]interface{}) {
	jsonOut, _ := cmd.Flags().GetBool("json")
	allFields, _ := cmd.Flags().GetBool("allfields")
	enc := json.NewEncoder(os.Stdout)
	llastID := 0
	for e := range in {
		if jsonOut {
			enc.Encode(e)
			continue
		}
		out := color.BlueString("[%"+strconv.Itoa(idChars)+"d] ", e["id"].(int)+lastID)
		if _, timeExists := e["_time"]; timeExists {
			timeFp := e["_time"].(float64)
			e["_time"] = util.FormatTimestamp(timeFp)
		}
		noop := func(orig interface{}) interface{} { return orig }
		out += colorToken(e, "_time", noop)
		out += colorToken(e, "proc", noop)
		out += sourcetypeColorToken(e)
		out += colorToken(e, "source", noop)
		out += colorToken(e, "data", func(orig interface{}) interface{} {
			ret := ""
			switch orig.(type) {
			case string:
				ret = ansiStrip(strings.TrimSpace(fmt.Sprintf("%s", orig)))
			case map[string]interface{}:
				ret = colorMap(orig.(map[string]interface{}), sourcetypeFields[e["sourcetype"].(string)])
			}
			return ret
		})

		// Delete uninteresting fields
		llastID = e["id"].(int)
		delete(e, "id")
		delete(e, "_channel")
		delete(e, "sourcetype")

		if allFields {
			out += fmt.Sprintf("[[%s]]", colorMap(e, []string{}))
		}
		fmt.Printf("%s\n", out)
	}
	lastID = llastID
}

func colorToken(e map[string]interface{}, field string, transform func(interface{}) interface{}) string {
	val := e[field]
	delete(e, field)
	return co.color(field).Sprintf("%s ", transform(val))
}

func sourcetypeColorToken(e map[string]interface{}) string {
	st := e["sourcetype"].(string)
	return sourcetypeColors.color(st).Sprintf("%s ", st)
}

func colorMap(e map[string]interface{}, onlyFields []string) string {
	kv := ""
	keys := make([]string, 0, len(e))
	for k := range e {
		if len(onlyFields) > 0 {
			for _, field := range onlyFields {
				if k == field {
					keys = append(keys, k)
				}
			}
		} else {
			keys = append(keys, k)
		}

	}
	sort.Strings(keys)
	for idx, k := range keys {
		if idx > 0 {
			kv += " "
		}
		kv += co.color("key").Sprintf("%s", k)
		kv += co.color("val").Sprintf(":%s", interfaceToString(e[k]))
	}
	return kv
}

func interfaceToString(i interface{}) string {
	switch v := i.(type) {
	case float64:
		if v-math.Floor(v) < 0.000001 && v < 1e9 {
			return fmt.Sprintf("%d", int(v))
		}
		return fmt.Sprintf("%g", v)
	case string:
		stringFmt := "%s"
		if strings.Contains(v, " ") {
			stringFmt = "%q"
		}
		return fmt.Sprintf(stringFmt, v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

const ansi = "[\u001B\u009B][[\\]()#;?]*(?:(?:(?:[a-zA-Z\\d]*(?:;[a-zA-Z\\d]*)*)?\u0007)|(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PRZcf-ntqry=><~]))"

var ansire = regexp.MustCompile(ansi)

func ansiStrip(str string) string {
	return ansire.ReplaceAllString(str, "")
}

type eventMatch struct {
	sources     []string
	sourcetypes []string
	skipEvents  int
}

func (em eventMatch) filter() func(string) bool {
	idx := 0
	return func(line string) bool {
		idx++
		if em.skipEvents > 0 && idx < em.skipEvents {
			return false
		}
		all := []util.MatchFunc{}
		if len(em.sources) > 0 {
			matchsources := []util.MatchFunc{}
			for _, s := range em.sources {
				matchsources = append(matchsources, util.MatchField("source", s))
			}
			all = append(all, util.MatchAny(matchsources...))
		}
		if len(em.sourcetypes) > 0 {
			matchsourcetypes := []util.MatchFunc{}
			for _, t := range em.sourcetypes {
				matchsourcetypes = append(matchsourcetypes, util.MatchField("sourcetype", t))
			}
			all = append(all, util.MatchAny(matchsourcetypes...))
		}
		if len(all) == 0 {
			all = append(all, util.MatchAlways)
		}
		return util.MatchAll(all...)(line)
	}
}

func init() {
	eventsCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
	eventsCmd.Flags().StringSliceP("source", "s", []string{}, "Display events matching supplied sources")
	eventsCmd.Flags().StringSliceP("sourcetype", "t", []string{}, "Display events matching supplied sourcetypes")
	eventsCmd.Flags().Bool("allfields", false, "Displaying hidden fields")
	eventsCmd.Flags().IntP("last", "n", 20, "Show last <n> events")
	eventsCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	eventsCmd.Flags().BoolP("follow", "f", false, "Follow a file, like tail -f")
	eventsCmd.Flags().Bool("color", false, "Force color on (if tty detection fails or pipeing)")
	RootCmd.AddCommand(eventsCmd)

	if co == nil {
		co = darkColorOpts // Hard coded for now
	}
}
