package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strings"

	"golang.org/x/crypto/ssh/terminal"

	"github.com/criblio/scope/events"
	"github.com/criblio/scope/util"
	"github.com/fatih/color"
	"github.com/spf13/cobra"

	"github.com/dop251/goja"
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
	"fs":      nc(color.FgHiCyan),
	"net":     nc(color.FgHiRed),
	"default": nc(color.FgYellow),
}

var sourceFields = map[string][]string{
	"http-req": {"http.host",
		"http.method",
		"http.request_content_length",
		"http.scheme",
		"http.target"},
	"http-resp": {"http.host",
		"http.method",
		"http.scheme",
		"http.target",
		"http.response_content_length",
		"http.status"},
	"http-metrics": {"duration",
		"req_per_sec"},
	"fs.open": {"file.name"},
	"fs.close": {"file.name",
		"file.read_bytes",
		"file.read_ops",
		"file.write_bytes",
		"file.write_ops"},
	"net.conn.open": {"net.peer.ip",
		"net.peer.port",
		"net.host.ip",
		"net.host.port",
		"net.protocol",
		"net.transport"},
	"net.conn.close": {"net.peer.ip",
		"net.peer.port",
		"net.bytes_recv",
		"net.bytes_sent",
		"net.close.reason",
		"net.protocol"},
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
	Use:   "events [flags] ([eventId])",
	Short: "Outputs events for a session",
	Long: `Outputs events for a session. Detailed information about each event can be obtained by inputting the Event ID (by default, in blue 
in []'s at the left) as a positional parameter. Filters can be provided to narrow down by source (e.g. http, net, fs, console) 
or source (e.g. fs.open, stdout, net.conn.open). JavaScript expressions can be used to further refine the query and express logic.`,
	Example: `scope events
scope events -t http
scope events -s stderr
scope events -e 'sourcetype!="net"'
scope events -n 1000 -e 'sourcetype!="console" && source.indexOf("cribl.log") == -1 && (data["file.name"] || "").indexOf("/proc") == -1'`,
	Args: cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		id, _ := cmd.Flags().GetInt("id")
		sources, _ := cmd.Flags().GetStringSlice("source")
		sourcetypes, _ := cmd.Flags().GetStringSlice("sourcetype")
		match, _ := cmd.Flags().GetString("match")
		lastN, _ := cmd.Flags().GetInt("last")
		follow, _ := cmd.Flags().GetBool("follow")
		forceColor, _ := cmd.Flags().GetBool("color")
		allEvents, _ := cmd.Flags().GetBool("all")

		if forceColor {
			color.NoColor = false
		}

		sessions := sessionByID(id)

		// Count events to figure out our starting place if we want the last N events or a specific event
		count, err := util.CountLines(sessions[0].EventsPath)
		util.CheckErrSprintf(err, "Invalid session. Command likely exited without capturing event data.\nerror opening events file: %v", err)
		skipEvents := 0
		offset := 0
		if len(args) > 0 {
			offset64, err := util.DecodeOffset(args[0])
			util.CheckErrSprintf(err, "Could not decode encoded offset: %v", err)
			offset = int(offset64)
			allEvents = false
		} else if lastN > -1 {
			skipEvents = count - lastN + 1
		}
		// How many events do we have? Determines width of the id field
		idChars := len(fmt.Sprintf("%d", count))

		// EventMatch contains parameters to match again, used in the filter function
		// sent to events.Reader
		em := eventMatch{
			sources:     sources,
			sourcetypes: sourcetypes,
			skipEvents:  skipEvents,
			allEvents:   allEvents,
			match:       match,
		}

		// Open events.json file
		var file util.ReadSeekCloser
		if follow {
			file, err = util.NewTailReader(sessions[0].EventsPath)
		} else {
			file, err = os.Open(sessions[0].EventsPath)
		}
		util.CheckErrSprintf(err, "error opening events file: %v", err)
		_, err = file.Seek(int64(offset), os.SEEK_SET)
		util.CheckErrSprintf(err, "error seeking events file: %v", err)

		// Read Events
		in := make(chan map[string]interface{})
		go func() {
			_, err := events.Reader(file, offset, em.filter(), in)
			util.CheckErrSprintf(err, "error reading events: %v", err)
		}()

		// If we were passed an EventID, read that one event, print and exit
		if len(args) > 0 {
			printEvent(cmd, in)
			os.Exit(0)
		}

		// Print events from events.json
		printEvents(cmd, idChars, in)
		file.Close()
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

func printEvents(cmd *cobra.Command, idChars int, in chan map[string]interface{}) {
	jsonOut, _ := cmd.Flags().GetBool("json")
	allFields, _ := cmd.Flags().GetBool("allfields")
	eval, _ := cmd.Flags().GetString("eval")
	enc := json.NewEncoder(os.Stdout)
	termWidth, _, err := terminal.GetSize(0)
	util.CheckErrSprintf(err, "error getting terminal width: %v", err)

	var vm *goja.Runtime
	var prog *goja.Program
	if eval != "" {
		vm = goja.New()
		prog, err = goja.Compile("expr", eval, false)
		util.CheckErrSprintf(err, "error compiling JavaScript expression: %v", err)
	}
	for e := range in {
		if eval != "" {
			for k, v := range e {
				vm.Set(k, v)
			}
			v, err := vm.RunProgram(prog)
			util.CheckErrSprintf(err, "error evaluating JavaScript expression: %v", err)
			res := v.Export()
			switch res.(type) {
			case bool:
				if !res.(bool) {
					continue
				}
			default:
				util.ErrAndExit("error: JavaScript return value is not boolean")
			}
		}
		if jsonOut {
			enc.Encode(e)
			continue
		}
		out := getEventText(e, termWidth, allFields)
		fmt.Printf("%s\n", out)
	}
}

func getEventText(e map[string]interface{}, width int, allFields bool) string {
	out := color.BlueString("[%s] ", e["id"].(string))
	if _, timeExists := e["_time"]; timeExists {
		timeFp := e["_time"].(float64)
		e["_time"] = util.FormatTimestamp(timeFp)
	}
	source := e["source"]
	noop := func(orig interface{}) interface{} { return orig }
	out += colorToken(e, "_time", noop)
	out += colorToken(e, "proc", noop)
	out += sourcetypeColorToken(e)
	out += colorToken(e, "source", noop)
	truncLen := width - len(ansiStrip(out)) - 1
	out += colorToken(e, "data", func(orig interface{}) interface{} {
		ret := ""
		switch orig.(type) {
		case string:
			ret = ansiStrip(strings.TrimSpace(fmt.Sprintf("%s", orig)))
			ret = util.TruncWithElipsis(ret, truncLen)
		case map[string]interface{}:
			ret = colorMap(orig.(map[string]interface{}), sourceFields[source.(string)])
		}
		return ret
	})

	// Delete uninteresting fields
	delete(e, "id")
	delete(e, "_channel")
	delete(e, "sourcetype")

	if allFields {
		out += fmt.Sprintf("[[%s]]", colorMap(e, []string{}))
	}
	return out
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
	if len(onlyFields) > 0 {
		keys = onlyFields
	} else {
		for k := range e {
			keys = append(keys, k)
		}
		sort.Strings(keys)
	}
	for idx, k := range keys {
		if _, ok := e[k]; ok {
			if idx > 0 {
				kv += " "
			}
			kv += co.color("key").Sprintf("%s", k)
			kv += co.color("val").Sprintf(":%s", interfaceToString(e[k]))
		}
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
	match       string
	skipEvents  int
	allEvents   bool
}

func (em eventMatch) filter() func(string) bool {
	all := []util.MatchFunc{}
	if !em.allEvents && em.skipEvents > 0 {
		all = append(all, util.MatchSkipN(em.skipEvents))
	}
	if em.match != "" {
		all = append(all, util.MatchString(em.match))
	}
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
	return func(line string) bool {
		return util.MatchAll(all...)(line)
	}
}

func init() {
	eventsCmd.Flags().IntP("id", "i", -1, "Display info from specific from session ID")
	eventsCmd.Flags().StringSliceP("source", "s", []string{}, "Display events matching supplied sources")
	eventsCmd.Flags().StringSliceP("sourcetype", "t", []string{}, "Display events matching supplied sourcetypes")
	eventsCmd.Flags().StringP("match", "m", "", "Display events containing supplied string")
	eventsCmd.Flags().Bool("allfields", false, "Displaying hidden fields")
	eventsCmd.Flags().IntP("last", "n", 20, "Show last <n> events")
	eventsCmd.Flags().BoolP("json", "j", false, "Output as newline delimited JSON")
	eventsCmd.Flags().BoolP("follow", "f", false, "Follow a file, like tail -f")
	eventsCmd.Flags().Bool("color", false, "Force color on (if tty detection fails or pipeing)")
	eventsCmd.Flags().BoolP("all", "a", false, "Show all events")
	eventsCmd.Flags().StringP("eval", "e", "", "Evaluate JavaScript expression against event. Must return truthy to print event.")
	RootCmd.AddCommand(eventsCmd)

	if co == nil {
		co = darkColorOpts // Hard coded for now
	}
}
