package events

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/ahmetb/go-linq/v3"
	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/util"
	"github.com/dop251/goja"
	"github.com/fatih/color"
	"github.com/rs/zerolog/log"
)

// Define color settings
var nc = color.New

type colorOpts map[string]*color.Color

var (
	darkColorOpts = colorOpts{
		"id":      nc(color.FgBlue),
		"key":     nc(color.FgGreen),
		"val":     nc(color.FgHiWhite),
		"_time":   nc(color.FgWhite),
		"data":    nc(color.FgHiWhite),
		"proc":    nc(color.FgHiMagenta),
		"source":  nc(color.FgHiBlack),
		"default": nc(color.FgWhite),
	}

	lightColorOpts = colorOpts{
		"id":      nc(color.FgBlue),
		"key":     nc(color.FgGreen),
		"val":     nc(color.FgBlack),
		"_time":   nc(color.FgBlack),
		"data":    nc(color.FgHiBlack),
		"proc":    nc(color.FgHiMagenta),
		"source":  nc(color.FgHiWhite),
		"default": nc(color.FgBlack),
	}

	sourcetypeColors = colorOpts{
		"console": nc(color.FgHiGreen),
		"http":    nc(color.FgHiBlue),
		"fs":      nc(color.FgHiCyan),
		"net":     nc(color.FgHiRed),
		"default": nc(color.FgYellow),
	}
)

// sourcefields sets Event.Body.Data fields to be displayed by default for a given source
var sourceFields = map[string][]string{
	"http.req": {"http_host",
		"http_method",
		"http_request_content_length",
		"http_scheme",
		"http_target"},
	"http.resp": {"http_host",
		"http_method",
		"http_scheme",
		"http_target",
		"http_response_content_length",
		"http_status"},
	"http-metrics": {"duration",
		"req_per_sec"},
	"fs.open": {"file"},
	"fs.close": {"file",
		"file_read_bytes",
		"file_read_ops",
		"file_write_bytes",
		"file_write_ops"},
	"net.open": {"net_peer_ip",
		"net_peer_port",
		"net_host_ip",
		"net_host_port",
		"net_protocol",
		"net_transport"},
	"net.close": {"net_peer_ip",
		"net_peer_port",
		"net_bytes_recv",
		"net_bytes_sent",
		"net_close_reason",
		"net_protocol"},
}

const ansi = "[\u001B\u009B][[\\]()#;?]*(?:(?:(?:[a-zA-Z\\d]*(?:;[a-zA-Z\\d]*)*)?\u0007)|(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PRZcf-ntqry=><~]))"

var ansire = regexp.MustCompile(ansi)

func AnsiStrip(str string) string {
	return ansire.ReplaceAllString(str, "")
}

// EventReader reads a newline delimited JSON documents and sends parsed documents
// to the passed out channel. It exits the process on error.
func EventReader(r io.Reader, initOffset int64, match func(string) bool, out chan libscope.EventBody) (int, error) {
	br, err := util.NewlineReader(r, match, func(idx int, Offset int64, b []byte) error {
		event, err := ParseEvent(b)
		if err != nil {
			if err.Error() == "config event" {
				return nil
			}
			log.Error().Err(err).Msg("error parsing event")
			return nil
		}
		event.Id = util.EncodeOffset(initOffset + Offset)
		out <- event
		return nil
	})

	// Tell the consumer we're done
	close(out)
	return br, err
}

// ParseEvent unmarshals event text into an Event, returning only the Event.Body
func ParseEvent(b []byte) (libscope.EventBody, error) {
	var event libscope.Event
	err := json.Unmarshal(b, &event)
	if err != nil {
		return event.Body, err
	}
	if event.Type == "evt" {
		return event.Body, nil
	}
	return event.Body, fmt.Errorf("could not find body in event")
}

// EventMatch helps retrieve events from events.json
// TODO works, but now that in library, write tests
type EventMatch struct {
	Sources     []string
	Sourcetypes []string
	Match       string
	SkipEvents  int
	AllEvents   bool
	LastN       int
	Offset      int64
}

// Events matches events based on EventMatch config
func (em EventMatch) Events(file io.ReadSeeker, in chan libscope.EventBody) error {
	var err error
	if !em.AllEvents && em.Offset == 0 {
		var err error
		em.Offset, err = util.FindReverseLineMatchOffset(em.LastN, file, em.filter())
		if err != nil {
			return fmt.Errorf("Error searching for Offset: %v", err)
		}
		if em.Offset < 0 {
			em.Offset = int64(0)
		}
	}
	_, err = file.Seek(em.Offset, io.SeekStart)
	if err != nil {
		return fmt.Errorf("error seeking events file: %v", err)
	}

	// Read Events
	_, err = EventReader(file, em.Offset, em.filter(), in)
	if err != nil {
		return fmt.Errorf("error reading events: %v", err)
	}
	return nil
}

// filter filters events based on EventMatch config
func (em EventMatch) filter() func(string) bool {
	all := []util.MatchFunc{}
	if !em.AllEvents && em.SkipEvents > 0 {
		all = append(all, util.MatchSkipN(em.SkipEvents))
	}
	if em.Match != "" {
		all = append(all, util.MatchString(em.Match))
	}
	if len(em.Sources) > 0 {
		matchsources := []util.MatchFunc{}
		for _, s := range em.Sources {
			matchsources = append(matchsources, util.MatchField("source", s))
		}
		all = append(all, util.MatchAny(matchsources...))
	}
	if len(em.Sourcetypes) > 0 {
		matchsourcetypes := []util.MatchFunc{}
		for _, t := range em.Sourcetypes {
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

// PrintEvent prints a single event
func PrintEvent(in chan libscope.EventBody, jsonOut bool) {
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

// PrintEvents prints multiple events
// handles --eval
// handles --json
func PrintEvents(in chan libscope.EventBody, fields []string, sortField, eval string, jsonOut, sortReverse, allFields, forceColor bool, width int) {
	enc := json.NewEncoder(os.Stdout)
	var vm *goja.Runtime
	var prog *goja.Program
	if eval != "" {
		vm = goja.New()
		var err error
		prog, err = goja.Compile("expr", eval, false)
		util.CheckErrSprintf(err, "error compiling JavaScript expression: %v", err)
	}

	events := make([]libscope.EventBody, 0)
	for e := range in {
		if eval != "" {
			// Initialize keys and values from the event for goja query
			vm.Set("args", e.Args)
			vm.Set("cmd", e.Cmd)
			vm.Set("gid", e.Gid)
			vm.Set("groupname", e.Groupname)
			vm.Set("host", e.Host)
			vm.Set("id", e.Id)
			vm.Set("pid", e.Pid)
			vm.Set("proc", e.Proc)
			vm.Set("source", e.Source)
			vm.Set("sourcetype", e.SourceType)
			vm.Set("_time", e.Time)
			vm.Set("uid", e.Uid)
			vm.Set("username", e.Username)
			vm.Set("data", e.Data)
			v, err := vm.RunProgram(prog)
			util.CheckErrSprintf(err, "error evaluating JavaScript expression: %v", err)
			res := v.Export()
			// We support a boolean result only
			switch r := res.(type) {
			case bool:
				if !r {
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

		events = append(events, e)
	}

	// Sort events
	if len(events) > 0 && sortField != "" {
		events = sortEvents(events, sortField, sortReverse)
	}

	// Print events
	for _, e := range events {
		out := GetEventText(e, forceColor, allFields, fields, width)
		util.Printf("%s\n", out)
	}
}

// color returns the matching color or the default
func (co colorOpts) color(color string) *color.Color {
	c := co[color]
	if c == nil {
		c = co["default"]
	}
	return c
}

// GetEventText selects and colors event fields
// handles --color
// handles --fields
// handles --allfields
func GetEventText(e libscope.EventBody, forceColor, allFields bool, fields []string, width int) string {
	co := darkColorOpts // Hard coded for now
	if forceColor {
		color.NoColor = false
	}

	out := co.color("id").Sprintf("[%s] ", e.Id)
	out += co.color("time").Sprintf("%s ", util.FormatTimestamp(e.Time))
	out += co.color("proc").Sprintf("%s ", e.Proc)
	out += sourcetypeColors.color(e.SourceType).Sprintf("%s ", e.SourceType)
	out += co.color("source").Sprintf("%s ", e.Source)

	// Truncate data field
	truncLen := width - len(AnsiStrip(out)) - 1

	if len(fields) > 0 {
		out += getEventDataText(e.Data, co, fields, truncLen)
	} else {
		out += getEventDataText(e.Data, co, sourceFields[e.Source], truncLen)
	}

	if allFields {
		out += fmt.Sprintf("[[%s]]", getEventDataText(e.Data, co, []string{}, truncLen))
	}
	return out
}

// getEventDataText selects and colors data fields
// We support a key:value map (per normal events) and a string (per notices)
func getEventDataText(e interface{}, co colorOpts, onlyFields []string, truncLen int) string {
	kv := ""

	switch d := e.(type) {
	case map[string]interface{}:
		// Truncate message field if present
		maxLen := truncLen - len(" message:\"\"")
		if msg, ok := d["message"]; ok {
			trimmed := strings.TrimSpace(fmt.Sprintf("%s", msg))
			removeRN := strings.ReplaceAll(trimmed, "\r\n", " ")
			removeN := strings.ReplaceAll(removeRN, "\n", " ")
			removeR := strings.ReplaceAll(removeN, "\r", " ")
			stripped := AnsiStrip(removeR)
			d["message"] = util.TruncWithEllipsis(stripped, maxLen)
		}

		// Which keys do we want to display?
		keys := make([]string, 0, len(d))
		if len(onlyFields) > 0 {
			keys = onlyFields
		} else {
			for k := range d {
				keys = append(keys, k)
			}
			sort.Strings(keys)
		}

		// Let's get their values
		for idx, k := range keys {
			if _, ok := d[k]; ok {
				if idx > 0 {
					kv += co.color("key").Sprint(" ")
				}
				kv += co.color("key").Sprintf("%s", k)
				kv += co.color("val").Sprintf(":%s", util.InterfaceToString(d[k]))

				// Delete from e so allfields knows what didn't get written yet
				delete(d, k)
			}
		}
	case string:
		trimmed := strings.TrimSpace(fmt.Sprintf("%s", d))
		removeRN := strings.ReplaceAll(trimmed, "\r\n", " ")
		removeN := strings.ReplaceAll(removeRN, "\n", " ")
		removeR := strings.ReplaceAll(removeN, "\r", " ")
		stripped := AnsiStrip(removeR)
		kv += co.color("val").Sprintf("%s", util.InterfaceToString(stripped))

		// Delete from e so allfields knows what didn't get written yet
		e = ""
	}

	return kv
}

// sortEvents sorts an array of events
// handles --sort
// handles --reverse
func sortEvents(arr []libscope.EventBody, sortField string, sortReverse bool) []libscope.EventBody {
	sortFunc := func(i, j interface{}) bool {
		f1 := util.GetJSONField(i, sortField)
		if f1 == nil {
			util.ErrAndExit("field %s does not exist in top level object %s", sortField, util.JSONBytes(i))
		}
		f2 := util.GetJSONField(j, sortField)
		if f2 == nil {
			util.ErrAndExit("field %s does not exist in top level object %s", sortField, util.JSONBytes(j))
		}
		var ret bool
		switch f1.Value().(type) {
		case int:
			ret = f1.Value().(int) > f2.Value().(int)
		case int64:
			ret = f1.Value().(int64) > f2.Value().(int64)
		case string:
			ret = f1.Value().(string) > f2.Value().(string)
		case float64:
			ret = f1.Value().(float64) > f2.Value().(float64)
		case time.Time:
			ret = f1.Value().(time.Time).After(f2.Value().(time.Time))
		case time.Duration:
			ret = f1.Value().(time.Duration) > f2.Value().(time.Duration)
		default:
			util.ErrAndExit("unsupported field type %s", sortField)
		}
		if sortReverse {
			return !ret
		}
		return ret
	}

	sorted := make([]libscope.EventBody, 0)
	from := linq.From(arr)
	from = from.Sort(sortFunc)
	from.ToSlice(&sorted)
	return sorted
}
