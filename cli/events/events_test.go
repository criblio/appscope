package events

import (
	"bytes"
	"testing"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/util"
	"github.com/stretchr/testify/assert"
)

func TestEventReader(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
`

	in := make(chan libscope.EventBody)

	r := bytes.NewBuffer([]byte(rawEvent))
	go EventReader(r, 0, util.MatchAlways, in)

	event := <-in

	assert.Equal(t, "console", event.SourceType)
	assert.Equal(t, 1609191683.985, event.Time)
	assert.Equal(t, "true", event.Data)
}

func TestParseEvent(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
`
	event, err := ParseEvent([]byte(rawEvent))
	assert.NoError(t, err)

	assert.Equal(t, "console", event.SourceType)
	assert.Equal(t, 1609191683.985, event.Time)
	assert.Equal(t, "true", event.Data)
}

func TestReaderWithFilter(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"foo","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10118,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.986,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10118,"_channel":"641503557208802","data":"true"}}
`

	testFilter := func(filter func(string) bool, len int) {
		in := make(chan libscope.EventBody)
		r := bytes.NewBuffer([]byte(rawEvent))
		go EventReader(r, 0, filter, in)
		events := []libscope.EventBody{}
		for e := range in {
			events = append(events, e)
		}
		assert.Len(t, events, len)
	}
	testFilter(util.MatchString("foo"), 1)
	testFilter(util.MatchField("source", "foo"), 1)
	testFilter(util.MatchField("pid", 10118), 2)
	testFilter(util.MatchField("_time", 1609191683.986), 1)
	testFilter(util.MatchAny([]util.MatchFunc{util.MatchField("pid", 10118), util.MatchString("foo")}...), 3)
	testFilter(util.MatchAll([]util.MatchFunc{util.MatchField("pid", 10118), util.MatchString("foo")}...), 0)
	testFilter(util.MatchAll([]util.MatchFunc{util.MatchField("pid", 10117), util.MatchString("foo")}...), 1)
}

// TestAnsiStrip
// Assertions
// - Ansi escape codes are removed
func TestAnsiStrip(t *testing.T) {

	tests := []struct {
		name      string
		input     string
		expOutput string
	}{
		{name: "codes", input: "\x1b[38;5;140m foo\x1b[0m bar", expOutput: " foo bar"},
		{name: "red", input: "\u001b[30m", expOutput: ""},
		{name: "green", input: "\u001b[32m", expOutput: ""},
		{name: "blue", input: "\u001b[34m", expOutput: ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := AnsiStrip(tc.input)
			assert.Equal(t, tc.expOutput, out)
		})
	}
}

// TestPrintEvents
// Assertions
// - Event output is as expected
// - Event format is as expected
func TestPrintEvents(t *testing.T) {

	// PrintEvents(in chan libscope.EventBody, fields []string, sortField, eval string, jsonOut, sortReverse, allFields, forceColor bool, width int) {

}

// TestSortEvents
// Assertions
// - Event sorting is as expected
func TestSortEvents(t *testing.T) {

	tests := []struct {
		name      string
		sortField string
		reverse   bool
		events    []libscope.EventBody
		expOutput []libscope.EventBody
	}{
		{name: "sort by time", sortField: "_time", reverse: false,
			events:    []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}},
			expOutput: []libscope.EventBody{{Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}, {Id: "1", Host: "a", Time: 1.01, Pid: 1}},
		},
		{name: "sort by reverse time", sortField: "_time", reverse: true,
			events:    []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}},
			expOutput: []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}},
		},
		{name: "sort by host", sortField: "host", reverse: false,
			events:    []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}},
			expOutput: []libscope.EventBody{{Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}, {Id: "1", Host: "a", Time: 1.01, Pid: 1}},
		},
		{name: "sort by reverse host", sortField: "host", reverse: true,
			events:    []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}},
			expOutput: []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}},
		},
		{name: "sort by pid", sortField: "pid", reverse: false,
			events:    []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}},
			expOutput: []libscope.EventBody{{Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}, {Id: "1", Host: "a", Time: 1.01, Pid: 1}},
		},
		{name: "sort by reverse pid", sortField: "pid", reverse: true,
			events:    []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}},
			expOutput: []libscope.EventBody{{Id: "1", Host: "a", Time: 1.01, Pid: 1}, {Id: "3", Host: "c", Time: 1.03, Pid: 3}, {Id: "2", Host: "f", Time: 1.04, Pid: 5}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out := sortEvents(tc.events, tc.sortField, tc.reverse)
			assert.Equal(t, tc.expOutput, out)
		})
	}
}
