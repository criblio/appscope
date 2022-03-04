package events

import (
	"bytes"
	"testing"

	"github.com/criblio/scope/libscope"
	"github.com/criblio/scope/util"
	"github.com/stretchr/testify/assert"
)

func TestReader(t *testing.T) {
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
