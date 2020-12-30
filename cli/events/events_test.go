package events

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReader(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
`
	ioutil.WriteFile("event.json", []byte(rawEvent), 0644)

	in := make(chan map[string]interface{})

	go Reader("event.json", MatchAlways, in)

	event := <-in

	assert.Equal(t, "console", event["sourcetype"])
	assert.Equal(t, 1609191683.985, event["_time"])
	assert.Equal(t, "true", event["data"])

	os.Remove("event.json")
}

func TestReaderWithFilter(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"foo","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10118,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.986,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10118,"_channel":"641503557208802","data":"true"}}
`
	ioutil.WriteFile("event.json", []byte(rawEvent), 0644)

	in := make(chan map[string]interface{})
	go Reader("event.json", MatchString("foo"), in)
	events := []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)

	in = make(chan map[string]interface{})
	go Reader("event.json", MatchField("source", "foo"), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)

	in = make(chan map[string]interface{})
	go Reader("event.json", MatchField("pid", 10118), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 2)

	in = make(chan map[string]interface{})
	go Reader("event.json", MatchField("_time", 1609191683.986), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)

	in = make(chan map[string]interface{})
	go Reader("event.json", MatchAny([]MatchFunc{MatchField("pid", 10118), MatchString("foo")}...), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 3)

	in = make(chan map[string]interface{})
	go Reader("event.json", MatchAll([]MatchFunc{MatchField("pid", 10118), MatchString("foo")}...), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 0)

	in = make(chan map[string]interface{})
	go Reader("event.json", MatchAll([]MatchFunc{MatchField("pid", 10117), MatchString("foo")}...), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)
}

func TestCount(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
`

	ioutil.WriteFile("event.json", []byte(rawEvent), 0644)

	count := Count("event.json")
	assert.Equal(t, 4, count)
	os.Remove("event.json")
}
