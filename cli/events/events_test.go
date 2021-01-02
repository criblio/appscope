package events

import (
	"io/ioutil"
	"os"
	"testing"

	"github.com/criblio/scope/util"
	"github.com/stretchr/testify/assert"
)

func TestReader(t *testing.T) {
	rawEvent := `{"type":"evt","body":{"sourcetype":"console","id":"d55805e5c25e-echo-/bin/echo true","_time":1609191683.985,"source":"stdout","host":"d55805e5c25e","proc":"echo","cmd":"/bin/echo true","pid":10117,"_channel":"641503557208802","data":"true"}}
`
	ioutil.WriteFile("event.json", []byte(rawEvent), 0644)

	in := make(chan map[string]interface{})
	file, err := os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchAlways, in)

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
	file, err := os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchString("foo"), in)
	events := []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)
	file.Close()

	in = make(chan map[string]interface{})
	file, err = os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchField("source", "foo"), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)
	file.Close()

	in = make(chan map[string]interface{})
	file, err = os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchField("pid", 10118), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 2)

	in = make(chan map[string]interface{})
	file, err = os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchField("_time", 1609191683.986), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)
	file.Close()

	in = make(chan map[string]interface{})
	file, err = os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchAny([]util.MatchFunc{util.MatchField("pid", 10118), util.MatchString("foo")}...), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 3)
	file.Close()

	in = make(chan map[string]interface{})
	file, err = os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchAll([]util.MatchFunc{util.MatchField("pid", 10118), util.MatchString("foo")}...), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 0)
	file.Close()

	in = make(chan map[string]interface{})
	file, err = os.Open("event.json")
	assert.NoError(t, err)
	go Reader(file, util.MatchAll([]util.MatchFunc{util.MatchField("pid", 10117), util.MatchString("foo")}...), in)
	events = []map[string]interface{}{}
	for e := range in {
		events = append(events, e)
	}
	assert.Len(t, events, 1)
	file.Close()

	os.Remove("event.json")
}
