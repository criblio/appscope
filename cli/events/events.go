package events

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/criblio/scope/util"
)

// Reader reads a newline delimited JSON documents and sends parsed documents
// to the passed out channel. It exits the process on error.
func Reader(r io.Reader, initOffset int, match func(string) bool, out chan map[string]interface{}) (int, error) {
	br, err := util.NewlineReader(r, match, func(idx int, offset int, b []byte) error {
		event, err := ParseEvent(b)
		if err != nil {
			if err.Error() == "config event" {
				return nil
			}
			return err
		}
		event["id"] = util.EncodeOffset(int64(initOffset + offset))
		out <- event
		return nil
	})
	close(out)
	return br, err
}

// ParseEvent returns an event from an array of bytes containing one event
func ParseEvent(b []byte) (map[string]interface{}, error) {
	event := map[string]interface{}{}
	err := json.Unmarshal(b, &event)
	if err != nil {
		return event, err
	}
	if eventType, typeExists := event["type"]; typeExists && eventType.(string) == "evt" {
		return event["body"].(map[string]interface{}), nil
	}
	// config event in event.json is not an error, special case
	if _, ok := event["info"]; ok {
		return event, fmt.Errorf("config event")
	}
	return event, fmt.Errorf("could not find body in event")
}
