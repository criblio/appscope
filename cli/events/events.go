package events

import (
	"encoding/json"
	"io"

	"github.com/criblio/scope/util"
)

// Reader reads a newline delimited JSON documents and sends parsed documents
// to the passed out channel. It exits the process on error.
func Reader(r io.Reader, match func(string) bool, out chan map[string]interface{}) (int, error) {
	br, err := util.NewlineReader(r, match, func(idx int, b []byte) error {
		event := map[string]interface{}{}
		err := json.Unmarshal(b, &event)
		if err != nil {
			return err
		}
		if eventType, typeExists := event["type"]; typeExists && eventType.(string) == "evt" {
			body := event["body"].(map[string]interface{})
			body["id"] = idx
			out <- body
		}
		return nil
	})
	close(out)
	return br, err
}
