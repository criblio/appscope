package events

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/criblio/scope/util"
	"github.com/rs/zerolog/log"
)

// Reader reads a newline delimited JSON documents and sends parsed documents
// to the passed out channel. It exits the process on error.
func Reader(r io.Reader, initOffset int64, match func(string) bool, out chan map[string]interface{}) (int, error) {
	br, err := util.NewlineReader(r, match, func(idx int, Offset int64, b []byte) error {
		event, err := ParseEvent(b)
		if err != nil {
			if err.Error() == "config event" {
				return nil
			}
			log.Error().Err(err).Msg("error parsing event")
			return nil
		}
		event["id"] = util.EncodeOffset(initOffset + Offset)
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

func (em EventMatch) Events(file io.ReadSeeker, in chan map[string]interface{}) error {
	var err error
	if !em.AllEvents && em.Offset == 0 {
		var err error
		em.Offset, err = util.FindReverseLineMatchOffset(em.LastN, file, em.Filter())
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
	_, err = Reader(file, em.Offset, em.Filter(), in)
	if err != nil {
		return fmt.Errorf("error reading events: %v", err)
	}
	return nil
}

func (em EventMatch) Filter() func(string) bool {
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
