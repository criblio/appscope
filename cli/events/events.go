package events

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/criblio/scope/util"
)

// Reader reads a newline delimited JSON documents and sends parsed documents
// to the passed out channel. It exits the process on error.
func Reader(r io.Reader, match func(string) bool, out chan map[string]interface{}) (int, error) {
	cr := &util.CountingReader{Reader: r}
	scanner := bufio.NewScanner(cr)
	idx := 0
	// Discard first line, which is scope setup event
	for scanner.Scan() {
		idx++
		event := map[string]interface{}{}
		if !match(scanner.Text()) {
			continue
		}
		err := json.Unmarshal(scanner.Bytes(), &event)
		if err != nil {
			return cr.BytesRead, err
		}
		if eventType, typeExists := event["type"]; typeExists && eventType.(string) == "evt" {
			body := event["body"].(map[string]interface{})
			body["id"] = idx
			out <- body
		}
	}

	if err := scanner.Err(); err != nil {
		return cr.BytesRead, err
	}
	close(out)
	return cr.BytesRead, nil
}

// MatchFunc will return true or false if a given event matches
type MatchFunc func(string) bool

// MatchAlways always returns true
func MatchAlways(in string) bool {
	return true
}

// MatchString searches for a given string
func MatchString(search string) MatchFunc {
	return func(in string) bool {
		if strings.Contains(in, search) {
			return true
		}
		return false
	}
}

// MatchField searches for a given JSON key/value
func MatchField(key string, val interface{}) MatchFunc {
	strval := ""
	switch val.(type) {
	case float64:
		strval = strconv.FormatFloat(val.(float64), 'f', -1, 64)
	case float32:
		strval = strconv.FormatFloat(val.(float64), 'f', -1, 32)
	case int64, int32, int:
		strval = fmt.Sprintf("%d", val)
	case string:
		strval = fmt.Sprintf("\"%s", val)
	default:
		strval = fmt.Sprintf("%v", val)
	}
	search := fmt.Sprintf("%s\":%s", key, strval)
	return func(in string) bool {
		// fmt.Printf("in: %s\n", in)
		if strings.Contains(in, search) {
			return true
		}
		return false
	}
}

// MatchAny will return true if any supplied Match functions are true
func MatchAny(match ...MatchFunc) MatchFunc {
	return func(in string) bool {
		for _, f := range match {
			if f(in) {
				return true
			}
		}
		return false
	}
}

// MatchAll will return true if all supplied Match functions are true
func MatchAll(match ...MatchFunc) MatchFunc {
	return func(in string) bool {
		for _, f := range match {
			if !f(in) {
				return false
			}
		}
		return true
	}
}

// Count returns the number of events in a given file
func Count(path string) (int, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	buf := make([]byte, 32*1024)
	count := 0
	lineSep := []byte{'\n'}

	for {
		c, err := file.Read(buf)
		count += bytes.Count(buf[:c], lineSep)

		switch {
		case err == io.EOF:
			return count, nil

		case err != nil:
			return count, err
		}
	}
}
