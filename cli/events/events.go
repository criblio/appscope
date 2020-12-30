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

// Reader reads a newline delimited JSON document and sends parsed documents
// to the passed out channel. It exits the process on error.
func Reader(path string, match func(string) bool, out chan map[string]interface{}) {
	file, err := os.Open(path)
	util.CheckErrSprintf(err, "error opening events file %s: %v", path, err)
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		event := map[string]interface{}{}
		if !match(scanner.Text()) {
			continue
		}
		err := json.Unmarshal(scanner.Bytes(), &event)
		util.CheckErrSprintf(err, "error decoding JSON %s\nerr: %v", scanner.Text(), err)
		if eventType, typeExists := event["type"]; typeExists && eventType.(string) == "evt" {
			out <- event["body"].(map[string]interface{})
		}
	}

	if err := scanner.Err(); err != nil {
		util.ErrAndExit("error scanning events file %s: %v", path, err)
	}
	close(out)
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
func Count(path string) int {
	file, err := os.Open(path)
	util.CheckErrSprintf(err, "error opening events file %s: %v", path, err)
	defer file.Close()

	buf := make([]byte, 32*1024)
	count := 0
	lineSep := []byte{'\n'}

	for {
		c, err := file.Read(buf)
		count += bytes.Count(buf[:c], lineSep)

		switch {
		case err == io.EOF:
			return count

		case err != nil:
			util.ErrAndExit("unexpected error counting events: %v", err)
		}
	}
}
