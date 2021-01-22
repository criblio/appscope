package util

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// NewlineReader reads from an io.Reader, matches against a given callback, and calls a callback with the line number and bytes
func NewlineReader(r io.Reader, match func(string) bool, callback func(line int, offset int, b []byte) error) (int, error) {
	cr := &CountingReader{Reader: r}
	scanner := bufio.NewScanner(cr)
	idx := 0
	offset := 0
	for scanner.Scan() {
		idx++
		if match(scanner.Text()) {
			err := callback(idx, offset, scanner.Bytes())
			if err != nil {
				return cr.BytesRead, err
			}
		}
		offset += len(scanner.Bytes()) + 1 // bad assumption here with newline length 1 but lazy right now
	}

	if err := scanner.Err(); err != nil {
		return cr.BytesRead, err
	}
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

// MatchSkipN will return true after N events have passed
func MatchSkipN(skip int) MatchFunc {
	idx := 0
	return func(in string) bool {
		idx++
		if skip > 0 && idx < skip {
			return false
		}
		return true
	}
}
