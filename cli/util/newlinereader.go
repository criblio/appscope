package util

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// NewlineReader reads from an io.Reader, matches against a given callback, and calls a callback with the line number and bytes
func NewlineReader(r io.Reader, match func(string) bool, callback func(line int, offset int64, b []byte) error) (int, error) {
	cr := &CountingReader{Reader: r}
	scanner := bufio.NewScanner(cr)
	idx := 0
	offset := int64(0)
	for scanner.Scan() {
		idx++
		if match(scanner.Text()) {
			err := callback(idx, offset, scanner.Bytes())
			if err != nil {
				return cr.BytesRead, err
			}
		}
		offset += int64(len(scanner.Bytes()) + 1) // bad assumption here with newline length 1 but lazy right now
	}

	if err := scanner.Err(); err != nil {
		return cr.BytesRead, err
	}
	return cr.BytesRead, nil
}

// FindReverseLineMatchOffset finds the passed number of lines from the end of the passed reader
func FindReverseLineMatchOffset(lines int, r io.ReadSeeker, match func(string) bool) (offset int64, err error) {
	found := 0

	// Move to the end of the file
	offset, err = r.Seek(0, io.SeekEnd)
	if err != nil {
		return
	}

	// Read 102400 byte buffers, backwards from the end of the file, counting lines which match a given match function
	buf := make([]byte, 102400)
	last := false
	for { // Each iteration reads 102400 bytes (or as much as it can) and returns if we've found the number of matches or if we're at the beginning of the file
		offset -= int64(cap(buf))
		if offset < 0 {
			offset = 0
			last = true
		}
		_, err = r.Seek(offset, io.SeekStart)
		if err != nil {
			offset = -1
			return
		}
		var rBytes int
		rBytes, err = r.Read(buf)
		if err != nil {
			offset = -1
			return
		}

		lastMatch := rBytes - 1               // Only search as much as we've read, -1 for offset 0
		for i := lastMatch - 2; i >= 0; i-- { // Start 2 characters in, skip last newline or if single char not newline won't match either
			if buf[i] == '\n' {
				line := string(buf[i+1 : lastMatch])
				lastMatch = i
				// fmt.Fprintf(os.Stderr, "%d line: %s\n", i, line)
				if match(line) {
					// fmt.Fprintf(os.Stderr, "found!\n")
					found++
					if found == lines {
						offset += int64(i) + int64(1) // Return character after the last match plus what we haven't yet read in the file
						return
					}
				}
			}
		}

		if rBytes != cap(buf) || last { // we couldn't read a full buffer, so we're at the beginning. Exit indicating offset -1 which means we didn't find.
			offset = -1
			return
		}

		offset += int64(lastMatch) // Set offset to last matched newline and start from there to avoid matching against broken lines
	}
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
		return strings.Contains(in, search)
	}
}

// MatchField searches for a given JSON key/value
func MatchField(key string, val interface{}) MatchFunc {
	strval := ""
	switch res := val.(type) {
	case float64:
		strval = strconv.FormatFloat(res, 'f', -1, 64)
	case float32:
		strval = strconv.FormatFloat(val.(float64), 'f', -1, 32)
	case int64, int32, int:
		strval = fmt.Sprintf("%d", val)
	case string:
		strval = fmt.Sprintf("\"%s", res)
	default:
		strval = fmt.Sprintf("%v", res)
	}
	search := fmt.Sprintf("%s\":%s", key, strval)
	return func(in string) bool {
		// fmt.Printf("in: %s\n", in)
		return strings.Contains(in, search)
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
