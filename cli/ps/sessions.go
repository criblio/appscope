package ps

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"

	"github.com/criblio/scope/run"
	"github.com/criblio/scope/util"
)

// Session represents a scoped run
type Session struct {
	ID        int      `json:"id"`
	Cmd       string   `json:"cmd"`
	Pid       int      `json:"pid"`
	Timestamp int64    `json:"timestamp"`
	WorkDir   string   `json:"workDir"`
	Args      []string `json:"args"`

	ArgsPath    string `json:"argspath"`
	MetricsPath string `json:"metricspath"`
	EventsPath  string `json:"eventspath"`
	CmdDirPath  string `json:"cmddirpath"`
}

// SessionList represents a list of sessions
type SessionList []Session

// GetSessions returns all sessions from history
func GetSessions() (ret SessionList) {
	histDir := run.HistoryDir()
	files, err := ioutil.ReadDir(histDir)
	util.CheckErrSprintf(err, "No prior sessions found. Have you tried `scope run`?\nerror listing: %v", err)
	re := regexp.MustCompile(`([^_]+)_(\d+)_(\d+)_(\d+)`)
	for _, f := range files {
		vals := re.FindStringSubmatch(f.Name())
		id, _ := strconv.Atoi(vals[2])
		pid, _ := strconv.Atoi(vals[3])
		ts, _ := strconv.ParseInt(vals[4], 10, 64)

		workDir := filepath.Join(histDir, f.Name())

		ret = append(ret, Session{
			ID:          id,
			Cmd:         vals[1],
			Pid:         pid,
			Timestamp:   ts,
			WorkDir:     workDir,
			ArgsPath:    filepath.Join(workDir, "args.json"),
			CmdDirPath:  filepath.Join(workDir, "cmd"),
			EventsPath:  filepath.Join(workDir, "events.json"),
			MetricsPath: filepath.Join(workDir, "metrics.dogstatsd"),
		})
	}
	return ret
}

// Last returns the last n sessions by time
func (sessions SessionList) Last(n int) (ret SessionList) {
	sort.Slice(sessions, func(i, j int) bool { return sessions[i].Timestamp > sessions[j].Timestamp })
	if n > 0 {
		for i := 0; i < n; i++ {
			ret = append(ret, sessions[i])
		}
	} else {
		ret = append(ret, sessions...)
	}
	return ret
}

// Running returns sessions which are current still running
func (sessions SessionList) Running() (ret SessionList) {
	for _, s := range sessions {
		if util.CheckFileExists(fmt.Sprintf("/proc/%d", s.Pid)) {
			ret = append(ret, s)
		}
	}
	return ret
}

// Args gets command arguments
func (sessions SessionList) Args() (ret SessionList) {
	for _, s := range sessions {
		rawBytes, err := ioutil.ReadFile(filepath.Join(s.WorkDir, "args.json"))
		if err == nil {
			json.Unmarshal(rawBytes, &s.Args)
		}
		ret = append(ret, s)
	}
	return ret
}

// ID returns only matching session ID (expected one)
func (sessions SessionList) ID(id int) (ret SessionList) {
	for _, s := range sessions {
		if s.ID == id {
			ret = append(ret, s)
		}
	}
	return ret
}
