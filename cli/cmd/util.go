package cmd

import (
	"github.com/criblio/scope/history"
	"github.com/criblio/scope/util"
)

// sessionByID returns a session by ID, or if -1 (not set) returns last session
func sessionByID(id int) history.SessionList {
	var sessions history.SessionList
	if id == -1 {
		sessions = history.GetSessions().Last(1)
	} else {
		sessions = history.GetSessions().ID(id)
	}
	sessionCount := len(sessions)
	if sessionCount != 1 {
		util.ErrAndExit("error expected a single session, saw: %d", sessionCount)
	}
	return sessions
}
