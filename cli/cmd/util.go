package cmd

import (
	"github.com/criblio/scope/ps"
	"github.com/criblio/scope/util"
)

// sessionByID returns a session by ID, or if -1 (not set) returns last session
func sessionByID(id int) ps.SessionList {
	var sessions ps.SessionList
	if id == -1 {
		sessions = ps.GetSessions().Last(1)
	} else {
		sessions = ps.GetSessions().ID(id)
	}
	sessionCount := len(sessions)
	if sessionCount != 1 {
		util.ErrAndExit("error expected a single session, saw: %d", sessionCount)
	}
	return sessions
}
