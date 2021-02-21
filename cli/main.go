package main

import (
	"github.com/criblio/scope/cmd"
	"github.com/criblio/scope/internal"
)

// GitBranch is set by govvv and represents the branch we were built on
var GitBranch string

// GitSummary is produced by govvv and stores tag, commit and branch status
var GitSummary string

// BuildDate is set by govvv and stores when we were built
var BuildDate string

// Version is set by govvv and stores the version at build time
var Version string

func main() {
	internal.SetVersionInfo(GitSummary, BuildDate, Version)
	cmd.Execute()
}
