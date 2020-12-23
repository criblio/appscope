package internal

// GitSummary is produced by govvv and stores tag, commit and branch status
var GitSummary string

// BuildDate is set by govvv and stores when we were built
var BuildDate string

// SetVersionInfo is called by main and brings the info from govvv to internal
func SetVersionInfo(gitSummary string, buildDate string) {
	GitSummary = gitSummary
	BuildDate = buildDate
}

// GetGitSummary returns a string in tag-commit-(dirty|clean) format
func GetGitSummary() string {
	return GitSummary
}

// GetBuildDate returns when we were built
func GetBuildDate() string {
	return BuildDate
}