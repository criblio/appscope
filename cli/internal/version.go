package internal

// GitSummary is produced by govvv and stores tag, commit and branch status
var GitSummary string

// BuildDate is set by govvv and stores when we were built
var BuildDate string

// Version is set by govvv and stores the version at build time
var Version string

// SetVersionInfo is called by main and brings the info from govvv to internal
func SetVersionInfo(gitSummary string, buildDate string, version string) {
	GitSummary = gitSummary
	BuildDate = buildDate
	Version = version
}

// GetGitSummary returns a string in tag-commit-(dirty|clean) format
func GetGitSummary() string {
	return GitSummary
}

// GetBuildDate returns when we were built
func GetBuildDate() string {
	return BuildDate
}

// GetVersion returns our version
func GetVersion() string {
	return Version
}

// GetNormalizedVersion returns version for official version or "dev"
func GetNormalizedVersion() string {
	if !IsVersionDev() {
		return Version
	}
	return "dev"
}

// IsVersionDev returns TRUE if used version is a developer version
func IsVersionDev() bool {
	return GitSummary[1:] != Version
}
