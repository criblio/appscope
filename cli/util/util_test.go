package util

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestScopeHome(t *testing.T) {
	homeBak := os.Getenv("HOME")
	scopeHomeBak := os.Getenv("SCOPE_HOME")
	tmpDirBak := os.Getenv("TMPDIR")

	os.Setenv("SCOPE_HOME", "/foo")
	ret := ScopeHome()
	assert.Equal(t, "/foo", ret)
	scopeHome = ""
	os.Unsetenv("SCOPE_HOME")

	os.Setenv("HOME", "/foo2")
	ret = ScopeHome()
	assert.Equal(t, "/foo2/.scope", ret)
	scopeHome = ""
	os.Unsetenv("HOME")

	os.Setenv("TMPDIR", "/foo3")
	ret = ScopeHome()
	assert.Equal(t, "/foo3/.scope", ret)
	scopeHome = ""
	os.Unsetenv("TMPDIR")

	ret = ScopeHome()
	assert.Equal(t, "/tmp/.scope", ret)
	scopeHome = ""

	os.Setenv("HOME", homeBak)
	os.Setenv("SCOPE_HOME", scopeHomeBak)
	os.Setenv("TMPDIR", tmpDirBak)
}

func TestGetAgo(t *testing.T) {
	ago := GetAgo(time.Now().Add(-1 * time.Second))
	assert.Equal(t, "1s", ago)
	ago = GetAgo(time.Now().Add((-1 * time.Minute) + (-1 * time.Second)))
	assert.Equal(t, "1m1s", ago)
	ago = GetAgo(time.Now().Add((-1 * time.Hour) + (-1 * time.Minute)))
	assert.Equal(t, "1h1m", ago)
	ago = GetAgo(time.Now().Add((-24 * time.Hour) + (-1 * time.Hour)))
	assert.Equal(t, "1d1h", ago)
	ago = GetAgo(time.Now().Add((-24 * 7 * time.Hour) + (-1 * time.Hour)))
	assert.Equal(t, "7d", ago)
}
