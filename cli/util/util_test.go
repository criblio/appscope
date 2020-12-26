package util

import (
	"os"
	"testing"

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
