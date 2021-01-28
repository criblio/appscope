package util

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func getLines(lines int) string {
	ret := ""
	strlen := len(fmt.Sprintf("%d", lines))
	fmtstr := fmt.Sprintf("%%0%dd\n", strlen)
	for i := 1; i <= lines; i++ {
		ret += fmt.Sprintf(fmtstr, i)
	}
	return ret
}

func TestFindReverseLineMatchOffset(t *testing.T) {
	lines := getLines(21)
	// fmt.Printf("%s\n", lines)

	r := strings.NewReader(lines)

	offset, err := FindReverseLineMatchOffset(2, r, MatchAlways)
	assert.NoError(t, err)
	assert.Equal(t, int64(3*19), offset)

	offset, err = FindReverseLineMatchOffset(4, r, MatchString("2"))
	assert.NoError(t, err)
	assert.Equal(t, int64(3), offset) // Should be line 2, offset 3

	lines = getLines(260)
	r = strings.NewReader(lines)

	offset, err = FindReverseLineMatchOffset(10, r, MatchAlways)
	assert.NoError(t, err)
	assert.Equal(t, int64((260-10)*4), offset)

	offset, err = FindReverseLineMatchOffset(250, r, MatchAlways)
	assert.NoError(t, err)
	assert.Equal(t, int64((260-250)*4), offset)
}
