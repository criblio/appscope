package util

import (
	"bytes"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPrintObjWithMap(t *testing.T) {
	buf := &bytes.Buffer{}
	setOut(buf)
	obj := map[string]interface{}{
		"field1": "value1",
		"field2": "value2",
		"field3": "value3",
		"nested": map[string]interface{}{
			"nfield1": "value4",
			"nfield2": map[string]interface{}{
				"nnfield1": "value5",
			},
		},
		"field4": "value6",
		"field5": []string{"foo", "bar"},
	}
	fields := []ObjField{
		{Name: "Field 1", Field: "field1"},
		{Name: "field2", Field: "field2", Transform: func(val interface{}) string { return strings.ToUpper(fmt.Sprintf("%v", val)) }},
		{Name: "Nested", Field: "nested"},
		{Name: "field4"},
		{Name: "field5"},
	}
	PrintObj(fields, obj)

	// os.Stdout.Write(buf.Bytes())

	rows := strings.Split(string(buf.Bytes()), "\n")

	// First row contains capital Field 1 and value1
	assert.True(t, strings.Contains(rows[0], "Field 1"))
	assert.True(t, strings.Contains(rows[0], "value1"))

	// Second row has been transformed!
	assert.True(t, strings.Contains(rows[1], "field2"))
	assert.True(t, strings.Contains(rows[1], "VALUE2"))

	// Field3 is not in the results
	assert.False(t, bytes.Contains(buf.Bytes(), []byte("field3")))
	assert.False(t, bytes.Contains(buf.Bytes(), []byte("value3")))

	// Remaining fields are nested
	assert.True(t, strings.Contains(rows[2], "Nested"))
	assert.True(t, strings.Contains(rows[3], "  nfield1"))
	assert.True(t, strings.Contains(rows[4], "  nfield2"))
	assert.True(t, strings.Contains(rows[5], "    nnfield1"))

	// We can leave field blank
	assert.True(t, strings.Contains(rows[6], "field4"))
	assert.True(t, strings.Contains(rows[6], "value6"))

	// String Slice
	assert.True(t, strings.Contains(rows[7], "field5"))
	assert.True(t, strings.Contains(rows[8], "foo"))
	assert.True(t, strings.Contains(rows[9], "bar"))
}

func TestPrintObjWithStruct(t *testing.T) {
	buf := &bytes.Buffer{}
	setOut(buf)
	type printObj struct {
		Field1 string                 `json:"field1,omitempty"`
		Field2 string                 `json:"field2"`
		Field3 string                 `json:"field3"`
		Nested map[string]interface{} `json:"nested"`
		Field4 int                    `json:"field4"`
		Field5 []string               `json:"field5"`
	}
	obj := printObj{
		Field1: "value1",
		Field2: "value2",
		Field3: "value3",
		Nested: map[string]interface{}{
			"nfield1": "value4",
			"nfield2": map[string]interface{}{
				"nnfield1": "value5",
			},
		},
		Field4: 6,
		Field5: []string{"foo", "bar"},
	}
	fields := []ObjField{
		{Name: "Field 1", Field: "field1"},
		{Name: "field2", Field: "field2", Transform: func(val interface{}) string { return strings.ToUpper(fmt.Sprintf("%v", val)) }},
		{Name: "Nested", Field: "nested"},
		{Name: "field4"},
		{Name: "field5"},
	}
	PrintObj(fields, obj)

	// os.Stdout.Write(buf.Bytes())

	rows := strings.Split(string(buf.Bytes()), "\n")

	// First row contains capital Field 1 and value1
	assert.True(t, strings.Contains(rows[0], "Field 1"))
	assert.True(t, strings.Contains(rows[0], "value1"))

	// Second row has been transformed!
	assert.True(t, strings.Contains(rows[1], "field2"))
	assert.True(t, strings.Contains(rows[1], "VALUE2"))

	// Field3 is not in the results
	assert.False(t, bytes.Contains(buf.Bytes(), []byte("field3")))
	assert.False(t, bytes.Contains(buf.Bytes(), []byte("value3")))

	// Remaining fields are nested
	assert.True(t, strings.Contains(rows[2], "Nested"))
	if strings.Contains(rows[3], "  nfield1") {
		assert.True(t, strings.Contains(rows[3], "  nfield1"))
		assert.True(t, strings.Contains(rows[4], "  nfield2"))
		assert.True(t, strings.Contains(rows[5], "    nnfield1"))
	} else {
		assert.True(t, strings.Contains(rows[3], "  nfield2"))
		assert.True(t, strings.Contains(rows[4], "    nnfield1"))
		assert.True(t, strings.Contains(rows[5], "  nfield1"))
	}

	// We can leave field blank
	assert.True(t, strings.Contains(rows[6], "field4"))
	assert.True(t, strings.Contains(rows[6], "6"))

	// String Slice
	assert.True(t, strings.Contains(rows[7], "field5"))
	assert.True(t, strings.Contains(rows[8], "foo"))
	assert.True(t, strings.Contains(rows[9], "bar"))
}

func TestPrintObjSliceMap(t *testing.T) {
	buf := &bytes.Buffer{}
	setOut(buf)
	obj := []map[string]interface{}{
		{
			"field1": "value1",
			"field2": "value2",
			"field3": "value3",
			"field4": "4",
		},
		{
			"field1": "2value1",
			"field2": "2value2",
			"field3": "2value3",
			"field4": "24",
		},
	}
	fields := []ObjField{
		{Name: "Field 1", Field: "field1"},
		{Name: "field2", Field: "field2", Transform: func(val interface{}) string { return strings.ToUpper(fmt.Sprintf("%v", val)) }},
		{Name: "field4"},
	}
	PrintObj(fields, obj)

	// os.Stdout.Write(buf.Bytes())

	rows := strings.Split(string(buf.Bytes()), "\n")

	// First row contains headers
	assert.True(t, strings.Contains(rows[0], "Field 1"))
	assert.True(t, strings.Contains(rows[0], "field2"))
	assert.True(t, strings.Contains(rows[0], "field4"))
	assert.False(t, strings.Contains(rows[0], "field3"))

	// Second row contains row divider
	assert.True(t, strings.Contains(rows[1], "--"))

	// Third row is the first row of the table
	assert.True(t, strings.Contains(rows[2], "value1"))
	assert.True(t, strings.Contains(rows[2], "VALUE2"))
	assert.True(t, strings.Contains(rows[2], "4"))
	assert.False(t, strings.Contains(rows[2], "value3"))

	// Fourth row is the second row of the table
	assert.True(t, strings.Contains(rows[3], "2value1"))
	assert.True(t, strings.Contains(rows[3], "2VALUE2"))
	assert.True(t, strings.Contains(rows[3], "24"))
	assert.False(t, strings.Contains(rows[3], "2value3"))
}
