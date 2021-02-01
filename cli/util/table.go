package util

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"sort"
	"strings"

	"github.com/fatih/color"
	"github.com/olekukonko/tablewriter"
)

var out io.Writer

func init() {
	out = os.Stdout
}

// SetOut is used by unit tests to change where we're writing
func SetOut(newOut io.Writer) {
	out = newOut
}

func printf(format string, a ...interface{}) (int, error) {
	return fmt.Fprintf(out, format, a...)
}

// ObjField references a field to print.
// Name is the name of the field, which will print on the left side of table.
// Field is the field to lookup in the object.
// Transform is a function which will transform the given field value.
type ObjField struct {
	Name      string
	Field     string
	Transform func(interface{}) string
}

// PrintObj prints an object as a table. Best effort, no errors.
// fields contains a list of tuples of fields you want to print from the source object. Accepts values
// of a string, which represents the field name you want to print, usually the same as the key name,
// or a func(val interface{}) string function which accepts the field value as an input and returns a string.
//
// obj is a map[string]interface{} which contain a table to be printed in columnar format.
//
// key   value
// key   value
//
func PrintObj(fields []ObjField, obj interface{}) error {
	v := GetValue(obj)
	switch v.Kind() {
	case reflect.Slice:
		return printSlice(fields, obj)
	case reflect.Struct:
		return printObj(fields, obj)
	case reflect.Map:
		return printObj(fields, obj)
	default:
		printf("%v", obj)
	}
	return nil
}

func printSlice(fields []ObjField, obj interface{}) error {
	table := tablewriter.NewWriter(out)
	header := []string{}
	// headerColors := []tablewriter.Colors{}
	for _, f := range fields {
		header = append(header, strings.ToUpper(f.Name))
		// headerColors = append(headerColors, tablewriter.Colors{tablewriter.Bold, tablewriter.FgGreenColor})
	}
	table.SetHeader(header)
	table.SetAutoFormatHeaders(false)
	table.SetAutoWrapText(false)
	table.SetHeaderAlignment(tablewriter.ALIGN_LEFT)
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	table.SetCenterSeparator("")
	table.SetColumnSeparator("")
	table.SetRowSeparator("")
	table.SetHeaderLine(false)
	table.SetBorder(false)
	table.SetTablePadding("\t")
	table.SetNoWhiteSpace(true)
	// table.SetHeaderColor(headerColors...)
	table.SetBorder(false)
	refval := GetValue(obj)
	for i := 0; i < refval.Len(); i++ {
		row := refval.Index(i).Interface()
		r := []string{}
		for _, f := range fields {
			v := getFieldValue(row, f)
			if f.Transform != nil {
				v = f.Transform(v)
			}
			r = append(r, fmt.Sprintf("%v", v))
		}
		table.Append(r)
	}
	table.Render()
	return nil
}

func printObj(fields []ObjField, obj interface{}) error {
	table := tablewriter.NewWriter(out)
	table.SetAutoWrapText(false)
	table.SetBorders(tablewriter.Border{Left: false, Top: false, Right: false, Bottom: false})
	table.SetColumnSeparator("  ")
	table.SetCenterSeparator("  ")
	table.SetAlignment(tablewriter.ALIGN_LEFT)
	// Iterate over fields
	for _, field := range fields {
		// Lookup field pointed to by Field
		var val string
		if v := getFieldValue(obj, field); v != nil {
			// If we have a transform function set, return that, otherwise return v
			if field.Transform != nil {
				val = field.Transform(v)
			} else {
				switch v.(type) {
				case map[string]interface{}:
					subtable := indentedSubtable(v.(map[string]interface{}), "  ")
					if len(subtable) > 0 {
						table.Append([]string{color.GreenString(fmt.Sprintf("%s:", field.Name)), ""})
						for _, row := range subtable {
							if len(row) == 2 {
								table.Append([]string{color.GreenString(row[0]), row[1]})
							}
						}
					}
					continue
				case []string:
					rows := v.([]string)
					if len(rows) > 0 {
						table.Append([]string{color.GreenString(fmt.Sprintf("%s:", field.Name)), ""})
						for _, row := range rows {
							table.Append([]string{"", row})
						}
					}
					continue
				case float64:
					val = color.HiBlueString("%v", v)
				case float32:
					val = color.HiBlueString("%v", v)
				case int64:
					val = color.HiBlueString("%v", v)
				case int32:
					val = color.HiBlueString("%v", v)
				case int:
					val = color.HiBlueString("%v", v)
					val = color.HiBlueString("%v", v)
				default:
					val = fmt.Sprintf("%v", v)
				}
			}
			table.Append([]string{color.GreenString("%s:", field.Name), val})
		}
	}
	table.Render()
	return nil
}

func indentedSubtable(obj map[string]interface{}, indent string) [][]string {
	ret := [][]string{}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := obj[k]
		switch v.(type) {
		case map[string]interface{}:
			ret = append(ret, []string{fmt.Sprintf("%s%s", indent, k), ""})
			ret = append(ret, indentedSubtable(v.(map[string]interface{}), fmt.Sprintf("%s  ", indent))...)
		default:
			ret = append(ret, []string{fmt.Sprintf("%s%s", indent, k), fmt.Sprintf("%v", v)})
		}
	}
	return ret
}

func getFieldValue(obj interface{}, field ObjField) interface{} {
	f := field.Field
	if f == "" {
		f = field.Name
	}
	v := GetValue(obj)
	switch v.Kind() {
	case reflect.Struct:
		f := GetJSONField(obj, f)
		if f == nil {
			return nil
		}
		return f.Value()
	case reflect.Map:
		ret, ok := obj.(map[string]interface{})[f]
		if !ok {
			return nil
		}
		return ret
	default:
		return fmt.Sprintf("%v", obj)
	}
}
