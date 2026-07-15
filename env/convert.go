package env

import (
	"reflect"
	"strconv"
	"strings"
)

func parseInt(v string, bitSize int) (int64, error) {
	return strconv.ParseInt(v, 10, bitSize)
}

func parseUint(v string, bitSize int) (uint64, error) {
	return strconv.ParseUint(v, 10, bitSize)
}

func parseFloat(v string, bitSize int) (float64, error) {
	return strconv.ParseFloat(v, bitSize)
}

func setSlice(fieldVal reflect.Value, val string) error {
	elemType := fieldVal.Type().Elem()
	isPtr := elemType.Kind() == reflect.Ptr
	if isPtr {
		elemType = elemType.Elem()
	}

	parts := splitSliceValue(val)
	slice := reflect.MakeSlice(fieldVal.Type(), 0, len(parts))

	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}

		var elem reflect.Value
		if elemType.Kind() == reflect.Struct && !isBuiltinStruct(elemType) {
			return nil
		}

		conv := builtinConverter(elemType)
		if conv != nil {
			converted, err := conv(part)
			if err != nil {
				return err
			}
			elem = converted
		} else if elemType.Kind() == reflect.String {
			elem = reflect.ValueOf(part)
		} else {
			return nil
		}

		if isPtr {
			ptr := reflect.New(elemType)
			ptr.Elem().Set(elem)
			slice = reflect.Append(slice, ptr)
		} else {
			if elem.Type() != elemType {
				elem = elem.Convert(elemType)
			}
			slice = reflect.Append(slice, elem)
		}
	}

	fieldVal.Set(slice)
	return nil
}

func splitSliceValue(v string) []string {
	if v == "" {
		return nil
	}
	return strings.Split(v, ",")
}
