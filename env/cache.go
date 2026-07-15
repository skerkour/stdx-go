package env

import (
	"reflect"
	"strings"
	"sync"
	"time"
)

type fieldInfo struct {
	key        string
	ignore     bool
	required   bool
	defaultVal string
}

type cache struct {
	mu     sync.RWMutex
	fields map[reflect.Type][]fieldInfo
	conv   map[reflect.Type]converter
}

type converter func(string) (reflect.Value, error)

func newCache() *cache {
	return &cache{
		fields: make(map[reflect.Type][]fieldInfo),
		conv:   make(map[reflect.Type]converter),
	}
}

func (c *cache) fieldInfo(field reflect.StructField) fieldInfo {
	info := fieldInfo{key: field.Name}

	tag, ok := field.Tag.Lookup("env")
	if !ok {
		return info
	}

	parts := strings.Split(tag, ",")
	info.key = parts[0]

	if info.key == "-" {
		info.ignore = true
		return info
	}

	for _, opt := range parts[1:] {
		opt = strings.TrimSpace(opt)
		switch {
		case opt == "required":
			info.required = true
		case strings.HasPrefix(opt, "default="):
			info.defaultVal = opt[len("default="):]
		}
	}

	return info
}

func (c *cache) converter(t reflect.Type) converter {
	c.mu.RLock()
	conv := c.conv[t]
	c.mu.RUnlock()
	if conv != nil {
		return conv
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if conv, ok := c.conv[t]; ok {
		return conv
	}

	conv = builtinConverter(t)
	if conv != nil {
		c.conv[t] = conv
	}
	return conv
}

func builtinConverter(t reflect.Type) converter {
	switch t.Kind() {
	case reflect.String:
		return func(v string) (reflect.Value, error) {
			return reflect.ValueOf(v), nil
		}
	case reflect.Bool:
		return func(v string) (reflect.Value, error) {
			switch v {
			case "1", "true", "TRUE", "True", "yes", "YES", "Yes", "on", "ON", "On":
				return reflect.ValueOf(true), nil
			case "0", "false", "FALSE", "False", "no", "NO", "No", "off", "OFF", "Off":
				return reflect.ValueOf(false), nil
			default:
				return reflect.Value{}, parseError(t, v)
			}
		}
	case reflect.Int:
		return intConverter[int](64, 0)
	case reflect.Int8:
		return intConverter[int8](64, 8)
	case reflect.Int16:
		return intConverter[int16](64, 16)
	case reflect.Int32:
		return intConverter[int32](64, 32)
	case reflect.Int64:
		if t == durationType {
			return func(v string) (reflect.Value, error) {
				d, err := time.ParseDuration(v)
				if err != nil {
					return reflect.Value{}, parseError(t, v)
				}
				return reflect.ValueOf(d), nil
			}
		}
		return intConverter[int64](64, 64)
	case reflect.Uint:
		return uintConverter[uint](64, 0)
	case reflect.Uint8:
		return uintConverter[uint8](64, 8)
	case reflect.Uint16:
		return uintConverter[uint16](64, 16)
	case reflect.Uint32:
		return uintConverter[uint32](64, 32)
	case reflect.Uint64:
		return uintConverter[uint64](64, 64)
	case reflect.Float32:
		return floatConverter[float32](32)
	case reflect.Float64:
		return floatConverter[float64](64)
	}
	return nil
}

func parseError(t reflect.Type, v string) error {
	return &strconvErr{typ: t, val: v}
}

type strconvErr struct {
	typ reflect.Type
	val string
}

func (e *strconvErr) Error() string {
	return "cannot parse " + e.val + " as " + e.typ.String()
}

func intConverter[T intVal](bitSize int, convSize int) converter {
	return func(v string) (reflect.Value, error) {
		n, err := parseInt(v, bitSize)
		if err != nil {
			return reflect.Value{}, parseError(reflect.TypeOf(T(0)), v)
		}
		return reflect.ValueOf(T(n)), nil
	}
}

func uintConverter[T uintVal](bitSize int, convSize int) converter {
	return func(v string) (reflect.Value, error) {
		n, err := parseUint(v, bitSize)
		if err != nil {
			return reflect.Value{}, parseError(reflect.TypeOf(T(0)), v)
		}
		return reflect.ValueOf(T(n)), nil
	}
}

func floatConverter[T floatVal](bitSize int) converter {
	return func(v string) (reflect.Value, error) {
		n, err := parseFloat(v, bitSize)
		if err != nil {
			return reflect.Value{}, parseError(reflect.TypeOf(T(0)), v)
		}
		return reflect.ValueOf(T(n)), nil
	}
}

type intVal interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type uintVal interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

type floatVal interface {
	~float32 | ~float64
}

var durationType = reflect.TypeOf(time.Duration(0))

func isBuiltinStruct(t reflect.Type) bool {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	return t == durationType
}
