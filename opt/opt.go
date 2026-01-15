// package opt provides optional types. It can be useful for optional configuration paramteters
package opt

import "time"

type Option[Value any] struct {
	value  Value
	exists bool
}

func None[V any]() Option[V] {
	return Option[V]{
		exists: false,
	}
}

func Some[V any](value V) Option[V] {
	return Option[V]{
		value:  value,
		exists: true,
	}
}

func (o Option[Value]) Get() (Value, bool) {
	return o.value, o.exists
}

// Ptr returns a pointer to the given value. It's useful for function or structs accepting optional
// parameters as pointers.
func Ptr[T any](v T) *T {
	return &v
}

func String(str string) *string {
	return Ptr(str)
}

func Int(i int) *int {
	return Ptr(i)
}

func Int8(i int8) *int8 {
	return Ptr(i)
}

func Int16(i int16) *int16 {
	return Ptr(i)
}

func Int32(i int32) *int32 {
	return Ptr(i)
}

func Int64(i int64) *int64 {
	return Ptr(i)
}

func Uint(i uint) *uint {
	return Ptr(i)
}

func Uint8(i uint8) *uint8 {
	return Ptr(i)
}

func Uint16(i uint16) *uint16 {
	return Ptr(i)
}

func Uint32(i uint32) *uint32 {
	return Ptr(i)
}

func Uint64(i uint64) *uint64 {
	return Ptr(i)
}

func Time(t time.Time) *time.Time {
	return Ptr(t)
}

func Bool(v bool) *bool {
	return Ptr(v)
}
