package jst

import "fmt"

var (
	ErrKeyNotFound = func(keyId string) error {
		return fmt.Errorf("key (%s) not found", keyId)
	}
)

type KeyProvider interface {
	GetKey(keyId string) (key []byte, err error)
}
