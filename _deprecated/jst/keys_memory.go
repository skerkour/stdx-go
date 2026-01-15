package jst

type KeyProviderMemory struct {
	keys map[string][]byte
}

func NewKeyProviderMemory(keys map[string][]byte) *KeyProviderMemory {
	return &KeyProviderMemory{
		keys: keys,
	}
}

func (provider *KeyProviderMemory) GetKey(keyId string) (key []byte, err error) {
	key, exists := provider.keys[keyId]
	if !exists {
		err = ErrKeyNotFound(keyId)
		return
	}

	return
}
