package chacha12blake3

import (
	"crypto/cipher"
	"crypto/subtle"
	"encoding/binary"
	"errors"

	"github.com/skerkour/stdx-go/crypto/blake3"
	"github.com/skerkour/stdx-go/crypto/chacha"
)

const (
	KeySize   = 32
	NonceSize = 32
	TagSize   = 32

	encryptionKeyContext     = "ChaCha12-BLAKE3 encryption key"
	authenticationKeyContext = "ChaCha12-BLAKE3 authentication key"
)

var (
	ErrOpen           = errors.New("chacha12blake3: error decrypting ciphertext")
	ErrBadKeyLength   = errors.New("chacha12blake3: bad key length for ChaCha12-BLAKE3. 32 bytes required")
	ErrBadNonceLength = errors.New("chacha12blake3: bad nonce length for ChaCha12-BLAKE3. 32 bytes required")
)

type ChaCha12Blake3 struct {
	key           [KeySize]byte
	encryptionKey [chacha.KeySize]byte
}

// ensure that ChaCha12Blake3 implements `cipher.AEAD` interface at build time
var _ cipher.AEAD = (*ChaCha12Blake3)(nil)

func New(key []byte) (*ChaCha12Blake3, error) {
	if len(key) != KeySize {
		return nil, ErrBadKeyLength
	}

	var ret ChaCha12Blake3
	copy(ret.key[:], key)

	// pre-compute the encryption key
	blake3.DeriveKey(ret.encryptionKey[:], encryptionKeyContext, key)

	return &ret, nil
}

func (*ChaCha12Blake3) NonceSize() int {
	return NonceSize
}

func (*ChaCha12Blake3) Overhead() int {
	return TagSize
}

func (cipher *ChaCha12Blake3) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	var authenticationKey [32]byte

	if len(nonce) != NonceSize {
		panic(ErrBadNonceLength)
	}

	var authenticationKeyMaterial [KeySize + NonceSize]byte
	copy(authenticationKeyMaterial[:], nonce)
	copy(authenticationKeyMaterial[len(nonce):], cipher.key[:])
	blake3.DeriveKey(authenticationKey[:], authenticationKeyContext, authenticationKeyMaterial[:])

	ret, out := sliceForAppend(dst, len(plaintext)+TagSize)
	ciphertext, tag := out[:len(plaintext)], out[len(plaintext):]

	chacha12Cipher, _ := chacha.NewCipher(nonce[:8], cipher.encryptionKey[:], 12)
	chacha12Cipher.XORKeyStream(ciphertext, plaintext)

	macHasher := blake3.New(32, authenticationKey[:])
	macHasher.Write(additionalData)
	writeUint64LittleEndian(macHasher, uint64(len(additionalData)))
	// macHasher.Write(nonce)
	macHasher.Write(ciphertext)
	// writeUint64(macHasher, uint64(len(nonce)))
	writeUint64LittleEndian(macHasher, uint64(len(ciphertext)))
	macHasher.Sum(tag[:0])

	zeroize(authenticationKey[:])

	return ret
}

func (cipher *ChaCha12Blake3) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	var authenticationKey [32]byte

	if len(nonce) != NonceSize {
		panic(ErrBadNonceLength)
	}

	var authenticationKeyMaterial [KeySize + NonceSize]byte
	copy(authenticationKeyMaterial[:], nonce)
	copy(authenticationKeyMaterial[len(nonce):], cipher.key[:])
	blake3.DeriveKey(authenticationKey[:], authenticationKeyContext, authenticationKeyMaterial[:])

	tag := ciphertext[len(ciphertext)-TagSize:]
	ciphertext = ciphertext[:len(ciphertext)-TagSize]

	var computedTag [TagSize]byte
	macHasher := blake3.New(32, authenticationKey[:])
	macHasher.Write(additionalData)
	writeUint64LittleEndian(macHasher, uint64(len(additionalData)))
	macHasher.Write(ciphertext)
	writeUint64LittleEndian(macHasher, uint64(len(ciphertext)))
	macHasher.Sum(computedTag[:0])

	if subtle.ConstantTimeCompare(computedTag[:], tag) != 1 {
		return nil, ErrOpen
	}

	ret, plaintext := sliceForAppend(dst, len(ciphertext))

	chacha12Cipher, _ := chacha.NewCipher(nonce[:8], cipher.encryptionKey[:], 12)
	chacha12Cipher.XORKeyStream(plaintext, ciphertext)

	zeroize(authenticationKey[:])

	return ret, nil
}

func (cipher *ChaCha12Blake3) Zeroize() {
	zeroize(cipher.key[:])
	zeroize(cipher.encryptionKey[:])
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

func writeUint64LittleEndian(p *blake3.Hasher, n uint64) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], n)
	p.Write(buf[:])
}

func zeroize(input []byte) {
	for i := range input {
		input[i] = 0
	}
}
