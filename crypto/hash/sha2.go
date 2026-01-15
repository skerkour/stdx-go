package hash

import (
	"crypto/sha512"
	"hash"

	"crypto/sha256"
)

// NewSha256 returns a new `hash.Hash` computing the SHA2-256 checksum.
func NewSha256() hash.Hash {
	return sha256.New()
}

// Sha256 returns the SHA2-256 checksum of the data.
func Sha256(data []byte) []byte {
	sum := sha256.Sum256(data)
	return sum[:]
}

// NewSha512 returns a new `hash.Hash` computing the SHA2-512 checksum.
func NewSha512() hash.Hash {
	return sha512.New()
}

// SHA512 returns the SHA2-512 checksum of the data.
func SHA512(data []byte) []byte {
	sum := sha512.Sum512(data)
	return sum[:]
}
